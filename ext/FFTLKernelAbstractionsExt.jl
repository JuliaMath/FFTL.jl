# GPU transforms through KernelAbstractions: FFTL's codelet generator in kernels that run a chain
# of global Stockham passes with large radices, each pass one launch computing its butterflies in
# workgroup (threadgroup/shared) memory. Any KA backend (Metal, CUDA, ROCm, oneAPI, CPU).
#
# Global pass (FFTL's stage formula, kernels.jl), N = R·m·s:
#   y[q + s(R p + k)] = W_N^(k p s) · Σ_j x[q + s(p + j m)] · W_R^(j k),   p < m, q < s.
# A lane is one butterfly b = q + s p; its inputs are x[b + (N/R) j], so G adjacent lanes are
# read as contiguous segments of G elements. A workgroup transforms G lanes: it keeps element j
# of lane l at l + G j in workgroup memory and runs the local Stockham chain of R (radices rs,
# start stride G). Output: for s ≥ G, y[(b0 mod s) + l + s R (b0 ÷ s) + s k] (segments of G);
# for s = 1 and m > 1, y[R (b0 + l) + k] (runs of G R) after a transpose in workgroup memory.
# Along dimension d of an array (inner size E, outer size B) the pass is the same with q-stride
# E s; outer indices add E N.
# The twiddle W_N^t is the product of TWLEV table entries: with t = Σ_i t_i 2^(i e) (digits
# t_i < 2^e, e = ⌈log2 N / TWLEV⌉), factor i is W_N^(t_i 2^(i e)) at tw[offw + i 2^e + t_i].
# Small tables stay in the L1 cache: the first pass looks up a different t in every thread.
#
# Kernels are generated per parameter set when a plan is made (Core.eval into `Kernels`) and
# called through `invokelatest`. The pass, workgroup and padding rules are measured on an Apple
# M1 Max (Metal, Float32) against Metal.jl's FFT; other GPUs work but are not tuned for.
# On CUDA a pass whose workgroup buffer exceeds the static 48 KB becomes a raw CUDA kernel with
# dynamic shared memory (see DYN_BYTES): rule generation then sees the larger capacity nmax.
module FFTLKernelAbstractionsExt

using FFTL, KernelAbstractions, LinearAlgebra
import AbstractFFTs
const KA = KernelAbstractions

# Float64 literals of the generated code as T (Metal has no Float64).
retype(ex::Expr, T) = Expr(ex.head, (retype(a, T) for a in ex.args)...)
retype(x::Float64, T) = T(x)
retype(x, T) = x

# Generated kernels live here: KernelAbstractions' macros and the codelet operations in scope.
module Kernels
  using KernelAbstractions
  import FFTL: fcadd90, fcadd270, ccmul, cconst
  # padded layout (n1, pad): logical index l at l + pad ⌊l / n1⌋
  @inline padmap(l, n1, pad) = l + pad * (l ÷ n1)
end

# Complex values per static workgroup buffer at most (32 KB of ComplexF32): the pass rules are
# measured up to this size. A device with less workgroup memory lowers it (Limits.nstat).
const NMAX = 4096

# On CUDA workgroup buffers may exceed the 32 KB of NMAX: @localmem lowers to static shared
# memory (ptxas caps it at 48 KB), but CUDA also offers dynamic shared memory with an opt-in
# limit beyond 48 KB (99 KB per block on the RTX A6000, sm_86). Such passes skip Kernel-
# Abstractions entirely: the generated kernel is a plain function using CUDA.CuDynamicShared
# Memory / sync_threads that KAPlan compiles with @cuda launch=false for the argument types it
# will see, and launches raw with a shmem kwarg (KA's backend_launch passes no launch kwargs).
# Only CUDA backends get this flavor (Limits.nmax > Limits.nstat); Metal, CPU and other
# backends keep their static buffers and launch path.
const DYN_BYTES = 96 << 10 # dynamic shared memory per CUDA workgroup at most (measured on sm_86)

# The package of a backend (CUDA, Metal, ...), loaded since the backend object exists
backend_module(backend) = Base.moduleroot(parentmodule(typeof(backend)))
const CUDA_MODULE = Ref{Module}()
cuda_mod() = (isassigned(CUDA_MODULE) || (CUDA_MODULE[] = Base.require(Base.identify_package("CUDA"))); CUDA_MODULE[])

# Hardware limits a plan is built for, in complex values of the plan's element type:
#   nstat  per static workgroup buffer (@localmem): device workgroup memory, at most NMAX
#   nmax   per workgroup with CUDA dynamic shared memory (opt-in limit, at most DYN_BYTES);
#          nstat on other backends
#   wmax   threads per workgroup
#   slc    bytes of the system level cache (Apple GPUs; the :apple rules use it)
struct Limits
  nstat::Int
  nmax::Int
  wmax::Int
  slc::Int
end
const LIM0 = Limits(NMAX, NMAX, 1024, 48 << 20)   # M1 Max, Float32

# System level cache of Apple chips by name (not reported by Metal). The M1 Max value is the one
# the :apple rules are measured with; the others follow the published tiers of the M1/M2
# generations (base 8 MB, Pro 24 MB, Max 48 MB, Ultra 96 MB) and are not measured.
function apple_slc(name)
  occursin("Ultra", name) ? 96 << 20 : occursin("Max", name) ? 48 << 20 :
  occursin("Pro", name) ? 24 << 20 : 8 << 20
end

# (workgroup memory bytes, dynamic shared memory bytes, threads per workgroup, SLC bytes) of the
# backend's current device; `nothing` where the backend is not known (then LIM0's values apply).
function device_info(backend)
  name = nameof(typeof(backend))
  M = backend_module(backend)
  if name === :CUDABackend
    CUDA_MODULE[] = M
    dev = M.device()
    attr(a) = Int(M.attribute(dev, getfield(M, Symbol(:DEVICE_ATTRIBUTE_, a))))
    return (attr(:MAX_SHARED_MEMORY_PER_BLOCK), min(attr(:MAX_SHARED_MEMORY_PER_BLOCK_OPTIN), DYN_BYTES),
            attr(:MAX_THREADS_PER_BLOCK), 0)
  elseif name === :MetalBackend
    dev = M.device()
    return (Int(dev.maxThreadgroupMemoryLength), 0, Int(dev.maxThreadsPerThreadgroup.width), apple_slc(String(dev.name)))
  end
  nothing
end

function device_limits(backend, ::Type{T}) where {T}
  info = try device_info(backend) catch; nothing end
  info === nothing && return LIM0   # other backends (CPU, ROCm, oneAPI): the M1 values
  tg, dyn, wmax, slc = info
  nstat = min(NMAX, tg ÷ sizeof(Complex{T}))
  Limits(nstat, max(nstat, dyn ÷ sizeof(Complex{T})), min(wmax, 1024), slc > 0 ? slc : LIM0.slc)
end

# CUDA needs to be in scope in Kernels for the dyn flavor; the compile helper is installed once
function cuda_support!()
  isdefined(Kernels, :CUDA) || Core.eval(Kernels, :(const CUDA = $(cuda_mod())))
  isdefined(Kernels, :cuda_dyn_compile) || Core.eval(Kernels, quote
    # bounds of the thread and block index as KernelAbstractions states them: LLVM then keeps
    # the index arithmetic in unsigned 32 bit (latency-bound passes with few blocks, 65536: 20 %)
    @inline assume(c::Bool) = Base.llvmcall(("""
      declare void @llvm.assume(i1)
      define void @entry(i8 %c) #0 {
        %b = trunc i8 %c to i1
        call void @llvm.assume(i1 %b)
        ret void
      }
      attributes #0 = { alwaysinline }""", "entry"), Cvoid, Tuple{Bool}, c)
    # compile a raw pass kernel for these argument types; with dynamic shared memory (bytes > 0)
    # raise the function's > 48 KB opt-in ceiling before it is ever launched, with static shared
    # memory set the launch bounds (maxthreads) as KernelAbstractions does for a static workgroup
    function cuda_dyn_compile(f, y, x, tw, cin, cout, bytes, maxthreads = 0)
      k = maxthreads > 0 ? CUDA.@cuda(launch=false, maxthreads=maxthreads, f(y, x, tw, cin, cout)) :
                           CUDA.@cuda(launch=false, f(y, x, tw, cin, cout))
      bytes > 0 && (CUDA.attributes(k.fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = bytes)
      return k
    end
  end)
end

# @synchronize (workgroup barrier) becomes sync_threads in the CUDA flavor
unsync(x) = x
unsync(ex::Expr) = ex.head === :macrocall && ex.args[1] === Symbol("@synchronize") ?
  :(CUDA.sync_threads()) : Expr(ex.head, (unsync(a) for a in ex.args)...)

rename(ex::Expr, suf) = Expr(ex.head, (rename(a, suf) for a in ex.args)...)
rename(s::Symbol, suf) = (str = string(s); (startswith(str, "c") || startswith(str, "a")) && length(str) > 1 && all(isdigit, str[2:end]) ? Symbol(str, "_", suf) : s)
rename(x, suf) = x

const PKERNELS = Dict{Any,Any}()
const PKERNELS_LOCK = ReentrantLock()
# Radix 8 from 512 on: twice the threads per workgroup (batch 4096 × 1024: 0.95× Metal
# instead of 1.36×, batch 512 × 4096: 0.89× instead of 1.08×); below, radix 16.
lradix(n) = n >= 512 ? 8 : 16
const TWLEV = 3             # number of factor tables of W_N^t

# Largest prime factor with a codelet in a kernel; lengths with larger ones use Bluestein's
# algorithm.
const KA_MAX_PRIME = 31

# Radices of a chain of length n: the power-of-two part as radix-R codelets and one remainder
# (first), then the odd part: pairs of 3 as 9, other primes (≤ KA_MAX_PRIME) alone.
function local_radices(n, R = lradix(n))
  r = Int[]; m = n >> trailing_zeros(n); p2 = n ÷ m
  while p2 >= R; push!(r, R); p2 ÷= R; end
  p2 > 1 && push!(r, p2)
  f, rest = FFTL.factorize(m, KA_MAX_PRIME)
  @assert rest == 1 "prime factor > $(KA_MAX_PRIME)"
  e3 = get(f, 3, 0)
  append!(r, fill(9, e3 ÷ 2)); isodd(e3) && push!(r, 3)
  for p in sort!(collect(keys(f))); p == 3 || append!(r, fill(p, f[p])); end
  r
end

# offset of the four-step factor tables behind the local twiddles
local_twcount(n, rs) = sum((R - 1) * (L ÷ R) for (R, L) in zip(rs, accumulate(÷, [n; rs[1:end-1]])); init = 0)

# Hooks for Bluestein along a dimension (Nx, Ny > 0): the first pass (s = 1) reads element i of
# an input of length Nx along the dimension times cin[i], zero for i ≥ Nx; the last pass (m = 1)
# writes element i < Ny of an output of length Ny along the dimension times cout[i]. With
# `xreal` the input is real with lines of Nx + 1 (DCT-I, see ka_real.jl) and element i is
# xe_2i + i xe_2i+1 of their even extension (xe_j = x_(2Nx-j) for j > Nx).
# `blue2`: the whole Bluestein convolution of a line in one kernel (one pass, N = M): forward
# FFT of the chirped input (cin), product with b̂ (cout) at the loads of the backward FFT, which
# writes the first Ny elements times the chirp (cin); workgroup memory holds the spectrum.
# Hooks for real transforms along dimension 1 (E = 1, N = n/2, see ka_real.jl):
#   rhook = +1 (c2r, first pass): element i of the input is Z_i = (X_i + conj X_(N-i)) +
#     cin[i] (X_i - conj X_(N-i)), X a half spectrum with N + 1 values per line;
#   rhook = -1 (r2c, last pass): the output is the half spectrum X_K = (Z_K + conj Z_(N-K))/2 +
#     cout[K] (Z_K - conj Z_(N-K)), K = 0…N, of the pass result Z. With s > 1 a workgroup takes
#     G/2 lanes q from the front and the G/2 mirrored lanes s - q, so that Z_K and Z_(N-K) meet
#     in its workgroup memory (q = 0 and q = s/2 are their own mirrors); G must be even.
#   rhook = 2 / 3 (DCT-II / DST-II of the real lines of length 2N, see ka_real.jl): `pre` reads
#     element i of z = r_2i + i r_2i+1 of the permuted real line (r_j = x_2j for j < N, else
#     ±x_2(2N-1-j)+1, minus for DST-II); `post` turns the pass result Z into the real output:
#     R_K = (Z_K + conj Z_(N-K))/2 + cout[K] (Z_K - conj Z_(N-K)), u_K = cin[K] R_K,
#     y_K = 2 Re u_K, y_(2N-K) = -2 Im u_K (DST-II: at 2N-1-K and K-1), pairs as for r2c.
#   rhook = 4 / 5 (DCT-III / DST-III of real lines of length 2N): `pre` computes element K of the
#     input C from the real line y (reversed for DST-III, y_2N = 0) as
#     C_K = cin[K] (y_K - i y_(2N-K)) + cout[K] (y_(N-K) + i y_(N+K)) (tables in ka_real.jl);
#     `post` writes element K of the result z as r_2K = Re z_K, r_2K+1 = Im z_K to x_2j = r_j
#     (j < N), x_2j+1 = r_(2N-1-j), signed (-1)^j for DST-III.
#   rhook = 6 / 7 (DCT-IV / DST-IV of real lines of length 2N): `pre` reads element K as
#     (x_2K ∓ i x_(2N-1-2K)) cin[K] (minus for DCT-IV); `post` writes element K of the result times
#     cout[K] as y_2K = 2 Re, y_(2N-1-2K) = 2 Im (reversed indices for DST-IV).
#   Hooks 4–7 act per element at the first pass's loads and the last pass's stores, any layout.
#   rhook = 8 (DCT-I of real lines of length N + 1 as the r2c of their even extension of length
#     2N): `pre` reads element K as xe_2K + i xe_2K+1 with xe_i = x_i (i ≤ N), x_(2N-i) (i > N);
#     `post` is the r2c pair hook writing only the real parts, y_K = Re X_K (K = 0…N).
#   `pre` and `post` select the hook in the first and the last pass (both in a one-pass plan).
#   Along a strided dimension (E > 1, DCT hooks only) lane e + E q is column e: the first pass
#   needs G | E; with s > 1 the last pass takes G/2 columns at q and the same columns at s - q
#   (q = 0 with s/2), which needs (G/2) | E.
function pass_kernel(R, G, σ, T, s, N, E = 1, B = 1; Nx = 0, Ny = 0, padx = (1, 0), pady = (1, 0), lim = LIM0, rhook = 0,
                     pre = rhook > 0, post = rhook < 0, xreal = false, blue2 = false, ragged = false, raw = false)
  rhook = abs(rhook)
  @lock PKERNELS_LOCK get!(PKERNELS, (R, G, σ, T, s, N, E, B, Nx, Ny, padx, pady, G * R > lim.nstat, rhook, pre, post, xreal, blue2, ragged, raw)) do
    # padded layouts (n1, pad): logical index L at L + pad ⌊L / n1⌋
    PX = Int32.(padx); PY = Int32.(pady)
    rs = local_radices(R)
    m = N ÷ (R * s)
    sp = E * s                           # stride of q in memory
    Lp = E * N ÷ R                       # lanes per batch item
    batchlanes = Lp == 1 && G > 1        # one-pass transforms with E = 1: lanes are batch items
    @assert R * m * s == N
    @assert ragged ? (s == 1 && m == 1 && E > 1 && rhook == 0 && padx == (1, 0) && pady == (1, 0)) :
            batchlanes ? B % G == 0 : (Lp % G == 0 && (sp > 1 || m % G == 0))
    crossing = !ragged && sp > 1 && sp % G != 0   # lanes of a workgroup span several p: p, q per lane
    # ragged (one pass along a strided dimension): lanes are the E·B lines in memory order,
    # cld(E B, G) workgroups; the lanes past the last line take the last line again and store
    # the same values twice
    NL = E * B
    lane(l) = ragged ? :(min(b0 + $l, $(Int32(NL - 1)))) : :(b0 + $l)
    Rmax = maximum(rs)
    nel = G * R
    W = cld(nel, Rmax)                   # threads per workgroup
    dyn = nel > lim.nstat                # workgroup buffer beyond static shared memory: raw CUDA
    transposed = (sp == 1 && m > 1) || batchlanes   # output runs y[R b + k]
    twiddle = m > 1
    @assert 1 <= W <= lim.wmax && nel <= (dyn ? lim.nmax : lim.nstat)
    @assert (Nx == 0 || s == 1) && (Ny == 0 || (m == 1 && (batchlanes || !transposed)))
    @assert !xreal || (Nx > 0 && padx == (1, 0))
    @assert !blue2 || (Nx > 0 && Ny > 0 && s == 1 && m == 1 && rhook == 0)
    rpre = rhook > 0 && pre; rpost = rhook > 0 && post
    dct = rhook >= 2; dst = rhook in (3, 5, 7)
    dct1 = rhook == 8
    ahook = rhook >= 4                   # element-wise load hooks (by address)
    apost = rpost && 4 <= rhook <= 7     # element-wise store hooks
    ppost = rpost && !apost              # pair post hooks (r2c, DCT-II/DST-II, DCT-I)
    # DCT-III/DST-III in a one-pass plan (lanes are whole lines): stage the real lines in workgroup
    # memory first, so that the four inputs of C_K are read from global memory once, not twice
    spre = rpre && 4 <= rhook <= 5 && R == N
    Lout = dct && !dct1 ? 2N : N + 1     # output line length of the post hook
    # the hooks read the user's input (first pass) and write the user's output (last pass): both
    # unpadded; padded intermediates in between are fine
    @assert rhook == 0 || ((E == 1 || dct) && Nx == 0 && Ny == 0 && !crossing)
    @assert !rpre || padx == (1, 0)
    @assert !rpost || pady == (1, 0)
    @assert !(ppost && E > 1 && s > 1) || E % (G ÷ 2) == 0
    @assert !rpre || s == 1
    @assert !ppost || (m == 1 && (s == 1 || (iseven(s) && iseven(G))))
    bufout = transposed || ppost         # the last stage writes to workgroup memory
    ies = Lp
    ebits = cld(8 * sizeof(Int) - leading_zeros(N - 1), TWLEV)
    offw = local_twcount(R, rs)
    I(v) = Int32(v)
    function twid(t)
      f = [:(@inbounds tw[$(I(offw + 1 + (i << ebits))) + ((t >> $(I(i * ebits))) & $(I((1 << ebits) - 1)))]) for i in 0:TWLEV-1]
      :(t = $t; v = ccmul(v, $(foldl((a, b) -> :(ccmul($a, $b)), f))))
    end
    lb(i) = :(buf[$i + one(Int32)])
    # c2r: element ii of the input Z of line `line`, from the half spectrum X (N + 1 per line)
    rload(line, ii) = quote
      rxl = $line * $(I(N + 1)); rii = $ii
      rxa = @inbounds x[rxl + rii + one(Int32)]; rxb = @inbounds x[rxl + $(I(N)) - rii + one(Int32)]
      rii == zero(Int32) ? complex(real(rxa) + real(rxb), real(rxa) - real(rxb)) :
        (rxa + conj(rxb)) + ccmul(@inbounds(cin[rii + one(Int32)]), rxa - conj(rxb))
    end
    # DCT-II/DST-II: element ii of z of line `line` from the real line (length 2N)
    dload(line, ii) = dloadb(:($line * $(I(2N))), ii)
    # the same from the offset `base` of the line, elements E apart
    dloadb(base, ii) = quote
      dxl = $base; dja = Int32(2) * $ii; djb = dja + one(Int32)
      dra = dja < $(I(N)) ? Int32(2) * dja : Int32(2) * ($(I(2N - 1)) - dja) + one(Int32)
      drb = djb < $(I(N)) ? Int32(2) * djb : Int32(2) * ($(I(2N - 1)) - djb) + one(Int32)
      dva = @inbounds x[dxl + $(I(E)) * dra + one(Int32)]; dvb = @inbounds x[dxl + $(I(E)) * drb + one(Int32)]
      $(dst ? :(complex(dja < $(I(N)) ? dva : -dva, djb < $(I(N)) ? dvb : -dvb)) : :(complex(dva, dvb)))
    end
    xload(line, ii) = dct ? dload(line, ii) : rload(line, ii)
    # DCT-II/DST-II output index j of the line (reversed for DST-II)
    didx(j) = dst ? :($(I(2N - 1)) - $j) : j
    # r2c, s > 1: lane q of local lane l (front lanes l < G/2, then their mirrors)
    rlane(l) = :($l < $(I(G ÷ 2)) ? b0 ÷ Int32(2) + $l :
                 b0 == zero(Int32) && $l == $(I(G ÷ 2)) ? $(I(s ÷ 2)) : $(I(s + G ÷ 2)) - b0 ÷ Int32(2) - $l)
    # E > 1: workgroup g = b0/G of its batch item has columns eb H … eb H + H - 1 at q = qf and
    # at s - qf (s/2 for qf = 0), g = qf (E/H) + eb; lane in memory e + E q
    HE = max(1, G ÷ 2); EB = max(1, E ÷ HE)
    rlaneE(l) = quote
      dg = b0 ÷ $(I(G)); dqf = dg ÷ $(I(EB)); deb = dg - dqf * $(I(EB))
      $l < $(I(HE)) ? deb * $(I(HE)) + $l + $(I(E)) * dqf :
        deb * $(I(HE)) + $l - $(I(HE)) + $(I(E)) * (dqf == zero(Int32) ? $(I(s ÷ 2)) : $(I(s)) - dqf)
    end
    # r2c: X_K and X_(N-K) (line offset ryl) from zk = Z_K and zm = Z_(N-K)
    rpair(K, zk, zm) = dct1 ? ipair(K, zk, zm) : dct ? dpair(K, zk, zm) : quote
      rK = $K; rzk = $zk; rzm = conj($zm)
      rh = (rzk + rzm) * $(T(0.5)); rd = ccmul(@inbounds(cout[rK + one(Int32)]), rzk - rzm)
      @inbounds y[ryl + rK + one(Int32)] = rh + rd
      @inbounds y[ryl + $(I(N)) - rK + one(Int32)] = conj(rh - rd)
    end
    rzero(zk) = dct1 ? izero(zk) : dct ? dzero(zk) : quote
      rz0 = $zk
      @inbounds y[ryl + one(Int32)] = complex(real(rz0) + imag(rz0), zero($T))
      @inbounds y[ryl + $(I(N + 1))] = complex(real(rz0) - imag(rz0), zero($T))
    end
    # DCT-II/DST-II: R_K and R_(N-K) as X_K, X_(N-K) above, then four real outputs (two when
    # K = N - K, written twice with the same values)
    dpair(K, zk, zm) = quote
      rK = $K; rzk = $zk; rzm = conj($zm)
      rh = (rzk + rzm) * $(T(0.5)); rd = ccmul(@inbounds(cout[rK + one(Int32)]), rzk - rzm)
      du = ccmul(@inbounds(cin[rK + one(Int32)]), rh + rd)
      dv = ccmul(@inbounds(cin[$(I(N)) - rK + one(Int32)]), conj(rh - rd))
      @inbounds y[ryl + $(I(E)) * $(didx(:rK)) + one(Int32)] = 2real(du)
      @inbounds y[ryl + $(I(E)) * $(didx(:($(I(2N)) - rK))) + one(Int32)] = -2imag(du)
      @inbounds y[ryl + $(I(E)) * $(didx(:($(I(N)) - rK))) + one(Int32)] = 2real(dv)
      @inbounds y[ryl + $(I(E)) * $(didx(:($(I(N)) + rK))) + one(Int32)] = -2imag(dv)
    end
    # DCT-I: real parts of X_K and X_(N-K) (real lines of N + 1, elements E apart)
    ipair(K, zk, zm) = quote
      rK = $K; rzk = $zk; rzm = conj($zm)
      rh = (rzk + rzm) * $(T(0.5)); rd = ccmul(@inbounds(cout[rK + one(Int32)]), rzk - rzm)
      @inbounds y[ryl + $(I(E)) * rK + one(Int32)] = real(rh + rd)
      @inbounds y[ryl + $(I(E)) * ($(I(N)) - rK) + one(Int32)] = real(rh - rd)
    end
    izero(zk) = quote
      rz0 = $zk
      @inbounds y[ryl + one(Int32)] = real(rz0) + imag(rz0)
      @inbounds y[ryl + $(I(E * N)) + one(Int32)] = real(rz0) - imag(rz0)
    end
    dzero(zk) = quote
      rz0 = $zk
      @inbounds y[ryl + $(I(E)) * $(didx(:(zero(Int32)))) + one(Int32)] = 2 * (real(rz0) + imag(rz0))
      @inbounds y[ryl + $(I(E)) * $(didx(I(N))) + one(Int32)] = 2 * real(@inbounds cin[$(I(N + 1))]) * (real(rz0) - imag(rz0))
    end
    # hooks 4–7: complex element at logical address A (lines of N, E apart) → line offset of the
    # real arrays (lines of 2N) and element K
    asplit(A) = quote
      ao = $A ÷ $(I(E * N)); arem = $A - ao * $(I(E * N)); aK = arem ÷ $(I(E)); ae = arem - aK * $(I(E))
      abase = ao * $(I(E * 2N)) + ae
    end
    ayv(j) = quote   # real input element j (reversed for DST-III), zero at 2N
      ayj = $j
      ayj >= $(I(2N)) ? zero($T) : @inbounds x[abase + $(I(E)) * $(dst ? :($(I(2N - 1)) - ayj) : :ayj) + one(Int32)]
    end
    aload(A) = dct1 ? quote
      ao = $A ÷ $(I(E * N)); arem = $A - ao * $(I(E * N)); aK = arem ÷ $(I(E)); ae = arem - aK * $(I(E))
      abase = ao * $(I(E * (N + 1))) + ae
      ai = Int32(2) * aK; aj = ai + one(Int32)
      ai = ai <= $(I(N)) ? ai : $(I(2N)) - ai; aj = aj <= $(I(N)) ? aj : $(I(2N)) - aj
      complex(@inbounds(x[abase + $(I(E)) * ai + one(Int32)]), @inbounds(x[abase + $(I(E)) * aj + one(Int32)]))
    end : rhook <= 5 ? quote
      $(asplit(A))
      ccmul(@inbounds(cin[aK + one(Int32)]), complex($(ayv(:aK)), -$(ayv(:($(I(2N)) - aK))))) +
        ccmul(@inbounds(cout[aK + one(Int32)]), complex($(ayv(:($(I(N)) - aK))), $(ayv(:($(I(N)) + aK)))))
    end : quote
      $(asplit(A))
      axe = @inbounds x[abase + $(I(E)) * (Int32(2) * aK) + one(Int32)]
      axo = @inbounds x[abase + $(I(E)) * ($(I(2N - 1)) - Int32(2) * aK) + one(Int32)]
      ccmul(complex(axe, $(dst ? :axo : :(-axo))), $(raw ? :(cispi($T(aK) / $T($(2N)))) : :(@inbounds cin[aK + one(Int32)])))
    end
    # staged DCT-III/DST-III: element K of lane l is C_K from the lane's real line y, held as
    # buf[l + G k] = y_2k + i y_2k+1 (y reversed for DST-III, y_2N = 0)
    sy(l, j) = quote
      syj = $j
      syj >= $(I(2N)) ? zero($T) : begin
        syr = $(dst ? :($(I(2N - 1)) - syj) : :syj)
        syv = $(lb(:($l + $(I(G)) * (syr >> one(Int32)))))
        iseven(syr) ? real(syv) : imag(syv)
      end
    end
    sload(l, K) = quote
      sK = $K; sl = $l
      ccmul(@inbounds(cin[sK + one(Int32)]), complex($(sy(:sl, :sK)), -$(sy(:sl, :($(I(2N)) - sK))))) +
        ccmul(@inbounds(cout[sK + one(Int32)]), complex($(sy(:sl, :($(I(N)) - sK))), $(sy(:sl, :($(I(N)) + sK)))))
    end
    # DCT-III/DST-III: r_q to x_j, j = 2q (q < N) or 2(2N-1-q)+1
    rput(q, val) = quote
      aq = $q; ar = $val
      aj = aq < $(I(N)) ? Int32(2) * aq : Int32(2) * ($(I(2N - 1)) - aq) + one(Int32)
      @inbounds y[abase + $(I(E)) * aj + one(Int32)] = $(dst ? :(isodd(aj) ? -ar : ar) : :ar)
    end
    aout(j) = dst && rhook >= 6 ? :($(I(2N - 1)) - $j) : j
    astore(A, v) = rhook <= 5 ? quote
      $(asplit(A)); av = $v
      $(rput(:(Int32(2) * aK), :(real(av))))
      $(rput(:(Int32(2) * aK + one(Int32)), :(imag(av))))
    end : quote
      $(asplit(A)); aw = ccmul($(raw ? :(cispi(($T(4) * $T(aK) + one($T)) / $T($(8N)))) : :(@inbounds(cout[aK + one(Int32)]))), $v)
      @inbounds y[abase + $(I(E)) * $(aout(:(Int32(2) * aK))) + one(Int32)] = 2real(aw)
      @inbounds y[abase + $(I(E)) * $(aout(:($(I(2N - 1)) - Int32(2) * aK))) + one(Int32)] = 2imag(aw)
    end
    # final store of v at logical address A (through the element-wise post hook if any)
    ystore(A) = apost ? astore(A, :v) : :(@inbounds y[padmap($A, $(PY[1]), $(PY[2])) + one(Int32)] = v)
    # lane-major output: y[ybase + l + sp k], twiddle exponent k p s with p = b0 ÷ sp
    lanemajor_store(l, k) = Ny == 0 && crossing ? quote
      bl = b0 + $l; pl = bl ÷ $(I(sp)); ql = bl - pl * $(I(sp))
      $(twiddle ? twid(:($k * pl * $(I(s)))) : nothing)
      $(ystore(:(obase + ql + $(I(sp)) * ($(I(R)) * pl + $k))))
    end : Ny == 0 && ragged ? quote
      bl = $(lane(l)); ol = bl ÷ $(I(E)); el = bl - ol * $(I(E))
      @inbounds y[ol * $(I(E * N)) + el + $(I(E)) * $k + one(Int32)] = v
    end : Ny == 0 ? quote
      $(twiddle ? twid(:($k * pss)) : nothing)
      $(ystore(:(ybase + $l + $(I(sp)) * $k)))
    end : ragged ? quote
      # line bl = e + E o: element k of line o
      bl = $(lane(l)); ol = bl ÷ $(I(E)); el = bl - ol * $(I(E))
      if $k < $(I(Ny))
        @inbounds y[ol * $(I(E * Ny)) + el + $(I(E)) * $k + one(Int32)] = ccmul(v, $(blue2 ? :cin : :cout)[$k + one(Int32)])
      end
    end : quote
      # m = 1: lane b0 + l = e + E q, element q + s k along the dimension
      bl = b0 + $l; ql = bl ÷ $(I(E)); el = bl - ql * $(I(E)); ii = ql + $(I(s)) * $k
      if ii < $(I(Ny))
        @inbounds y[padmap(yob + el + $(I(E)) * ii, $(PY[1]), $(PY[2])) + one(Int32)] = ccmul(v, $(blue2 ? :cin : :cout)[ii + one(Int32)])
      end
    end
    # Bluestein input element ii at line offset `base`, element stride `str`
    bload(base, str, ii) = xreal ? quote
      bxa = Int32(2) * $ii; bxb = bxa + one(Int32)
      bxa = bxa <= $(I(Nx)) ? bxa : $(I(2Nx)) - bxa; bxb = bxb <= $(I(Nx)) ? bxb : $(I(2Nx)) - bxb
      complex(@inbounds(x[$base + $str * bxa + one(Int32)]), @inbounds(x[$base + $str * bxb + one(Int32)]))
    end : :(@inbounds x[padmap($base + $str * $ii, $(PX[1]), $(PX[2])) + one(Int32)])
    body = Any[]
    if spre
      # lane l, real pair k at logical complex address xbase + l R + k (batch lanes) or
      # xbase + l + Lp k; neighbouring threads take neighbouring addresses
      for i in 0:cld(nel, W)-1
        push!(body, quote
          idx = tid + $(I(W * i))
          if idx < $(I(nel))
            $(E == 1 ? :((k, l) = (idx % $(I(R)), idx ÷ $(I(R)))) : :((l, k) = (idx % $(I(G)), idx ÷ $(I(G)))))
            $(asplit(batchlanes ? :(xbase + l * $(I(R)) + k) : :(xbase + l + $(I(ies)) * k)))
            @inbounds $(lb(:(l + $(I(G)) * k))) = complex(x[abase + $(I(E)) * (Int32(2) * k) + one(Int32)],
                                                         x[abase + $(I(E)) * (Int32(2) * k + one(Int32)) + one(Int32)])
          end
        end)
      end
      push!(body, :(@synchronize))
    elseif batchlanes
      # G transforms stored one after another: contiguous load, element j of lane l to l + G j
      for i in 0:cld(nel, W)-1
        push!(body, quote
          idx = tid + $(I(W * i))
          if idx < $(I(nel))
            j = idx % $(I(R)); l = idx ÷ $(I(R))
            $(rpre && ahook ? :($(lb(:(l + $(I(G)) * j))) = $(aload(:(xbase + idx)))) :
              rpre ? :($(lb(:(l + $(I(G)) * j))) = $(xload(:(b0 + l), :j))) :
              Nx == 0 ? :(@inbounds $(lb(:(l + $(I(G)) * j))) = x[padmap(xbase + idx, $(PX[1]), $(PX[2])) + one(Int32)]) :
              :(@inbounds $(lb(:(l + $(I(G)) * j))) = j < $(I(Nx)) ? ccmul($(bload(:($(lane(:l)) * $(I(xreal ? Nx + 1 : Nx))), I(1), :j)), cin[j + one(Int32)]) : zero(VT)))
          end
        end)
      end
      push!(body, :(@synchronize))
    end
    for (ci, σc) in enumerate(blue2 ? (-1, 1) : (σ,))
    L = R; st = G; off = 0
    for (si, r) in enumerate(rs)
      mm = L ÷ r
      clast = si == length(rs)                       # last stage of this chain
      first = si == 1 && ci == 1; last = clast && (!blue2 || ci == 2)
      bfirst = blue2 && ci == 2 && si == 1           # backward chain: loads times b̂
      fload = first && !batchlanes && !spre
      nb = nel ÷ r                       # butterflies of this stage
      cnt = cld(nb, W)
      # the last round may run past nb: compute a clamped butterfly, skip its stores
      guard(c, ex) = W * (c + 1) > nb ? :(if tid + $(I(W * c)) < $(I(nb)); $ex; end) : ex
      stmts, a, X = FFTL.codelet(r, σc)
      stmts = [retype(e, T) for e in stmts]
      for c in 0:cnt-1
        pc = Symbol("p_", c); qc = Symbol("q_", c)
        gc = W * (c + 1) > nb ? :(min(tid + $(I(W * c)), $(I(nb - 1)))) : :(tid + $(I(W * c)))
        push!(body, :(g = $gc), :($pc = g ÷ $(I(st))), :($qc = g - $pc * $(I(st))))
        for j in 1:r
          # first local stage: st = G, so local index q + G (p + j mm) is lane q, element p + j mm
          src = bfirst ? :(ccmul($(lb(:($qc + $(I(st)) * ($pc + $(I((j - 1) * mm)))))), cout[$pc + $(I((j - 1) * mm + 1))])) :
                first && spre ? sload(qc, :($pc + $(I((j - 1) * mm)))) :
                !fload ? lb(:($qc + $(I(st)) * ($pc + $(I((j - 1) * mm))))) :
                rpre && ahook ? aload(:(xbase + $qc + $(I(ies)) * ($pc + $(I((j - 1) * mm))))) :
                rpre && E > 1 ? quote   # lane b0 + q = e + E p: column e, element p + m t
                  dgl = b0 + $qc; dgt = $pc + $(I((j - 1) * mm)); dgp = dgl ÷ $(I(E))
                  $(dloadb(:(ob * $(I(E * 2N)) + dgl - dgp * $(I(E))), :(dgp + $(I(m)) * dgt)))
                end :
                rpre ? xload(:ob, :(b0 + $qc + $(I(ies)) * ($pc + $(I((j - 1) * mm))))) :
                ppost && s > 1 && E > 1 ? :(x[padmap(obase + $(rlaneE(qc)) + $(I(ies)) * ($pc + $(I((j - 1) * mm))), $(PX[1]), $(PX[2])) + one(Int32)]) :
                ppost && s > 1 ? :(x[padmap(obase + $(rlane(qc)) + $(I(ies)) * ($pc + $(I((j - 1) * mm))), $(PX[1]), $(PX[2])) + one(Int32)]) :
                Nx == 0 && ragged ? quote
                  bl = $(lane(qc)); ol = bl ÷ $(I(E)); el = bl - ol * $(I(E))
                  x[ol * $(I(E * N)) + el + $(I(E)) * ($pc + $(I((j - 1) * mm))) + one(Int32)]
                end :
                Nx == 0 ? :(x[padmap(xbase + $qc + $(I(ies)) * ($pc + $(I((j - 1) * mm))), $(PX[1]), $(PX[2])) + one(Int32)]) :
                ragged ? quote   # line bl = e + E o, element ii
                  bl = $(lane(qc)); ol = bl ÷ $(I(E)); el = bl - ol * $(I(E))
                  ii = $pc + $(I((j - 1) * mm))
                  $(Nx >= N ? :(ccmul($(bload(:(ol * $(I(E * (xreal ? Nx + 1 : Nx))) + el), I(E), :ii)), cin[ii + one(Int32)])) :
                    :(ii < $(I(Nx)) ? ccmul($(bload(:(ol * $(I(E * (xreal ? Nx + 1 : Nx))) + el), I(E), :ii)), cin[ii + one(Int32)]) : zero(VT)))
                end :
                quote   # s = 1: lane b0 + q = e + E p, element p + m jR along the dimension
                  bl = b0 + $qc
                  pl = bl ÷ $(I(E)); el = bl - pl * $(I(E))
                  ii = pl + $(I(m)) * ($pc + $(I((j - 1) * mm)))
                  $(Nx >= N ? :(ccmul($(bload(:(xob + el), I(E), :ii)), cin[ii + one(Int32)])) :
                    :(ii < $(I(Nx)) ? ccmul($(bload(:(xob + el), I(E), :ii)), cin[ii + one(Int32)]) : zero(VT)))
                end
          push!(body, :($(rename(a[j], c)) = @inbounds $src))
        end
        append!(body, [rename(e, c) for e in stmts])
      end
      fload || push!(body, :(@synchronize))
      for c in 0:cnt-1
        pc = Symbol("p_", c); qc = Symbol("q_", c)
        push!(body, :(o = $qc + $(I(st * r)) * $pc))
        stores = Any[]
        for k in 0:r-1
          v = rename(X[k+1], c)
          if last && !bufout
            # mm = 1: local index q + k st = l + G kel
            push!(stores, quote
              v = $v
              l = $qc % $(I(G)); kel = $qc ÷ $(I(G)) + $(I(k * (R ÷ r)))
              $(lanemajor_store(:l, :kel))
            end)
          else
            twv = :(@inbounds tw[$(I(off + 1)) + $(I(r - 1)) * $pc + $(I(k - 1))])
            val = (clast || k == 0) ? v : :(ccmul($v, $(ci == 2 ? :(conj($twv)) : twv)))
            push!(stores, :(@inbounds $(lb(:(o + $(I(k * st))))) = $val))
          end
        end
        push!(body, guard(c, Expr(:block, stores...)))
      end
      last && !bufout || push!(body, :(@synchronize))
      off += (r - 1) * mm
      L = mm; st *= r
    end
    end
    if ppost
      # element k of local lane l is at l + G k in workgroup memory
      H = G ÷ 2
      cnt = s == 1 ? G * (R ÷ 2 + 1) : H * R
      for i in 0:cld(cnt, W)-1
        push!(body, quote
          idx = tid + $(I(W * i))
          if idx < $(I(cnt))
            $(s == 1 ? quote
              # lanes are whole lines: pairs k, R - k of a lane (along a strided dimension the
              # neighbouring threads take neighbouring columns, so that their stores coalesce)
              $(E == 1 ? :((k, l) = (idx % $(I(R ÷ 2 + 1)), idx ÷ $(I(R ÷ 2 + 1)))) :
                         :((l, k) = (idx % $(I(G)), idx ÷ $(I(G)))))
              ryl = $(E == 1 ? :((ob + b0 + l) * $(I(Lout))) : :(ob * $(I(E * Lout)) + b0 + l))
              if k == zero(Int32)
                $(rzero(lb(:l)))
              else
                $(rpair(:k, lb(:(l + $(I(G)) * k)), lb(:(l + $(I(G)) * ($(I(R)) - k)))))
              end
            end : E > 1 ? quote
              # front column l (q = qf) element k pairs with lane H + l (q = s - qf) element R - 1 - k
              l = idx % $(I(H)); k = idx ÷ $(I(H))
              dg = b0 ÷ $(I(G)); dqf = dg ÷ $(I(EB)); deb = dg - dqf * $(I(EB))
              ryl = ob * $(I(E * Lout)) + deb * $(I(H)) + l
              if dqf == zero(Int32)
                if k == zero(Int32)
                  $(rzero(lb(:l)))
                elseif k <= $(I(R ÷ 2))
                  $(rpair(:($(I(s)) * k), lb(:(l + $(I(G)) * k)), lb(:(l + $(I(G)) * ($(I(R)) - k)))))
                end
                if k < $(I(cld(R, 2)))
                  $(rpair(:($(I(s ÷ 2)) + $(I(s)) * k), lb(:($(I(H)) + l + $(I(G)) * k)), lb(:($(I(H)) + l + $(I(G)) * ($(I(R - 1)) - k)))))
                end
              else
                $(rpair(:(dqf + $(I(s)) * k), lb(:(l + $(I(G)) * k)), lb(:($(I(H)) + l + $(I(G)) * ($(I(R - 1)) - k)))))
              end
            end : quote
              # front lane l (q = b0/2 + l) element k pairs with lane H + l element R - 1 - k
              l = idx % $(I(H)); k = idx ÷ $(I(H))
              ryl = ob * $(I(Lout))
              if b0 == zero(Int32) && l == zero(Int32)
                # q = 0: elements k, R - k; q = s/2: elements k, R - 1 - k
                if k == zero(Int32)
                  $(rzero(lb(:(zero(Int32)))))
                elseif k <= $(I(R ÷ 2))
                  $(rpair(:($(I(s)) * k), lb(:($(I(G)) * k)), lb(:($(I(G)) * ($(I(R)) - k)))))
                end
                if k < $(I(cld(R, 2)))
                  $(rpair(:($(I(s ÷ 2)) + $(I(s)) * k), lb(:($(I(H)) + $(I(G)) * k)), lb(:($(I(H)) + $(I(G)) * ($(I(R - 1)) - k)))))
                end
              else
                $(rpair(:(b0 ÷ Int32(2) + l + $(I(s)) * k), lb(:(l + $(I(G)) * k)), lb(:($(I(H)) + l + $(I(G)) * ($(I(R - 1)) - k)))))
              end
            end)
          end
        end)
      end
    elseif transposed
      for i in 0:cld(nel, W)-1
        push!(body, quote
          idx = tid + $(I(W * i))
          if idx < $(I(nel))
            k = idx % $(I(R)); l = idx ÷ $(I(R))
            v = @inbounds $(lb(:(l + $(I(G)) * k)))
            $(twiddle ? twid(:((b0 + l) * k)) : nothing)
            $(Ny == 0 ? ystore(:(obase + $(I(R)) * (b0 + l) + k)) :
              :(if k < $(I(Ny)); @inbounds y[padmap($(lane(:l)) * $(I(Ny)) + k, $(PY[1]), $(PY[2])) + one(Int32)] = ccmul(v, $(blue2 ? :cin : :cout)[k + one(Int32)]); end))
          end
        end)
      end
    end
    gpb = batchlanes ? 1 : Lp ÷ G        # workgroups per batch item
    ngroups = ragged ? cld(NL, G) : batchlanes ? B ÷ G : B * gpb
    name = Symbol(crossing ? "ka_passx_" : "ka_pass_", R, "_", join(rs, "x"), "_G", G, "_s", s, "_N", N, "_E", E, "_B", B, "_",
                  σ < 0 ? "f" : "b", "_", T, "_x", Nx, "_y", Ny, padx[2] > 0 ? "_px$(padx[1])x$(padx[2])" : "", pady[2] > 0 ? "_py$(pady[1])x$(pady[2])" : "",
                  dyn ? "_dyn" : "", rhook == 0 ? "" : string("_h", rhook, rpre ? "p" : "", rpost ? "q" : ""), xreal ? "_xr" : "", blue2 ? "_b2" : "", ragged ? "_rg" : "")
    idxpre = quote
      ob = $(batchlanes || ragged ? :(zero(Int32)) : :(grp ÷ $(I(gpb))))
      b0 = (grp - ob * $(I(gpb))) * $(I(G))
      obase = ob * $(I(E * N))
      xbase = $(batchlanes ? :(b0 * $(I(N))) : :(obase + b0))
      p0 = b0 ÷ $(I(sp))
      ybase = obase + b0 - p0 * $(I(sp)) + $(I(sp * R)) * p0
      pss = p0 * $(I(s))
      xob = ob * $(I(E * (xreal ? Nx + 1 : Nx)))
      yob = ob * $(I(E * Ny))
      VT = Complex{$T}
    end
    if dyn || raw
      cuda_support!()
      ex = quote
        function $(Symbol(name, raw && !dyn ? "_raw" : ""))(y, x, tw, cin, cout)
          tid = CUDA.threadIdx().x % Int32 - Int32(1)
          grp = CUDA.blockIdx().x % Int32 - Int32(1)
          assume(Int32(0) <= tid < $(Int32(W))); assume(Int32(0) <= grp < $(Int32(ngroups)))
          $(dyn ? nothing : :((x, tw, cin, cout) = map(Base.Experimental.Const, (x, tw, cin, cout))))
          $(idxpre.args...)
          buf = $(dyn ? :(CUDA.CuDynamicSharedArray(Complex{$T}, ($nel,))) : :(CUDA.CuStaticSharedArray(Complex{$T}, ($nel,))))
          $((unsync(e) for e in body)...)
          return nothing
        end
      end
    else
      ex = quote
        @kernel function $name(y, @Const(x), @Const(tw), @Const(cin), @Const(cout))
          tid = (@index(Local, Linear) - 1) % Int32
          grp = (@index(Group, Linear) - 1) % Int32
          $(idxpre.args...)
          buf = @localmem Complex{$T} ($nel,)
          $(body...)
        end
      end
    end
    Core.eval(Kernels, ex)
    # the kernel exists from the next world age on
    (Base.invokelatest(getfield, Kernels, Symbol(name, raw && !dyn ? "_raw" : "")), W, ngroups,
     dyn ? nel * sizeof(Complex{T}) : 0)
  end
end

# local chain twiddles (stage i: w_L^(k p) at off_i + (r-1) p + k), then the factor tables of W_N
function pass_twiddles(T, R, σ, N)
  t = Complex{T}[]; L = R
  for r in local_radices(R)
    mm = L ÷ r
    append!(t, vec([Complex{T}(cispi(2σ * k * p / L)) for k in 1:r-1, p in 0:mm-1]))
    L = mm
  end
  ebits = cld(8 * sizeof(Int) - leading_zeros(N - 1), TWLEV)
  [t; [Complex{T}(cispi(2σ * mod(j << (i * ebits), N) / N)) for j in 0:(1 << ebits)-1, i in 0:TWLEV-1][:]]
end

# largest G ≤ lim dividing d (1: G is always divisible; also covers lim < 1 from the W cap)
divisor_le(d, lim) = maximum(g for g in 1:min(d, lim) if d % g == 0; init = 1)

# G for a pass: G R ≤ cap (workgroup memory), G ≤ gmax, G | sp (or G | m when sp = 1, G | B for
# batch lanes).
function pass_G_cap(R, s, m, E, B, gmax, N, cap, wmax)
  lim = min(cap ÷ R, gmax, E > 1 && R == N ? 16 : gmax)   # one-pass strided dimension: G ≤ 16 measured best
  # also keep W = cld(G·R, largest radix) ≤ wmax threads per workgroup
  lim = min(lim, (wmax * maximum(local_radices(R); init = 1)) ÷ R)
  sp = E * s
  # one-pass transforms of contiguous data: about 512 elements per workgroup measured best
  E * m * s == 1 && return divisor_le(B, min(lim, max(1, 512 ÷ R)))
  sp == 1 && return divisor_le(m, lim)
  # G dividing sp keeps p fixed per workgroup; otherwise lanes may span several p (any G | sp m)
  g1 = divisor_le(sp, lim); g2 = divisor_le(sp * m, lim)
  2g1 >= g2 ? g1 : g2
end

# G for a pass of the plan. The CUDA dynamic shared memory flavor (lim.nmax > lim.nstat) only
# adopts the larger G for power-of-two pass lengths of at least 1024 (passes of 512 lose with it:
# 2^17 [512, 256] 16.2 µs vs 10.0 µs static, 2^18 [512, 512] 19.2 vs 14.3 µs on the A6000), and only
# when the workgroup runs with (nearly) the maximum of threads: a >48 KB shared block is one workgroup per SM, so G growth that leaves W = 512 halves
# the occupancy of the static plan (2^22 [256] G32 dyn 1.39× cuFFT vs G16 static 1.24×), and
# the longer register chains of odd-radix stages do not pay either (10^6 [1000] G8 dyn 0.89×
# vs G4 static 0.80×); the measured wins are 2^20 [1024] G8, 2^21 [2048] G4 + [1024] G8,
# 2^22 [2048] G4 -- all W = 1000..1024 threads.
function pass_G(R, s, m, E, B, gmax, lim = LIM0)
  N = R * m * s
  G = pass_G_cap(R, s, m, E, B, gmax, N, lim.nstat, lim.wmax)
  lim.nmax <= lim.nstat && return G
  Gd = pass_G_cap(R, s, m, E, B, gmax, N, lim.nmax, lim.wmax)
  Gd > G && ispow2(R) && R >= 1024 && 8 * cld(Gd * R, maximum(local_radices(R); init = 1)) >= 7 * lim.wmax ? Gd : G
end

# Pass lengths (powers of two). A last pass of radix ≤ 16 needs no exchange in threadgroup
# memory; on the M1 a last pass with exchange (large store stride) runs at half bandwidth
# once the data no longer fit the system level cache.
# Rule sets: :apple measured on an M1 Max against Metal.jl's FFT (48 MB system level cache, a
# pass with large power-of-two strides on both sides is slow there), :generic measured on an
# RTX A6000 against cuFFT (6 MB L2; passes up to 512 run at copy speed, no stride penalty).
device_profile(backend) = nameof(typeof(backend)) === :MetalBackend ? :apple : :generic

# Generic 1D rule (A6000): two balanced passes up to 2^21 (2^20 [1024, 1024] 1.11× cuFFT, three
# passes 1.22×); from 2^22 on at least three passes ≤ 512 (2^22 [256, 128, 128] 1.23× vs
# [1024, 256, 16] 1.41×; 2^24 [256, 256, 256] 0.97× vs [64, 64, 64, 64] 1.30×).
# With dynamic shared memory (lim.nmax > lim.nstat) two passes reach one pass further: 2^21 G 4/8 and
# 2^22 [2048, 2048] 1.09× (a 3rd pass is worth its cost only when W = 1024 workgroups saturate:
# 2^22 3-pass dyn 1.38×, at 2^24 nothing beats the static plan 0.97×).
function generic_passes(N, lim = LIM0)
  ispow2(N) || return mixed_passes(N, lim)
  e = trailing_zeros(N)
  N <= lim.nstat && return [N]
  e <= (lim.nmax > lim.nstat ? 22 : 21) && 1 << cld(e, 2) <= lim.nmax && return [1 << cld(e, 2), 1 << (e ÷ 2)]
  balance_passes(N, max(3, cld(e, 9)), lim.nstat)
end

function default_passes(N, prof = :apple, lim = LIM0)
  prof === :generic && return generic_passes(N, lim)
  ispow2(N) || return mixed_passes(N, Limits(lim.nstat, lim.nstat, lim.wmax, lim.slc))
  e = trailing_zeros(N)
  N <= lim.nstat && return [N]
  e <= 21 && return [1 << cld(e, 2), 1 << (e ÷ 2)]       # 2^21 complex F32 still fit the SLC
  e <= 23 && return [1024, 1 << (e - 14), 16]
  P = cld(e, 6)
  [1 << (e ÷ P + (i <= e % P)) for i in 1:P]
end

# Pass lengths along a dimension with inner size E. Strided dimensions (E > 1) beyond the SLC:
# on the M1 a pass whose workgroups touch more than ~32 rows far apart (load and store) runs
# at a fraction of the bandwidth, radix ≤ 16 passes run at copy speed.
# "Beyond the SLC" means more than 5/6 of it (M1 Max: 40 of 48 MB, measured).
beyond_slc(bytes, lim) = bytes > lim.slc ÷ 6 * 5
function dim_passes(N, E, bytes; padded = false, first = true, last = true, prof = :apple, gmax = 64, lim = LIM0)
  E == 1 && return default_passes(N, prof, lim)
  if prof === :generic   # A6000: 256³ [256] per dimension 0.99× cuFFT; 4096² dim 2 [64, 64]
    N <= 1024 && return [N]
    # one pass of a strided dimension when its workgroups read/write at least 4 lanes per row
    # (2048² dim 2: [2048] G 4;  4096² and 3000² stay multi-pass: W > 1024 or G < 4)
    pass_G(N, 1, 1, E, 1, gmax, lim) >= 4 && return [N]
    return balance_passes(N, max(2, cld(ceil(Int, log2(N)), 8)), lim.nstat)
  end
  if beyond_slc(bytes, lim) && padded
    # (measured, M1 Max) small stride (≤ 64 KB): one pass up to 1024 (1024×1024×32 dim 2).
    # Reading the unpadded input: passes ≤ 64 (4096² dim 2 only: [64, 64] 0.91×, [256, 16] 1.11×).
    # Between padded buffers passes ≤ 256; the one writing the unpadded result ≤ 16, unless the
    # whole dimension is ≤ 32 (1024×1024×32 dim 3: [32] instead of [2, 16]).
    E * 8 <= 64 << 10 && N <= 1024 && return [N]
    last && N <= 32 && return [N]
    first && return balance_passes(N, cld(ceil(Int, log2(N)), 6))
    if last
      f = divisor_le(N, 16); f < 4 && return pack_passes(N, 16)
      return [N ÷ f <= 256 ? [N ÷ f] : balance_passes(N ÷ f, cld(ceil(Int, log2(N ÷ f)), 8)); f]
    end
    N <= 256 && return [N]
    return balance_passes(N, cld(ceil(Int, log2(N)), 8))
  end
  beyond_slc(bytes, lim) && return pack_passes(N, 16)
  N <= 1024 && return [N]
  balance_passes(N, 2)
end

# Non-powers of two (M1 Max, F32, measured):
# - From 2^19 on, a small odd part o ≤ 256 is a pass of its own, padded with factors 2 to ≥ 12,
#   the power of two rest as default_passes: 15·2^16 [15, 256, 256] 0.89× Metal vs [1024, 960]
#   1.08×; 5·2^18 [20, 256, 256] 0.80× vs [5, 512, 512] 0.98× and [1280, 1024] (G = 2) 1.03×.
# - Two passes (launch-bound sizes): the split with the most lanes per workgroup in the worse
#   pass, then the most balanced one.
# - Otherwise balanced passes; one more pass if a pass gets G < 4 (3^13: [243, 81, 81]).
function mixed_passes(N, lim = LIM0)
  N <= lim.nstat && return [N]
  o = N >> trailing_zeros(N)
  if o <= 256 && N >= 1 << 19
    a = o; p2 = N ÷ o
    while a < 12 && p2 > 1; a *= 2; p2 ÷= 2; end
    p2 > 1 && return [a; p2 <= lim.nstat ? [p2] : default_passes(p2, :apple, lim)]
  end
  P = N <= 1 << 21 ? 2 : N <= 1 << 24 ? 3 : cld(ceil(Int, log2(N)), 6)
  if P == 2
    cands = [[d, N ÷ d] for d in 1:lim.nmax if N % d == 0 && N ÷ d <= lim.nmax]
    rs = argmax(c -> (minimum(passes_G(N, c, lim)), -abs(log(c[1] / c[2]))), cands)
  else
    rs = balance_passes(N, P, lim.nstat)
  end
  minimum(passes_G(N, rs, lim)) >= 4 && return rs
  rs2 = balance_passes(N, P + 1, lim.nstat)
  minimum(passes_G(N, rs2, lim)) > minimum(passes_G(N, rs, lim)) ? rs2 : rs
end

passes_G(N, rs, lim = LIM0) = (s = 1; [(G = pass_G(R, s, N ÷ (R * s), 1, 1, 64, lim); s *= R; G) for R in rs])

# N as few pass lengths ≤ cap as first-fit decreasing packing of the prime factors finds
function pack_passes(N, cap)
  f, rest = FFTL.factorize(N, KA_MAX_PRIME); @assert rest == 1
  bins = Int[]
  for p in sort!(reduce(vcat, [fill(p, k) for (p, k) in f]; init = Int[]); rev = true)
    i = findfirst(b -> b * p <= cap, bins)
    i === nothing ? push!(bins, p) : (bins[i] *= p)
  end
  sort!(bins; rev = true)
end

# N as a product of P pass lengths ≤ cap: prime factors, largest first, each to the pass with
# the smallest product (more passes if a length exceeds cap)
function balance_passes(N, P, cap = NMAX)
  f, rest = FFTL.factorize(N, KA_MAX_PRIME); @assert rest == 1
  ps = sort!(reduce(vcat, [fill(p, k) for (p, k) in f]; init = Int[]); rev = true)
  while true
    bins = ones(Int, P)
    for p in ps; bins[argmin(bins)] *= p; end
    maximum(bins) <= cap && return sort!(bins; rev = true)
    P += 1
  end
end

# Lines per workgroup of a one-pass strided dimension with ragged lanes (as pass_G_cap without
# the divisibility; 128 and 224 along E = 101 on the M1: G = 1 runs at a third of the bandwidth).
# Only below G = 4: with a good divisor ragged lanes are mixed (whole transforms, interleaved:
# 125³ G 5 → 16 0.88×, 243² G 9 → 16 1.12×, 99³, 105³, 135²×64 1.00–1.03×).
ragged_G(R, E, B, gmax, lim) = max(1, min(lim.nstat ÷ R, gmax, 16, (lim.wmax * maximum(local_radices(R))) ÷ R, E * B))

# Lines per workgroup of the fused Bluestein kernel with ragged lanes along a strided dimension
# (any number of lines; 101³ dim 2 on the M1: G 1/2/4/8/16 663/423/357/335/413 µs). Along
# dimension 1 the two launches with G = 1 stay faster (101³ 284 µs against 364 µs at G = 8).
blue_ragged_G(M, E, B, lim) = max(1, min(lim.nstat ÷ M, (lim.wmax * maximum(local_radices(M))) ÷ M, 8, E * B))

# Bluestein along a dimension of length N (prime factor > KA_MAX_PRIME):
# X_k = c_k Σ_j (x_j c_j) conj(c_{k-j}), c_j = exp(σ iπ j²/N), as a cyclic convolution of length
# M ≥ 2N - 1. Forward passes of length M read x times c (load hook, zero padding), the first
# backward pass reads times b̂ = FFT_M(conj(c) wrapped) / M, the last one writes the first N
# elements times c (store hook): 2P launches, scratch arrays of E·M·B elements.
# On Apple GPUs a power of two whenever it fits one pass: the fused kernel runs its odd radix
# stages slowly (M1, 101³ dim 2, µs: 256 [16, 16] 142, 224 [16, 2, 7] 335, 240 [16, 3, 5] 385,
# 288 [16, 2, 9] 454, 320 [16, 4, 5] 434; c2c 101³ 1008 → 482).
function bluestein_length(n, prof = :generic, lim = LIM0)
  prof === :apple && nextpow(2, n) <= lim.nstat && return nextpow(2, n)
  minimum(o << max(0, ceil(Int, log2(n / o))) for o in (1, 3, 5, 7, 9, 15))
end

function bluestein_tables(backend, T, N, σ)
  M = bluestein_length(2N - 1, device_profile(backend), device_limits(backend, T))
  c = [Complex{T}(cispi(σ * mod(j * j, 2N) / N)) for j in 0:N-1]   # j² mod 2N keeps the argument small
  b = zeros(Complex{T}, M)
  b[1] = conj(c[1])
  for j in 1:N-1; b[j+1] = b[M-j+1] = conj(c[j+1]); end
  db = KA.allocate(backend, Complex{T}, M); copyto!(db, b)
  bhat = KA.allocate(backend, Complex{T}, M)
  P = KAPlan(backend, T, (M,), -1)
  launch!(bhat, db, P); KA.synchronize(backend)
  bhat .*= T(1 / M)
  dc = KA.allocate(backend, Complex{T}, N); copyto!(dc, c)
  M, dc, bhat
end

mutable struct KAPlan{T,N,B} <: AbstractFFTs.Plan{Complex{T}}
  sz::NTuple{N,Int}
  region::Vector{Int}
  σ::Int
  radices::Vector{Vector{Int}}  # pass lengths per transformed dimension (of M for Bluestein)
  steps::Vector{Any}            # (kernel instance | RawLaunch (CUDA), ndrange, tw, cin, cout, src, dst); src, dst: :x, :y or a scratch array
  bufs::Dict{Symbol,Any}        # :a, :b array-shaped; :c, :d Bluestein scratch
  backend::B
  gmax::Int
  rhook::Int                    # real-transform hook in use (see pass_kernel), 0 for complex plans
  pinv::AbstractFFTs.Plan       # set by AbstractFFTs.inv on first use
  KAPlan{T,N,B}(sz, region, σ, radices, steps, bufs, backend, gmax, rhook = 0) where {T,N,B} =
    new{T,N,B}(sz, region, σ, radices, steps, bufs, backend, gmax, rhook)
end

# DFT along the dimensions `region` of an array of size `dims` (column major). The buffers :a
# and :b may use a padded layout (`pad` elements after each run of dims[1]).
# `rhook` = ±1 with the table `rtab` (region 1:1, no Bluestein): the half-length line plan of a
# real transform, see pass_kernel; the r2c hook is dropped (plan.rhook = 0) when the last pass
# has an odd G. `rhook` = 2 / 3 with `rtab` = (w, t): DCT-II / DST-II lines of length 2·dims[d]
# along the one dimension d of `region` (pre hook in the first pass, post hook in the last);
# dropped as a whole (plan.rhook = 0) when the workgroups of the first or last pass do not fit
# the hooks (see pass_kernel).
# Largest lanes per workgroup for a pass: on CUDA, one-dimensional transforms of 2^13 to 2^18
# values run best with 8 lanes (A6000 kernel times, F32 c2c: 8192 4.6 µs against 9.0 µs with 64,
# 65536 6.0 against 8.0; 2^12 and 2^20 do not change or prefer 16–64).
plan_gmax(backend, dims) =
  nameof(typeof(backend)) === :CUDABackend && length(dims) == 1 && 2^13 <= dims[1] <= 2^18 ? 8 : 64

function KAPlan(backend, ::Type{T}, dims::NTuple{D,Int}, σ; region = 1:D, radices = nothing, gmax = plan_gmax(backend, dims), rhook = 0, rtab = nothing, xreal = false, blueG = nothing,
                lim = device_limits(backend, T),   # override: hardware limits (rule tuning)
                pad = default_pad(dims, region, T, device_profile(backend), lim)) where {T,D}
  region = FFTL.normregion(region, D)
  all(>=(1), dims) || throw(ArgumentError("array size must be positive"))
  filter!(d -> dims[d] > 1, region)          # a DFT of length 1 is the identity
  isempty(region) && return KAPlan{T,D,typeof(backend)}(dims, region, σ, Vector{Int}[], Any[], Dict{Symbol,Any}(), backend, gmax)
  prof = device_profile(backend)
  specs = Any[]; rsall = Vector{Int}[]
  dummy = KA.allocate(backend, Complex{T}, 1)
  cur = :x; scratch = 0
  arr_next(src) = src === :a ? :b : :a
  scr_next(src) = src === :c ? :d : :c
  for (i, d) in enumerate(region)
    N = dims[d]; E = prod(dims[1:d-1]; init = 1); B = prod(dims[d+1:end]; init = 1)
    blue = FFTL.factorize(N, KA_MAX_PRIME)[2] > 1
    M, cin, bhat = blue ? bluestein_tables(backend, T, N, σ) : (N, dummy, dummy)
    rs = radices === nothing ? dim_passes(M, E, E * M * B * sizeof(Complex{T}); padded = pad > 0, first = i == 1, last = i == length(region), prof, gmax = gmax, lim = lim) : radices[i]
    @assert prod(rs) == M
    push!(rsall, collect(rs))
    # one pass of M: the whole convolution in one kernel without scratch, unless its workgroups
    # are below one SIMD group (G = 1 for 101³: 14 threads, where the two launches run more
    # workgroups at once; 101³ 1.7× slower fused)
    G0 = blue && length(rs) == 1 ? pass_G(M, 1, 1, E, B, gmax, lim) : 0
    if blue && length(rs) == 1 && (G0 * M ÷ maximum(local_radices(M)) >= 32 || (E > 1 && blue_ragged_G(M, E, B, lim) > G0))
      dst = arr_next(cur)
      rg = E > 1 && blue_ragged_G(M, E, B, lim) > G0
      sp = (R = M, s = 1, m = 1, M = M, E = E, B = B, σ = -1, Nx = N, Ny = N, blue2 = true,
            cin = cin, cout = bhat, src = cur, dst = dst)
      push!(specs, rg ? merge(sp, (ragged = true, G = something(blueG, blue_ragged_G(M, E, B, lim)))) : sp)
      cur = dst
      continue
    end
    chains = blue ? [(-1, rs), (1, rs)] : [(σ, rs)]
    blue && (scratch = max(scratch, E * M * B))
    for (ci, (σc, rsc)) in enumerate(chains)
      s = 1
      for (pi, R) in enumerate(rsc)
        m = M ÷ (R * s)
        Nx = blue && pi == 1 ? (ci == 1 ? N : M) : 0
        Ny = blue && ci == 2 && pi == length(rsc) ? N : 0
        dst = !blue || Ny > 0 ? arr_next(cur) : scr_next(cur)
        push!(specs, (R = R, s = s, m = m, M = M, E = E, B = B, σ = σc, Nx = Nx, Ny = Ny,
                      cin = Nx == N ? cin : Nx > 0 ? bhat : dummy, cout = Ny > 0 ? cin : dummy, src = cur, dst = dst))
        cur = dst; s *= R
      end
    end
  end
  specs[end] = merge(specs[end], (dst = :y,))
  if xreal
    @assert length(region) == 1 && specs[1].Nx > 0
    specs[1] = merge(specs[1], (xreal = true,))
  end
  if 4 <= rhook <= 7
    # element-wise hooks: the first pass must not cross lanes (G | E)
    @assert length(region) == 1 && specs[1].Nx == 0 && specs[end].Ny == 0
    sf = specs[1]; E = sf.E
    Gf = pass_G(sf.R, sf.s, sf.m, E, sf.B, gmax, hook_limits(lim, backend))
    if E == 1 || E % Gf == 0
      ta, tb = rtab
      specs[1] = merge(specs[1], rhook <= 5 ? (rh = rhook, pre = true, cin = ta, cout = tb) : (rh = rhook, pre = true, cin = ta))
      specs[end] = merge(specs[end], rhook <= 5 ? (rh = rhook, post = true) : (rh = rhook, post = true, cout = tb))
    else
      rhook = 0
    end
  elseif rhook >= 2   # 2, 3, 8: pair post hook; pre hook in the first pass
    @assert length(region) == 1 && specs[1].Nx == 0 && specs[end].Ny == 0
    sf = specs[1]; sp = specs[end]; E = sf.E
    Gf = pass_G(sf.R, sf.s, sf.m, E, sf.B, gmax, lim)
    G = pass_G(sp.R, sp.s, sp.m, E, sp.B, gmax, lim)
    # mirrored lanes q, s - q need an even s (the lane s/2 pairs with q = 0)
    if (E == 1 || E % Gf == 0) && (sp.s == 1 || (iseven(sp.s) && iseven(G) && (E == 1 || E % (G ÷ 2) == 0)))
      w, t = rtab
      specs[1] = merge(specs[1], (rh = rhook, pre = true))
      specs[end] = merge(specs[end], (rh = rhook, post = true, cout = w, cin = t))
    else
      rhook = 0
    end
  elseif rhook != 0
    @assert collect(region) == [1] && specs[1].Nx == 0 && specs[end].Ny == 0
    i = rhook > 0 ? 1 : length(specs)
    G = pass_G(specs[i].R, specs[i].s, specs[i].m, 1, specs[i].B, gmax, lim)
    if rhook > 0 || specs[i].s == 1 || (iseven(specs[i].s) && iseven(G))
      specs[i] = merge(specs[i], rhook > 0 ? (cin = rtab, rh = 1) : (cout = rtab, rh = -1))
    else
      rhook = 0
    end
  end
  # Pad an intermediate result when its writer or its reader has a load or store stride ≥ 256 KB
  # and the other one ≥ 32 KB (4096² dim 2 [256, 16]: 512 KB / 32 KB and 8 MB / 8 MB); the
  # index map costs a little in every pass that uses it (1024×1024×32 padded throughout: 1.01×
  # vs 0.94×, but unpadded the dim-3 pass varies from 0.95× to 1.53× between runs).
  function bigstride(sp)
    L = sp.E * (sp.M ÷ sp.R) * sizeof(Complex{T}); S = sp.E * sp.s * sizeof(Complex{T})
    max(L, S) >= 256 << 10 && min(L, S) >= 32 << 10
  end
  padded = [pad > 0 && specs[i].dst in (:a, :b) && (bigstride(specs[i]) || bigstride(specs[i+1])) for i in 1:length(specs)-1]
  push!(padded, false)
  lay(p) = p ? (dims[1], pad) : (1, 0)
  # CUDA launches every pass as a raw kernel compiled here: a call costs about the bare CUDA.jl
  # launch, the KernelAbstractions path about 2 µs more per pass (A6000, small transforms)
  raw = nameof(typeof(backend)) === :CUDABackend
  steps = Any[]
  for (i, sp) in enumerate(specs)
    G = haskey(sp, :G) ? sp.G : pass_G(sp.R, sp.s, sp.m, sp.E, sp.B, gmax, 4 <= get(sp, :rh, 0) <= 7 ? hook_limits(lim, backend) : lim)
    # one pass along a strided dimension whose line count has no good divisor: ragged lanes
    if !haskey(sp, :G) && sp.s == 1 && sp.m == 1 && sp.E > 1 && sp.Nx == 0 && sp.Ny == 0 && get(sp, :rh, 0) == 0 &&
       !(i > 1 && padded[i-1]) && !padded[i] && G < 4 && ragged_G(sp.R, sp.E, sp.B, gmax, lim) > G
      G = ragged_G(sp.R, sp.E, sp.B, gmax, lim)
      sp = merge(sp, (ragged = true,))
    end
    k, W, groups, bytes = pass_kernel(sp.R, G, sp.σ, T, sp.s, sp.M, sp.E, sp.B; Nx = sp.Nx, Ny = sp.Ny,
                                      padx = lay(i > 1 && padded[i-1]), pady = lay(padded[i]), lim = lim,
                                      rhook = get(sp, :rh, 0), pre = get(sp, :pre, get(sp, :rh, 0) == 1),
                                      post = get(sp, :post, get(sp, :rh, 0) == -1), xreal = get(sp, :xreal, false),
                                      blue2 = get(sp, :blue2, false), ragged = get(sp, :ragged, false), raw)
    t = pass_twiddles(T, sp.R, sp.σ, sp.M)
    tw = KA.allocate(backend, Complex{T}, length(t)); copyto!(tw, t)
    if raw || bytes > 0
      # CUDA: compile the raw kernel for exactly the argument types this step launches with
      # (user arrays: the plan's ndims, buffers: vectors)
      cu = cuda_mod()
      d1 = cu.CuArray{Complex{T},1}(undef, (0,))
      dD = cu.CuArray{Complex{T},D}(undef, ntuple(zero, D))
      dcc = Base.invokelatest(getfield, Kernels, :cuda_dyn_compile)
      # DCT hooks read and write the real lines
      rD = cu.CuArray{T,D}(undef, ntuple(zero, D))
      ya = sp.dst === :y ? (get(sp, :post, false) && get(sp, :rh, 0) >= 2 ? rD : dD) : d1
      xa = sp.src === :x ? ((get(sp, :pre, false) && get(sp, :rh, 0) >= 2) || get(sp, :xreal, false) ? rD : dD) : d1
      kk = Base.invokelatest(dcc, k, ya, xa, tw, tw, tw, bytes, bytes > 0 ? 0 : W)
      push!(steps, (RawLaunch(kk, W, groups, bytes, (ya, xa, tw, sp.cin, sp.cout), cu), groups * W, tw, sp.cin, sp.cout, sp.src, sp.dst))
    else
      kk = Base.invokelatest(k, backend, W)
      push!(steps, (kk, groups * W, tw, sp.cin, sp.cout, sp.src, sp.dst))
    end
  end
  n = prod(dims)
  used = Set(x for st in steps for x in st[6:7])
  bufs = Dict{Symbol,Any}()
  for b in (:a, :b); b in used && (bufs[b] = KA.allocate(backend, Complex{T}, n + pad * (n ÷ dims[1]))); end
  for b in (:c, :d); b in used && (bufs[b] = KA.allocate(backend, Complex{T}, scratch)); end
  # the steps hold their scratch arrays themselves (:x and :y stay names of the call's arrays)
  res(b) = b === :x || b === :y ? b : bufs[b]
  steps = Any[(st[1:5]..., res(st[6]), res(st[7])) for st in steps]
  KAPlan{T,D,typeof(backend)}(dims, region, σ, rsall, steps, bufs, backend, gmax, rhook)
end

# Passes with the element-wise hooks 4–7 hold more values per thread. Metal allows only 448 threads
# for the DCT-III hook kernel of 1024² (instead of 1024), so other backends keep the 256 cap. On CUDA
# the cap would force G ≤ 4 in the first pass of a 2^20 DCT-IV and cost up to 1.4× (A6000).
hook_limits(lim::Limits, backend) =
  nameof(typeof(backend)) === :CUDABackend ? lim : Limits(lim.nstat, lim.nmax, min(lim.wmax, 256), lim.slc)

# Padding the intermediate arrays: a pass that reads and writes with a large power-of-two
# stride (64 rows 2 MB apart) runs at a third of the bandwidth beyond the SLC; with padded
# buffers each pass has at most one such side (4096²: 0.83× Metal instead of 1.03×).
default_pad(dims, region, T, prof = :apple, lim = LIM0) = prof === :apple && length(dims) > 1 && any(>(1), region) && beyond_slc(prod(dims) * sizeof(Complex{T}), lim) ? 64 : 0


# y = DFT(x) without synchronization: kernels are queued in order on the backend's stream
# A raw CUDA pass with its parameter block built at plan time: the kernel state and the five
# arguments, of which each call rewrites y and x only. CUDA.jl's call path converts, boxes and
# registers every argument and sets launch attributes per launch, about 2 µs on an A6000, as
# much as the whole kernel of a small transform. Without these CUDA.jl internals, or for
# arguments of other types than the plan's arrays (views), the pass is launched through the
# kernel object.
struct RawLaunch{K,F,A,B,C,L,S}
  kern::K                      # CUDA.jl HostKernel
  fun::F                       # its CuFunction
  W::Int
  groups::Int
  bytes::Int
  ry::Base.RefValue{A}         # device array y of the parameter block
  rx::Base.RefValue{B}         # device array x
  refs::Vector{Any}            # all parameter Refs (kernel state, y, x, tw, cin, cout), kept alive
  ptrs::Vector{Ptr{Cvoid}}
  fast::Bool                   # CUDA.jl internals found
  conv::C                      # cudaconvert
  launch::L                    # cuLaunchKernel
  stream::S                    # current stream
end

function RawLaunch(kern, W, groups, bytes, args, cu)
  lib = parentmodule(typeof(kern))
  fast = isdefined(lib, :cuLaunchKernel) && hasfield(typeof(kern), :state) && hasfield(typeof(kern), :fun)
  ry = Ref(cu.cudaconvert(args[1])); rx = Ref(cu.cudaconvert(args[2]))
  refs = Any[fast ? Ref(kern.state) : Ref(nothing); ry; rx; [Ref(cu.cudaconvert(a)) for a in args[3:end]]]
  RawLaunch(kern, fast ? kern.fun : nothing, W, groups, bytes, ry, rx, refs,
            Ptr{Cvoid}[Base.unsafe_convert(Ptr{Cvoid}, r) for r in refs], fast, cu.cudaconvert,
            fast ? lib.cuLaunchKernel : nothing, cu.stream)
end

function launch_raw!(r::RawLaunch{K,F,A,B}, y, x, tw, cin, cout) where {K,F,A,B}
  if r.fast
    a = r.conv(y); b = r.conv(x)
    if a isa A && b isa B
      r.ry[] = a; r.rx[] = b
      GC.@preserve r y x r.launch(r.fun, r.groups, 1, 1, r.W, 1, 1, r.bytes, r.stream(), r.ptrs, C_NULL)
      return
    end
  end
  r.kern(y, x, tw, cin, cout; threads = r.W, blocks = r.groups, shmem = r.bytes)
  return
end

function launch!(y, x, P::KAPlan)
  isempty(P.steps) && return copyto!(y, x)
  for st in P.steps
    run_step!(st, y, x)
  end
  y
end

# a function barrier per step: the step's tuple type is concrete here
function run_step!((kern, nd, tw, cin, cout, src, dst), y, x)
  s = src === :x ? x : src === :y ? y : src
  d = dst === :x ? x : dst === :y ? y : dst
  if kern isa RawLaunch
    launch_raw!(kern, d, s, tw, cin, cout)
  else
    Base.invokelatest(kern, d, s, tw, cin, cout; ndrange = nd)
  end
  nothing
end

FFTL.plan_ka(backend, ::Type{T}, dims::Tuple, σ::Integer; region = 1:length(dims), kw...) where {T<:FFTL.FFTLFloat} =
  KAPlan(backend, T, Tuple(Int.(dims)), Int(σ); region, kw...)

Base.size(p::KAPlan) = p.sz
Base.ndims(::KAPlan{T,N}) where {T,N} = N
AbstractFFTs.AdjointStyle(::KAPlan) = AbstractFFTs.FFTAdjointStyle()
AbstractFFTs.fftdims(p::KAPlan) = Tuple(p.region)

function Base.show(io::IO, p::KAPlan{T}) where {T}
  print(io, "FFTL KernelAbstractions plan: ", p.σ < 0 ? "forward" : "backward", " Complex{", T, "} ",
        join(p.sz, "×"), ", dims ", Tuple(p.region), ", ", typeof(p.backend).name.name,
        ", passes ", join(string.(p.radices), " "))
end

function LinearAlgebra.mul!(y::AbstractArray{Complex{T}}, p::KAPlan{T}, x::AbstractArray{Complex{T}}) where {T}
  size(x) == p.sz && size(y) == p.sz || throw(DimensionMismatch("plan for $(p.sz), arrays $(size(x)), $(size(y))"))
  y === x && throw(ArgumentError("in-place transforms are not supported: y and x must differ"))
  launch!(y, x, p)
end

Base.:*(p::KAPlan{T}, x::AbstractArray{Complex{T}}) where {T} = mul!(similar(x), p, x)
Base.:*(p::KAPlan{T}, x::AbstractArray{<:Number}) where {T} = p * Complex{T}.(x)

function AbstractFFTs.plan_inv(p::KAPlan{T}) where {T}
  AbstractFFTs.ScaledPlan(KAPlan(p.backend, T, p.sz, -p.σ; region = p.region, gmax = p.gmax),
                          AbstractFFTs.normalization(T, p.sz, p.region))
end

include("ka_real.jl")

end
