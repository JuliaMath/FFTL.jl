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

const NMAX = 4096           # complex Float32 values in 32 KB of threadgroup memory

rename(ex::Expr, suf) = Expr(ex.head, (rename(a, suf) for a in ex.args)...)
rename(s::Symbol, suf) = (str = string(s); (startswith(str, "c") || startswith(str, "a")) && length(str) > 1 && all(isdigit, str[2:end]) ? Symbol(str, "_", suf) : s)
rename(x, suf) = x

const PKERNELS = Dict{Any,Any}()
const PKERNELS_LOCK = ReentrantLock()
# Radix 8 from 512 on: twice the threads per workgroup (batch 4096 × 1024: 0.95× Metal
# instead of 1.36×, batch 512 × 4096: 0.89× instead of 1.08×); below, radix 16.
lradix(n) = n >= 512 ? 8 : 16
const TWLEV = 3             # number of factor tables of W_N^t

# Radices of a chain of length n: the power-of-two part as radix-R codelets and one remainder
# (first), then the odd part: pairs of 3 as 9, other primes (≤ FFTL.MAX_RADIX_PRIME) alone.
function local_radices(n, R = lradix(n))
  r = Int[]; m = n >> trailing_zeros(n); p2 = n ÷ m
  while p2 >= R; push!(r, R); p2 ÷= R; end
  p2 > 1 && push!(r, p2)
  f, rest = FFTL.factorize(m)
  @assert rest == 1 "prime factor > $(FFTL.MAX_RADIX_PRIME)"
  e3 = get(f, 3, 0)
  append!(r, fill(9, e3 ÷ 2)); isodd(e3) && push!(r, 3)
  for p in sort!(collect(keys(f))); p == 3 || append!(r, fill(p, f[p])); end
  r
end

# offset of the four-step factor tables behind the local twiddles
local_twcount(n, rs) = sum((R - 1) * (L ÷ R) for (R, L) in zip(rs, accumulate(÷, [n; rs[1:end-1]])); init = 0)

# Hooks for Bluestein along a dimension (Nx, Ny > 0): the first pass (s = 1) reads element i of
# an input of length Nx along the dimension times cin[i], zero for i ≥ Nx; the last pass (m = 1)
# writes element i < Ny of an output of length Ny along the dimension times cout[i].
function pass_kernel(R, G, σ, T, s, N, E = 1, B = 1; Nx = 0, Ny = 0, padx = (1, 0), pady = (1, 0))
  @lock PKERNELS_LOCK get!(PKERNELS, (R, G, σ, T, s, N, E, B, Nx, Ny, padx, pady)) do
    # padded layouts (n1, pad): logical index L at L + pad ⌊L / n1⌋
    PX = Int32.(padx); PY = Int32.(pady)
    rs = local_radices(R)
    m = N ÷ (R * s)
    sp = E * s                           # stride of q in memory
    Lp = E * N ÷ R                       # lanes per batch item
    batchlanes = Lp == 1 && G > 1        # one-pass transforms with E = 1: lanes are batch items
    @assert R * m * s == N
    @assert batchlanes ? B % G == 0 : (Lp % G == 0 && (sp > 1 || m % G == 0))
    crossing = sp > 1 && sp % G != 0     # lanes of a workgroup span several p: p, q per lane
    Rmax = maximum(rs)
    nel = G * R
    W = cld(nel, Rmax)                   # threads per workgroup
    transposed = (sp == 1 && m > 1) || batchlanes   # output runs y[R b + k]
    twiddle = m > 1
    @assert 1 <= W <= 1024 && nel <= NMAX
    @assert (Nx == 0 || s == 1) && (Ny == 0 || (m == 1 && (batchlanes || !transposed)))
    ies = Lp
    ebits = cld(8 * sizeof(Int) - leading_zeros(N - 1), TWLEV)
    offw = local_twcount(R, rs)
    I(v) = Int32(v)
    function twid(t)
      f = [:(@inbounds tw[$(I(offw + 1 + (i << ebits))) + ((t >> $(I(i * ebits))) & $(I((1 << ebits) - 1)))]) for i in 0:TWLEV-1]
      :(t = $t; v = ccmul(v, $(foldl((a, b) -> :(ccmul($a, $b)), f))))
    end
    lb(i) = :(buf[$i + one(Int32)])
    # lane-major output: y[ybase + l + sp k], twiddle exponent k p s with p = b0 ÷ sp
    lanemajor_store(l, k) = Ny == 0 && crossing ? quote
      bl = b0 + $l; pl = bl ÷ $(I(sp)); ql = bl - pl * $(I(sp))
      $(twiddle ? twid(:($k * pl * $(I(s)))) : nothing)
      @inbounds y[padmap(obase + ql + $(I(sp)) * ($(I(R)) * pl + $k), $(PY[1]), $(PY[2])) + one(Int32)] = v
    end : Ny == 0 ? quote
      $(twiddle ? twid(:($k * pss)) : nothing)
      @inbounds y[padmap(ybase + $l + $(I(sp)) * $k, $(PY[1]), $(PY[2])) + one(Int32)] = v
    end : quote
      # m = 1: lane b0 + l = e + E q, element q + s k along the dimension
      bl = b0 + $l; ql = bl ÷ $(I(E)); el = bl - ql * $(I(E)); ii = ql + $(I(s)) * $k
      if ii < $(I(Ny))
        @inbounds y[padmap(yob + el + $(I(E)) * ii, $(PY[1]), $(PY[2])) + one(Int32)] = ccmul(v, cout[ii + one(Int32)])
      end
    end
    body = Any[]
    if batchlanes
      # G transforms stored one after another: contiguous load, element j of lane l to l + G j
      for i in 0:cld(nel, W)-1
        push!(body, quote
          idx = tid + $(I(W * i))
          if idx < $(I(nel))
            j = idx % $(I(R)); l = idx ÷ $(I(R))
            $(Nx == 0 ? :(@inbounds $(lb(:(l + $(I(G)) * j))) = x[padmap(xbase + idx, $(PX[1]), $(PX[2])) + one(Int32)]) :
              :(@inbounds $(lb(:(l + $(I(G)) * j))) = j < $(I(Nx)) ? ccmul(x[padmap((b0 + l) * $(I(Nx)) + j, $(PX[1]), $(PX[2])) + one(Int32)], cin[j + one(Int32)]) : zero(VT)))
          end
        end)
      end
      push!(body, :(@synchronize))
    end
    L = R; st = G; off = 0
    for (si, r) in enumerate(rs)
      mm = L ÷ r
      first = si == 1; last = si == length(rs)
      fload = first && !batchlanes
      nb = nel ÷ r                       # butterflies of this stage
      cnt = cld(nb, W)
      # the last round may run past nb: compute a clamped butterfly, skip its stores
      guard(c, ex) = W * (c + 1) > nb ? :(if tid + $(I(W * c)) < $(I(nb)); $ex; end) : ex
      stmts, a, X = FFTL.codelet(r, σ)
      stmts = [retype(e, T) for e in stmts]
      for c in 0:cnt-1
        pc = Symbol("p_", c); qc = Symbol("q_", c)
        gc = W * (c + 1) > nb ? :(min(tid + $(I(W * c)), $(I(nb - 1)))) : :(tid + $(I(W * c)))
        push!(body, :(g = $gc), :($pc = g ÷ $(I(st))), :($qc = g - $pc * $(I(st))))
        for j in 1:r
          # first local stage: st = G, so local index q + G (p + j mm) is lane q, element p + j mm
          src = !fload ? lb(:($qc + $(I(st)) * ($pc + $(I((j - 1) * mm))))) :
                Nx == 0 ? :(x[padmap(xbase + $qc + $(I(ies)) * ($pc + $(I((j - 1) * mm))), $(PX[1]), $(PX[2])) + one(Int32)]) :
                quote   # s = 1: lane b0 + q = e + E p, element p + m jR along the dimension
                  bl = b0 + $qc
                  pl = bl ÷ $(I(E)); el = bl - pl * $(I(E))
                  ii = pl + $(I(m)) * ($pc + $(I((j - 1) * mm)))
                  $(Nx >= N ? :(ccmul(x[padmap(xob + el + $(I(E)) * ii, $(PX[1]), $(PX[2])) + one(Int32)], cin[ii + one(Int32)])) :
                    :(ii < $(I(Nx)) ? ccmul(x[padmap(xob + el + $(I(E)) * ii, $(PX[1]), $(PX[2])) + one(Int32)], cin[ii + one(Int32)]) : zero(VT)))
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
          if last && !transposed
            # mm = 1: local index q + k st = l + G kel
            push!(stores, quote
              v = $v
              l = $qc % $(I(G)); kel = $qc ÷ $(I(G)) + $(I(k * (R ÷ r)))
              $(lanemajor_store(:l, :kel))
            end)
          else
            val = (last || k == 0) ? v : :(ccmul($v, @inbounds tw[$(I(off + 1)) + $(I(r - 1)) * $pc + $(I(k - 1))]))
            push!(stores, :(@inbounds $(lb(:(o + $(I(k * st))))) = $val))
          end
        end
        push!(body, guard(c, Expr(:block, stores...)))
      end
      last && !transposed || push!(body, :(@synchronize))
      off += (r - 1) * mm
      L = mm; st *= r
    end
    if transposed
      for i in 0:cld(nel, W)-1
        push!(body, quote
          idx = tid + $(I(W * i))
          if idx < $(I(nel))
            k = idx % $(I(R)); l = idx ÷ $(I(R))
            v = @inbounds $(lb(:(l + $(I(G)) * k)))
            $(twiddle ? twid(:((b0 + l) * k)) : nothing)
            $(Ny == 0 ? :(@inbounds y[padmap(obase + $(I(R)) * (b0 + l) + k, $(PY[1]), $(PY[2])) + one(Int32)] = v) :
              :(if k < $(I(Ny)); @inbounds y[padmap((b0 + l) * $(I(Ny)) + k, $(PY[1]), $(PY[2])) + one(Int32)] = ccmul(v, cout[k + one(Int32)]); end))
          end
        end)
      end
    end
    gpb = batchlanes ? 1 : Lp ÷ G        # workgroups per batch item
    name = Symbol(crossing ? "ka_passx_" : "ka_pass_", R, "_", join(rs, "x"), "_G", G, "_s", s, "_N", N, "_E", E, "_B", B, "_",
                  σ < 0 ? "f" : "b", "_", T, "_x", Nx, "_y", Ny, padx[2] > 0 ? "_px$(padx[2])" : "", pady[2] > 0 ? "_py$(pady[2])" : "")
    ex = quote
      @kernel function $name(y, @Const(x), @Const(tw), @Const(cin), @Const(cout))
        tid = (@index(Local, Linear) - 1) % Int32
        grp = (@index(Group, Linear) - 1) % Int32
        ob = $(batchlanes ? :(zero(Int32)) : :(grp ÷ $(I(gpb))))
        b0 = (grp - ob * $(I(gpb))) * $(I(G))
        obase = ob * $(I(E * N))
        xbase = $(batchlanes ? :(b0 * $(I(N))) : :(obase + b0))
        p0 = b0 ÷ $(I(sp))
        ybase = obase + b0 - p0 * $(I(sp)) + $(I(sp * R)) * p0
        pss = p0 * $(I(s))
        xob = ob * $(I(E * Nx))
        yob = ob * $(I(E * Ny))
        VT = Complex{$T}
        buf = @localmem Complex{$T} ($nel,)
        $(body...)
      end
    end
    Core.eval(Kernels, ex)
    # the kernel exists from the next world age on
    (Base.invokelatest(getfield, Kernels, name), W, batchlanes ? B ÷ G : B * gpb)
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

# largest G ≤ lim dividing d
divisor_le(d, lim) = maximum(g for g in 1:min(d, lim) if d % g == 0)

# G for a pass: G R ≤ NMAX, G ≤ gmax, G | sp (or G | m when sp = 1, G | B for batch lanes)
function pass_G(R, s, m, E, B, gmax)
  N = R * m * s
  lim = min(NMAX ÷ R, gmax, E > 1 && R == N ? 16 : gmax)   # one-pass strided dimension: G ≤ 16 measured best
  sp = E * s
  # one-pass transforms of contiguous data: about 512 elements per workgroup measured best
  E * m * s == 1 && return divisor_le(B, min(lim, max(1, 512 ÷ R)))
  sp == 1 && return divisor_le(m, lim)
  # G dividing sp keeps p fixed per workgroup; otherwise lanes may span several p (any G | sp m)
  g1 = divisor_le(sp, lim); g2 = divisor_le(sp * m, lim)
  2g1 >= g2 ? g1 : g2
end

# Pass lengths (powers of two). A last pass of radix ≤ 16 needs no exchange in threadgroup
# memory; on the M1 a last pass with exchange (large store stride) runs at half bandwidth
# once the data no longer fit the system level cache.
function default_passes(N)
  ispow2(N) || return mixed_passes(N)
  e = trailing_zeros(N)
  e <= 12 && return [N]
  e <= 21 && return [1 << cld(e, 2), 1 << (e ÷ 2)]       # 2^21 complex F32 still fit the SLC
  e <= 23 && return [1024, 1 << (e - 14), 16]
  P = cld(e, 6)
  [1 << (e ÷ P + (i <= e % P)) for i in 1:P]
end

# Pass lengths along a dimension with inner size E. Strided dimensions (E > 1) beyond the SLC:
# on the M1 a pass whose workgroups touch more than ~32 rows far apart (load and store) runs
# at a fraction of the bandwidth, radix ≤ 16 passes run at copy speed.
const SLC_BYTES = 40 << 20  # M1 Max system level cache: 48 MB
function dim_passes(N, E, bytes; padded = false, first = true, last = true)
  E == 1 && return default_passes(N)
  if bytes > SLC_BYTES && padded
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
  bytes > SLC_BYTES && return pack_passes(N, 16)
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
function mixed_passes(N)
  N <= NMAX && return [N]
  o = N >> trailing_zeros(N)
  if o <= 256 && N >= 1 << 19
    a = o; p2 = N ÷ o
    while a < 12 && p2 > 1; a *= 2; p2 ÷= 2; end
    p2 > 1 && return [a; p2 <= NMAX ? [p2] : default_passes(p2)]
  end
  P = N <= 1 << 21 ? 2 : N <= 1 << 24 ? 3 : cld(ceil(Int, log2(N)), 6)
  if P == 2
    cands = [[d, N ÷ d] for d in 1:NMAX if N % d == 0 && N ÷ d <= NMAX]
    rs = argmax(c -> (minimum(passes_G(N, c)), -abs(log(c[1] / c[2]))), cands)
  else
    rs = balance_passes(N, P)
  end
  minimum(passes_G(N, rs)) >= 4 && return rs
  rs2 = balance_passes(N, P + 1)
  minimum(passes_G(N, rs2)) > minimum(passes_G(N, rs)) ? rs2 : rs
end

passes_G(N, rs) = (s = 1; [(G = pass_G(R, s, N ÷ (R * s), 1, 1, 64); s *= R; G) for R in rs])

# N as few pass lengths ≤ cap as first-fit decreasing packing of the prime factors finds
function pack_passes(N, cap)
  f, rest = FFTL.factorize(N); @assert rest == 1
  bins = Int[]
  for p in sort!(reduce(vcat, [fill(p, k) for (p, k) in f]; init = Int[]); rev = true)
    i = findfirst(b -> b * p <= cap, bins)
    i === nothing ? push!(bins, p) : (bins[i] *= p)
  end
  sort!(bins; rev = true)
end

# N as a product of P pass lengths ≤ NMAX: prime factors, largest first, each to the pass with
# the smallest product (more passes if a length exceeds NMAX)
function balance_passes(N, P)
  f, rest = FFTL.factorize(N); @assert rest == 1
  ps = sort!(reduce(vcat, [fill(p, k) for (p, k) in f]; init = Int[]); rev = true)
  while true
    bins = ones(Int, P)
    for p in ps; bins[argmin(bins)] *= p; end
    maximum(bins) <= NMAX && return sort!(bins; rev = true)
    P += 1
  end
end

# Bluestein along a dimension of length N (prime factor > FFTL.MAX_RADIX_PRIME):
# X_k = c_k Σ_j (x_j c_j) conj(c_{k-j}), c_j = exp(σ iπ j²/N), as a cyclic convolution of length
# M ≥ 2N - 1. Forward passes of length M read x times c (load hook, zero padding), the first
# backward pass reads times b̂ = FFT_M(conj(c) wrapped) / M, the last one writes the first N
# elements times c (store hook): 2P launches, scratch arrays of E·M·B elements.
bluestein_length(n) = minimum(o << max(0, ceil(Int, log2(n / o))) for o in (1, 3, 5, 7, 9, 15))

function bluestein_tables(backend, T, N, σ)
  M = bluestein_length(2N - 1)
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
  steps::Vector{Any}            # (kernel instance, ndrange, tw, cin, cout, src, dst)
  bufs::Dict{Symbol,Any}        # :a, :b array-shaped; :c, :d Bluestein scratch
  backend::B
  gmax::Int
  pinv::AbstractFFTs.Plan       # set by AbstractFFTs.inv on first use
  KAPlan{T,N,B}(sz, region, σ, radices, steps, bufs, backend, gmax) where {T,N,B} =
    new{T,N,B}(sz, region, σ, radices, steps, bufs, backend, gmax)
end

# DFT along the dimensions `region` of an array of size `dims` (column major). The buffers :a
# and :b may use a padded layout (`pad` elements after each run of dims[1]).
function KAPlan(backend, ::Type{T}, dims::NTuple{D,Int}, σ; region = 1:D, radices = nothing, gmax = 64,
                pad = default_pad(dims, region, T)) where {T,D}
  specs = Any[]; rsall = Vector{Int}[]
  dummy = KA.allocate(backend, Complex{T}, 1)
  cur = :x; scratch = 0
  arr_next(src) = src === :a ? :b : :a
  scr_next(src) = src === :c ? :d : :c
  for (i, d) in enumerate(region)
    N = dims[d]; E = prod(dims[1:d-1]; init = 1); B = prod(dims[d+1:end]; init = 1)
    blue = FFTL.factorize(N)[2] > 1
    M, cin, bhat = blue ? bluestein_tables(backend, T, N, σ) : (N, dummy, dummy)
    rs = radices === nothing ? dim_passes(M, E, E * M * B * sizeof(Complex{T}); padded = pad > 0, first = i == 1, last = i == length(region)) : radices[i]
    @assert prod(rs) == M
    push!(rsall, collect(rs))
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
  steps = Any[]
  for (i, sp) in enumerate(specs)
    G = pass_G(sp.R, sp.s, sp.m, sp.E, sp.B, gmax)
    k, W, groups = pass_kernel(sp.R, G, sp.σ, T, sp.s, sp.M, sp.E, sp.B; Nx = sp.Nx, Ny = sp.Ny,
                               padx = lay(i > 1 && padded[i-1]), pady = lay(padded[i]))
    kk = Base.invokelatest(k, backend, W)
    t = pass_twiddles(T, sp.R, sp.σ, sp.M)
    tw = KA.allocate(backend, Complex{T}, length(t)); copyto!(tw, t)
    push!(steps, (kk, groups * W, tw, sp.cin, sp.cout, sp.src, sp.dst))
  end
  n = prod(dims)
  used = Set(x for st in steps for x in st[6:7])
  bufs = Dict{Symbol,Any}()
  for b in (:a, :b); b in used && (bufs[b] = KA.allocate(backend, Complex{T}, n + pad * (n ÷ dims[1]))); end
  for b in (:c, :d); b in used && (bufs[b] = KA.allocate(backend, Complex{T}, scratch)); end
  KAPlan{T,D,typeof(backend)}(dims, collect(Int, region), σ, rsall, steps, bufs, backend, gmax)
end

# Padding the intermediate arrays: a pass that reads and writes with a large power-of-two
# stride (64 rows 2 MB apart) runs at a third of the bandwidth beyond the SLC; with padded
# buffers each pass has at most one such side (4096²: 0.83× Metal instead of 1.03×).
default_pad(dims, region, T) = length(dims) > 1 && any(>(1), region) && prod(dims) * sizeof(Complex{T}) > SLC_BYTES ? 64 : 0


# y = DFT(x) without synchronization: kernels are queued in order on the backend's stream
function launch!(y, x, P::KAPlan)
  get(b) = b === :x ? x : b === :y ? y : P.bufs[b]
  for (kk, nd, tw, cin, cout, src, dst) in P.steps
    Base.invokelatest(kk, get(dst), get(src), tw, cin, cout; ndrange = nd)
  end
  y
end

FFTL.plan_ka(backend, ::Type{T}, dims::Tuple, σ::Integer; region = 1:length(dims), kw...) where {T<:FFTL.FFTLFloat} =
  KAPlan(backend, T, Tuple(Int.(dims)), Int(σ); region, kw...)

Base.size(p::KAPlan) = p.sz
Base.ndims(::KAPlan{T,N}) where {T,N} = N
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

end
