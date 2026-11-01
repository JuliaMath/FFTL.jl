# libfftl: FFTL.jl as a trimmed C library (juliac --trim), phase 0 probe. Interface in
# include/fftl.h: plans by handle for one-dimensional complex DFTs of a fixed set of sizes,
# single thread, forward and backward (unnormalised).
#
# A trimmed image has no compiler for new type combinations, so every call must resolve at
# build time. What that means for FFTL (validated with the probes in proto/trim-probes/, the
# details are in notes/libfftl.md):
# - A call into a @generated function resolves only if the generated body contains no direct
#   call to another @generated function. FFTL's stage codelets pass (their bodies call plain
#   SIMD helpers plus, behind @inline, leaf @generated functions like swap_reim); run! does not
#   (its body calls the @generated stage functions), and neither do the generic constructors:
#   Chain{T} applies the plan type with run-time radices, and Bluestein's constructor calls
#   run! on its inner chains.
#   So this file builds everything per variant at build time: the kernel with the table code of
#   src/plan.jl spliced in, and per chain a flat runner — a plain function with the S = 1 body
#   of run! (out of place and in place), put together with FFTL's own chain_body so that only
#   stage!, stage_last! and stage_first_group! are called directly. Keep all of this in sync
#   with src/plan.jl.
# - Closures in compiled code that capture a Type value cannot be applied at run time. The
#   run-time dispatch is therefore an @eval'd straight-line if chain without closures.
# - Comprehensions that capture a local which is reassigned later (the `L = m` pattern of
#   plan.jl's Chain constructor) box that local and every step on it becomes an unresolved call.
#   The construction code therefore goes through typed helpers (stage_tw, first_tw, chirp_vec)
#   with the build-time known lengths as plain arguments.

using FFTL
using FFTL: Chain, Bluestein, cptr, radices, next_smooth, chain_body,
            stage!, stage_last!, stage_first_group!

# FFTL computes no BLAS or LAPACK routine: the library starts without loading OpenBLAS, and
# build.sh leaves it out of the bundle.
using LinearAlgebra
@eval LinearAlgebra.OpenBLAS_jll __init__() = nothing
@eval LinearAlgebra __init__() = nothing

# Variants: element type × sign × size. 101 is a prime larger than MAX_RADIX_PRIME and
# exercises Bluestein (M = next_smooth(2n - 1) = 216); the other sizes are Stockham chains.
const TS = (Float64, Float32)
const SIGNS = (-1, 1)
const SIZES = (256, 1000, 65536, 1048576, 101)

# The radix sequences of the chain variants (sizes 1–4) and of the inner chains of Bluestein
# (size 101), as FFTL.radices picks them on aarch64 for one interleaved sequence (S = 1; the
# element size does not matter there). Checked at load, so a change of the heuristic fails the
# build with an AssertionError instead of a wrong variant table.
const RS = ((4, 8, 8), (4, 10, 5, 5), (4, 4, 4, 4, 4, 4, 4, 4), (4, 4, 4, 4, 4, 4, 4, 4, 4, 4),
            (12, 6, 3))
@assert RS[1] == Tuple(radices(SIZES[1], 1))
@assert RS[2] == Tuple(radices(SIZES[2], 1))
@assert RS[3] == Tuple(radices(SIZES[3], 1))
@assert RS[4] == Tuple(radices(SIZES[4], 1))
@assert RS[5] == Tuple(radices(next_smooth(2 * SIZES[5] - 1), 1))

# One plan of the library: the kernel object and two work buffers of n complex values.
struct Entry{C,CT}
  k::C
  w1::Vector{CT}
  w2::Vector{CT}
end

# Table construction helpers (keep in sync with the Chain{T} and Bluestein{T} constructors of
# src/plan.jl). One typed call per stage with the lengths as arguments; no comprehension of the
# emitted code captures a local that is reassigned later.
function stage_tw(::Type{Complex{T}}, L::Int, r::Int, twc::Int) where {T}
  m = L ÷ r
  vec([Complex{T}(cispi(twc * k * p / L)) for k in 1:r-1, p in 0:m-1])
end

function first_tw(::Type{Complex{T}}, n::Int, r1::Int, twc::Int) where {T}
  vec([Complex{T}(cispi(twc * k * p / n)) for p in 0:n ÷ r1 - 1, k in 1:r1-1])
end

function chirp_vec(::Type{Complex{T}}, n::Int, σ::Int) where {T}
  n2 = 2n
  [Complex{T}(cispi(σ * mod(j * j, n2) / n)) for j in 0:n-1]
end

# Stage input lengths L_i = n ÷ (R_1 ⋯ R_(i-1)) of a chain.
function stage_lengths(n::Int, rs::Tuple)
  Ls = Int[n]
  for r in rs[1:end-1]
    push!(Ls, Ls[end] ÷ r)
  end
  Ls
end

# Expression of the twiddle tuple argument of a Chain constructor.
tw_tuple(CT::Type, Ls::Vector{Int}, rs::Tuple, twc::Int) =
  Expr(:tuple, [:(stage_tw($CT, $L, $r, $twc)) for (L, r) in zip(Ls, rs)]...)

# Destination pointers of run!'s two paths, K stages: out of place the stages alternate between
# py and pw1 and end in py; in place the first stage reads px, the last writes px and the ones
# in between alternate between the work buffers (plan.jl, keep in sync).
function plan_dsts(K::Int)
  oop = [iseven(K - i) ? :py : :pw1 for i in 1:K]
  ip = Any[:px for _ in 1:K]
  for i in K-1:-1:1
    ip[i] = isodd(K - i) ? :pw1 : :pw2
  end
  oop, ip
end

# The kernel type of a variant, computed at build time so that the emitted functions and the
# registries are concrete.
function variant_type(::Type{T}, n::Int, σ::Int, I::Int) where {T}
  if n == SIZES[5]
    rs = RS[5]
    K = length(rs)
    cf = Chain{T,Tuple{rs...},-1,K}
    cb = Chain{T,Tuple{rs...},1,K}
    return Bluestein{T,cf,cb}
  end
  rs = RS[I]
  Chain{T,Tuple{rs...},σ,length(rs)}
end

# One registry per variant; the plan handle is index << 32 | ti << 16 | si << 8 | ni with
# ti, si, ni the positions in TS, SIGNS, SIZES.
for (ti, T) in enumerate(TS), (si, σ) in enumerate(SIGNS), (ni, n) in enumerate(SIZES)
  C = variant_type(T, n, σ, ni)
  CT = Complex{T}
  @eval const $(Symbol(:regs_, ti, :_, si, :_, ni)) = Union{Nothing,Entry{$C,$CT}}[]
end

# Flat runners, one per chain: run!'s body for S = 1, both branches, so that only the
# @generated stage codelets (which do resolve) are called. run! itself cannot be called from
# trimmed code: its generated body calls the @generated stage functions.
let
  for (ti, T) in enumerate(TS), (si, σ) in enumerate(SIGNS), (ni, n) in enumerate(SIZES)
    ni == 5 && continue
    rs = RS[ni]
    K = length(rs)
    C = Chain{T,Tuple{rs...},σ,K}
    oop, ip = plan_dsts(K)
    name = Symbol(:run_, ti, :_, si, :_, ni, :!)
    mkdst(dsts) = quote
      S = 1
      $(chain_body(T, rs, σ, :px, dsts, 1, true)...)
    end
    @eval function $name(py::Ptr{$T}, px::Ptr{$T}, pw1::Ptr{$T}, pw2::Ptr{$T}, P::$C)
      GC.@preserve P begin
        if py != px
          $(mkdst(oop))
        else
          $(mkdst(ip))
        end
      end
      nothing
    end
  end
  # The inner chains of the Bluestein variants: M = 216 points, forward and backward, both
  # independent of the outer sign (plan.jl's Bluestein{T} constructor).
  for (ti, T) in enumerate(TS)
    rs = RS[5]
    K = length(rs)
    for (fi, σ) in enumerate((-1, 1))
      C = Chain{T,Tuple{rs...},σ,K}
      oop, ip = plan_dsts(K)
      name = Symbol(:run_, fi == 1 ? :fwd_ : :bwd_, ti, :!)
      mkdst(dsts) = quote
        S = 1
        $(chain_body(T, rs, σ, :px, dsts, 1, true)...)
      end
      @eval function $name(py::Ptr{$T}, px::Ptr{$T}, pw1::Ptr{$T}, pw2::Ptr{$T}, P::$C)
        GC.@preserve P begin
          if py != px
            $(mkdst(oop))
          else
            $(mkdst(ip))
          end
        end
        nothing
      end
    end
  end
end

# Body of mk_<ti>_<si>_<ni>(): build the kernel of one variant with the table code of
# src/plan.jl, everything length-related spliced in as literals. The convolution kernel of
# Bluestein is transformed at plan time through the in-place branch of the inner runner.
function mk_exprs(::Type{T}, σ::Int, n::Int, ni::Int, ti::Int) where {T}
  CT = Complex{T}
  ex = Expr[]
  if ni == 5
    M = next_smooth(2n - 1)
    rs = RS[5]
    K = length(rs)
    CF = Chain{T,Tuple{rs...},-1,K}
    CB = Chain{T,Tuple{rs...},1,K}
    BT = Bluestein{T,CF,CB}
    nm1 = n - 1
    Ls = stage_lengths(M, rs)
    fwd_tw = tw_tuple(CT, Ls, rs, -2)
    bwd_tw = tw_tuple(CT, Ls, rs, 2)
    rfw = Symbol(:run_fwd_, ti, :!)
    rbw = Symbol(:run_bwd_, ti, :!)
    append!(ex, Expr[
      :(fwd = $CF($M, $fwd_tw, first_tw($CT, $M, $(rs[1]), -2))),
      :(bwd = $CB($M, $bwd_tw, first_tw($CT, $M, $(rs[1]), 2))),
      :(w1 = Vector{$CT}(undef, $M)),
      :(w2 = Vector{$CT}(undef, $M)),
      :(chirp = chirp_vec($CT, $n, $σ)),
      :(b = zeros($CT, $M)),
      :(b[1] = conj(chirp[1])),
      :(for j in 1:$nm1
          b[j + 1] = b[$M - j + 1] = conj(chirp[j + 1])
        end),
      :(GC.@preserve b w1 w2 $rfw(cptr(b), cptr(b), cptr(w1), cptr(w2), fwd)),
      :(for j in 1:$M
          b[j] = b[j] / $M
        end),
      :(k = $BT($n, chirp, b, fwd, bwd, zeros($CT, $M), w1, w2)),
    ])
  else
    rs = RS[ni]
    K = length(rs)
    C = Chain{T,Tuple{rs...},σ,K}
    twc = 2σ
    Ls = stage_lengths(n, rs)
    tws = tw_tuple(CT, Ls, rs, twc)
    push!(ex, :(k = $C($n, $tws, first_tw($CT, $n, $(rs[1]), $twc))))
  end
  push!(ex, :(Entry(k, Vector{$CT}(undef, $n), Vector{$CT}(undef, $n))))
  ex
end

# Body of dft_<ti>_<si>_<ni>!(): the DFT of one plan. For a chain the runner with the plan's
# work buffers; for Bluestein the single-lane form of run_lanes! (Sx = Sy = B = 1, keep in sync
# with plan.jl), with the inner chains through their flat runners.
function dft_exprs(::Type{T}, σ::Int, n::Int, ni::Int, ti::Int, si::Int) where {T}
  CT = Complex{T}
  if ni == 5
    M = next_smooth(2n - 1)
    np1 = n + 1
    rfw = Symbol(:run_fwd_, ti, :!)
    rbw = Symbol(:run_bwd_, ti, :!)
    Expr[
      :(k = e.k),
      :(y = unsafe_wrap(Array, Ptr{$CT}(out), $n)),
      :(x = unsafe_wrap(Array, Ptr{$CT}(in_), $n)),
      :(buf = k.buf),
      :(chirp = k.chirp),
      :(kern = k.kernel),
      :(w1 = k.work1),
      :(w2 = k.work2),
      :(GC.@preserve k y x begin
          for j in 1:$n
            buf[j] = x[j] * chirp[j]
          end
          for j in $np1:$M
            buf[j] = zero($CT)
          end
          $rfw(cptr(buf), cptr(buf), cptr(w1), cptr(w2), k.fwd)
          for j in 1:$M
            buf[j] = buf[j] * kern[j]
          end
          $rbw(cptr(buf), cptr(buf), cptr(w1), cptr(w2), k.bwd)
          for j in 1:$n
            y[j] = buf[j] * chirp[j]
          end
        end),
    ]
  else
    run = Symbol(:run_, ti, :_, si, :_, ni, :!)
    Expr[
      :(y = unsafe_wrap(Array, Ptr{$CT}(out), $n)),
      :(x = unsafe_wrap(Array, Ptr{$CT}(in_), $n)),
      :(GC.@preserve y x begin
          $run(cptr(y), cptr(x), cptr(e.w1), cptr(e.w2), e.k)
        end),
    ]
  end
end

# mk_* and dft_*: one plain function per variant, eval'd at build time.
for (ti, T) in enumerate(TS), (si, σ) in enumerate(SIGNS), (ni, n) in enumerate(SIZES)
  CT = Complex{T}
  C = variant_type(T, n, σ, ni)
  eval(Expr(:function, Expr(:call, Symbol(:mk_, ti, :_, si, :_, ni)),
            Expr(:block, mk_exprs(T, σ, n, ni, ti)...)))
  dsig = Expr(:call, Symbol(:dft_, ti, :_, si, :_, ni, :!),
              Expr(:(::), :e, Entry{C,CT}), Expr(:(::), :out, Ptr{Cvoid}),
              Expr(:(::), :in_, Ptr{Cvoid}))
  eval(Expr(:function, dsig, Expr(:block, dft_exprs(T, σ, n, ni, ti, si)...)))
end

# The handle is index << 32 | ti << 16 | si << 8 | ni.
unhandle(h) = (Int((h >> 16) & 0xff), Int((h >> 8) & 0xff), Int(h & 0xff), Int(h >> 32))

# Plans can be created and destroyed from any thread; a plan must not run on two threads at once.
const REGISTRY_LOCK = ReentrantLock()

const LAST_ERROR = Ref("")
function set_last_error(e::Union{ArgumentError,DimensionMismatch,ErrorException})
  m = e.msg
  m isa String ? (LAST_ERROR[] = m) : (LAST_ERROR[] = "operation failed")
  nothing
end

# Run-time dispatch: one straight-line branch per variant, generated at build time. The
# variants_* functions are called from the @ccallable interface below; unknown combinations
# throw (create) or report an invalid handle (exec, destroy).
let
  ex = :(throw(ArgumentError("unsupported variant")))
  for (ti, T) in enumerate(TS), (si, σ) in enumerate(SIGNS), (ni, n) in enumerate(SIZES)
    reg = Symbol(:regs_, ti, :_, si, :_, ni)
    mk = Symbol(:mk_, ti, :_, si, :_, ni)
    body = quote
      idx = @lock REGISTRY_LOCK begin
        push!($reg, $mk())
        length($reg)
      end
      Int64(idx - 1) << 32 | Int64($ti) << 16 | Int64($si) << 8 | Int64($ni)
    end
    global ex2 = :(tt == $ti && ss == $σ && nn == $n ? $body : $ex)
    ex = ex2
  end
  @eval variants_create(tt::Int, ss::Int, nn::Int) = $ex
end

let
  ex = quote
    LAST_ERROR[] = "invalid plan handle"
    Cint(-1)
  end
  for (ti, T) in enumerate(TS), (si, σ) in enumerate(SIGNS), (ni, n) in enumerate(SIZES)
    reg = Symbol(:regs_, ti, :_, si, :_, ni)
    dft = Symbol(:dft_, ti, :_, si, :_, ni, :!)
    body = quote
      e = @lock REGISTRY_LOCK (0 <= idx <= length($reg) - 1 ? $reg[idx + 1] : nothing)
      if e === nothing
        LAST_ERROR[] = "invalid plan handle"
        Cint(-1)
      else
        $dft(e, out, in_)
        Cint(0)
      end
    end
    global ex2 = :(ti == $ti && si == $si && ni == $ni ? $body : $ex)
    ex = ex2
  end
  @eval function variants_exec(ti::Int, si::Int, ni::Int, idx::Int, out::Ptr{Cvoid},
                               in_::Ptr{Cvoid})::Cint
    $ex
  end
end

let
  ex = quote
    LAST_ERROR[] = "invalid plan handle"
    Cint(-1)
  end
  for (ti, T) in enumerate(TS), (si, σ) in enumerate(SIGNS), (ni, n) in enumerate(SIZES)
    reg = Symbol(:regs_, ti, :_, si, :_, ni)
    body = quote
      @lock REGISTRY_LOCK begin
        0 <= idx <= length($reg) - 1 && ($reg[idx + 1] = nothing)
      end
      Cint(0)
    end
    global ex2 = :(ti == $ti && si == $si && ni == $ni ? $body : $ex)
    ex = ex2
  end
  @eval function variants_destroy(ti::Int, si::Int, ni::Int, idx::Int)::Cint
    $ex
  end
end

Base.@ccallable function fftl_plan_create(t::Cint, size::Clonglong, sign::Cint)::Clonglong
  try
    variants_create(Int(t), Int(sign), Int(size))
  catch e
    if e isa ArgumentError || e isa DimensionMismatch || e isa ErrorException
      set_last_error(e)
    else
      LAST_ERROR[] = "plan creation failed"
    end
    Int64(-1)
  end
end

Base.@ccallable function fftl_plan_dft(h::Clonglong, out::Ptr{Cvoid}, in_::Ptr{Cvoid})::Cint
  ti, si, ni, idx = unhandle(Int64(h))
  variants_exec(ti, si, ni, idx, out, in_)
end

Base.@ccallable function fftl_plan_destroy(h::Clonglong)::Cint
  ti, si, ni, idx = unhandle(Int64(h))
  variants_destroy(ti, si, ni, idx)
end

# The message of the last failed call, for fftl_last_error.
Base.@ccallable function fftl_last_error(buf::Ptr{UInt8}, len::Csize_t)::Csize_t
  m = codeunits(LAST_ERROR[])
  nerr = min(length(m), Int(len) - 1)
  if nerr >= 0
    for i in 1:nerr
      unsafe_store!(buf, m[i], i)
    end
    unsafe_store!(buf, 0x00, nerr + 1)
  end
  Csize_t(length(m))
end
