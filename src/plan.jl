# Kernels for one dimension: a Stockham chain for lengths whose prime factors are
# ≤ MAX_RADIX_PRIME, Bluestein's algorithm on top of two chains otherwise. Both transform a batch
# of S interleaved sequences (element j of lane i at offset i + S·j), see kernels.jl.

abstract type Kernel{T} end

"""
    Chain{T,Rs,σ,K}

Stockham chain with radix sequence `Rs` (a tuple, part of the type so that the stage sequence is
compiled into one function) and sign `σ`.
"""
struct Chain{T,Rs,σ,K} <: Kernel{T}
  n::Int
  tw::NTuple{K,Vector{Complex{T}}}    # stage i: w^(kp) at (R-1)p + k
  tw1::Vector{Complex{T}}             # first stage: w^(kp) at (k-1)m + p
end

function Chain{T}(n::Int, σ::Int, rs::AbstractVector{Int} = radices(n)) where {T}
  prod(rs; init = 1) == n || throw(ArgumentError("radices $rs do not multiply to $n"))
  tw = Vector{Complex{T}}[]
  L = n
  for r in rs
    m = L ÷ r
    push!(tw, vec([Complex{T}(cispi(2σ * k * p / L)) for k in 1:r-1, p in 0:m-1]))
    L = m
  end
  r1 = isempty(rs) ? 1 : rs[1]
  tw1 = vec([Complex{T}(cispi(2σ * k * p / n)) for p in 0:n÷r1-1, k in 1:r1-1])
  Chain{T,Tuple(rs),σ,length(rs)}(n, Tuple(tw), tw1)
end

"""
    Pruning{FM,LM}

Pruned first and last stage of a chain on lines with zero inputs or unneeded outputs: the first
stage runs segment i over p ∈ `first[i]` with zero-input mask FM[i], the last stage segment i over
q ∈ `last[i]` (q0 in lane blocks) with needed-output mask LM[i]. The masks are type parameters so
that each segment calls its own generated codelet; an empty tuple means no pruning there.
"""
struct Pruning{FM,LM}
  first::Vector{UnitRange{Int}}
  last::Vector{UnitRange{Int}}
end
const NOPRUNE = Pruning{(),()}(UnitRange{Int}[], UnitRange{Int}[])

# Stage calls of a chain from `src` through `dsts` (one destination per stage). `base` is a
# number of complex values known to divide S (1 if nothing is known); stage i has stride
# S·(R_1⋯R_(i-1)) and takes the widest vector type whose complex count divides base·R_1⋯R_(i-1).
# With `single` (S = 1) the first stage groups 2 or 4 butterflies in one vector when m allows.
function chain_body(T, Rs, σ, src, dsts, base, single, FM = (), LM = ())
  K = length(Rs)
  n = prod(Rs; init = 1)
  ladder = vec_ladder(T)
  # vector type of a grouped first stage: the widest with 2 or 4 complex values dividing m
  group = findfirst(V -> ncomplex(V) in (2, 4) && (n ÷ Rs[1]) % ncomplex(V) == 0, ladder)
  body = Any[:(L = P.n), :(s = S)]
  sst = 1
  for i in 1:K
    r = Rs[i]
    dst = dsts[i]
    V = ladder[findfirst(V -> (base * sst) % ncomplex(V) == 0, ladder)]
    if i == K
      if isempty(LM) || K == 1
        push!(body, :(stage_last!($dst, $src, s, Val($r), Val($σ), $V)))
      else
        for (j, need) in enumerate(LM)
          push!(body, :(stage_last!($dst, $src, s, first(Pr.last[$j]), last(Pr.last[$j]) + 1, Val($r), Val($σ), $V, Val($need))))
        end
      end
    elseif single && i == 1 && group !== nothing && ncomplex(V) == 1
      push!(body, :(m = L ÷ $r))
      if isempty(FM)
        push!(body, :(stage_first_group!($dst, $src, cptr(P.tw1), m, Val($r), Val($σ), $(ladder[group]))))
      else
        for (j, zin) in enumerate(FM)
          push!(body, :(stage_first_group!($dst, $src, cptr(P.tw1), m, first(Pr.first[$j]), last(Pr.first[$j]) + 1,
                                           Val($r), Val($σ), $(ladder[group]), Val($zin))))
        end
      end
      push!(body, :(L = m), :(s *= $r))
    else
      push!(body, :(m = L ÷ $r))
      if i > 1 || isempty(FM)
        push!(body, :(stage!($dst, $src, cptr(P.tw[$i]), m, s, Val($r), Val($σ), $V)))
      else
        for (j, zin) in enumerate(FM)
          push!(body, :(stage!($dst, $src, cptr(P.tw[$i]), m, s, first(Pr.first[$j]), last(Pr.first[$j]) + 1,
                               Val($r), Val($σ), $V, Val($zin))))
        end
      end
      push!(body, :(L = m), :(s *= $r))
    end
    src = dst
    sst *= r
  end
  body
end

# Branches on S: S = 1, then the complex counts of the ladder (widest first) that divide S.
function stride_dispatch(T, body)
  ex = body(1, false)
  for V in reverse(vec_ladder(T))
    C = ncomplex(V)
    C == 1 && continue
    ex = :(S % $C == 0 ? $(body(C, false)) : $ex)
  end
  :(S == 1 ? $(body(1, true)) : $ex)
end

"""
    run!(py, px, pw1, pw2, P::Kernel, S)

Transform S interleaved sequences of length `P.n` from `px` into `py` (which may equal `px`),
using the work buffers `pw1`, `pw2` (S·n complex values each).
"""
run!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::Chain{T}, S::Int) where {T} =
  run!(py, px, pw1, pw2, P, S, NOPRUNE)

@generated function run!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::Chain{T,Rs,σ,K}, S::Int,
                         Pr::Pruning{FM,LM}) where {T,Rs,σ,K,FM,LM}
  K == 0 && return :(py == px || unsafe_copyto!(py, px, 2S); nothing)
  # Out of place: the stages alternate between py and pw1 and end in py; px is only read.
  oop = [iseven(K - i) ? :py : :pw1 for i in 1:K]
  # In place: the first stage reads px, the last writes px, the ones between alternate between
  # the two work buffers. A single stage goes through pw1.
  ip = Any[:px for _ in 1:K]
  for i in K-1:-1:1
    ip[i] = isodd(K - i) ? :pw1 : :pw2
  end
  dispatch(dsts) = stride_dispatch(T, (base, single) -> quote
    $(chain_body(T, Rs, σ, :px, dsts, base, single, FM, LM)...)
  end)
  inplace = K == 1 ? quote
    $(dispatch([:pw1]))
    unsafe_copyto!(px, pw1, 2 * S * P.n)
  end : dispatch(ip)
  quote
    GC.@preserve P begin
      if py != px
        $(dispatch(oop))
      else
        $inplace
      end
    end
    nothing
  end
end

"""
    Bluestein{T}

DFT of any length n as a cyclic convolution of length M ≥ 2n - 1 (M smooth):
  X_k = c_k · Σ_j (x_j c_j) · conj(c_{k-j}),  c_j = exp(σ·πi·j²/n).
"""
struct Bluestein{T,F,B} <: Kernel{T}
  n::Int
  chirp::Vector{Complex{T}}       # c_j, j = 0…n-1
  kernel::Vector{Complex{T}}      # DFT of the convolution kernel, scaled by 1/M
  fwd::F
  bwd::B
  buf::Vector{Complex{T}}
  work1::Vector{Complex{T}}
  work2::Vector{Complex{T}}
end

function next_smooth(n::Int)
  m = n
  while true
    f, rest = factorize(m)
    rest == 1 && all(p -> p <= 5, keys(f)) && return m
    m += 1
  end
end

function Bluestein{T}(n::Int, σ::Int) where {T}
  M = next_smooth(2n - 1)
  # j² mod 2n keeps the argument of the chirp small
  chirp = [Complex{T}(cispi(σ * mod(j * j, 2n) / n)) for j in 0:n-1]
  b = zeros(Complex{T}, M)
  b[1] = conj(chirp[1])
  for j in 1:n-1
    b[j+1] = b[M-j+1] = conj(chirp[j+1])
  end
  fwd = Chain{T}(M, -1)
  bwd = Chain{T}(M, 1)
  w1 = Vector{Complex{T}}(undef, M); w2 = Vector{Complex{T}}(undef, M)
  GC.@preserve b w1 w2 run!(cptr(b), cptr(b), cptr(w1), cptr(w2), fwd, 1)
  b ./= M
  Bluestein{T,typeof(fwd),typeof(bwd)}(n, chirp, b, fwd, bwd, zeros(Complex{T}, M), w1, w2)
end

# Lane by lane: gather, convolve, scatter. Element j of lane i at px + i + Sx·j, py + i + Sy·j.
# Uses the plan's own buffers, so one Bluestein kernel runs on one thread at a time.
function run_lanes!(py::Ptr{T}, px::Ptr{T}, ::Ptr{T}, ::Ptr{T}, P::Bluestein{T}, Sx::Int, Sy::Int, B::Int) where {T}
  n = P.n; buf = P.buf; M = length(buf)
  cx = Ptr{Complex{T}}(px); cy = Ptr{Complex{T}}(py)
  GC.@preserve P begin
    pb = cptr(buf); pw1 = cptr(P.work1); pw2 = cptr(P.work2)
    for i in 1:B
      @inbounds for j in 1:n
        buf[j] = unsafe_load(cx, i + Sx * (j - 1)) * P.chirp[j]
      end
      fill!(view(buf, n+1:M), zero(Complex{T}))
      run!(pb, pb, pw1, pw2, P.fwd, 1)
      buf .*= P.kernel
      run!(pb, pb, pw1, pw2, P.bwd, 1)
      @inbounds for k in 1:n
        unsafe_store!(cy, buf[k] * P.chirp[k], i + Sy * (k - 1))
      end
    end
  end
  nothing
end

run!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::Bluestein{T}, S::Int) where {T} =
  run_lanes!(py, px, pw1, pw2, P, S, S, S)

kernel(::Type{T}, n::Int, σ::Int, S::Int = 1) where {T} =
  smooth(n) ? Chain{T}(n, σ, radices(n, S)) : Bluestein{T}(n, σ)

"""
    run_lanes!(py, px, pw1, pw2, P::Chain, Sx, Sy, B)

Transform a block of B lanes; element j of lane i at `px + i + Sx·j` and `py + i + Sy·j`
(py may equal px). The work buffers hold B·n complex values each.
"""
run_lanes!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::Chain{T}, Sx::Int, Sy::Int, B::Int) where {T} =
  run_lanes!(py, px, pw1, pw2, P, Sx, Sy, B, NOPRUNE)

@generated function run_lanes!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::Chain{T,Rs,σ,K},
                               Sx::Int, Sy::Int, B::Int, Pr::Pruning{FM,LM}) where {T,Rs,σ,K,FM,LM}
  function body(V)
    ex = Any[:(L = P.n), :(s0 = 1)]
    src, srcS = :px, :Sx
    for i in 1:K
      r = Rs[i]
      last = i == K
      dst, dstS = last ? (:py, :Sy) : (isodd(i) ? :pw1 : :pw2, :B)
      if last
        if isempty(LM)
          push!(ex, :(stage_last_lanes!($dst, $src, s0, $srcS, $dstS, B, Val($r), Val($σ), $V)))
        else
          for (j, need) in enumerate(LM)
            push!(ex, :(stage_last_lanes!($dst, $src, s0, $srcS, $dstS, B, first(Pr.last[$j]), last(Pr.last[$j]) + 1,
                                          Val($r), Val($σ), $V, Val($need))))
          end
        end
      else
        push!(ex, :(m = L ÷ $r))
        if i > 1 || isempty(FM)
          push!(ex, :(stage_lanes!($dst, $src, cptr(P.tw[$i]), m, s0, $srcS, $dstS, B, Val($r), Val($σ), $V)))
        else
          for (j, zin) in enumerate(FM)
            push!(ex, :(stage_lanes!($dst, $src, cptr(P.tw[$i]), m, s0, $srcS, $dstS, B, first(Pr.first[$j]), last(Pr.first[$j]) + 1,
                                     Val($r), Val($σ), $V, Val($zin))))
          end
        end
        push!(ex, :(L = m), :(s0 *= $r))
      end
      src, srcS = dst, dstS
    end
    ex
  end
  K == 0 && return :(lane_copy!(py, px, Sy, Sx, B, 1); nothing)
  ladder = vec_ladder(T)
  # widest vector type whose complex count divides B
  function by_width(f)
    ex = f(ladder[end])
    for V in reverse(ladder[1:end-1])
      ex = :(B % $(ncomplex(V)) == 0 ? $(f(V)) : $ex)
    end
    ex
  end
  if K == 1
    # a single stage cannot run in place: go through pw1
    r = Rs[1]
    single(V) = quote
      stage_last_lanes!(pw1, px, 1, Sx, B, B, Val($r), Val($σ), $V)
      lane_copy!(py, pw1, Sy, B, B, P.n)
    end
    return quote
      GC.@preserve P $(by_width(single))
      nothing
    end
  end
  quote
    GC.@preserve P $(by_width(V -> quote $(body(V)...) end))
    nothing
  end
end

# y[i + Sy·j] = x[i + Sx·j] for B lanes and n elements
function lane_copy!(py::Ptr{T}, px::Ptr{T}, Sy::Int, Sx::Int, B::Int, n::Int) where {T}
  cy = Ptr{Complex{T}}(py); cx = Ptr{Complex{T}}(px)
  for j in 0:n-1
    unsafe_copyto!(cy + Sy * j * sizeof(Complex{T}), cx + Sx * j * sizeof(Complex{T}), B)
  end
end
