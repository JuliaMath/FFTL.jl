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

full_vec(::Type{Float64}) = Vec{2,Float64}
full_vec(::Type{Float32}) = Vec{4,Float32}
half_vec(::Type{Float64}) = Vec{2,Float64}
half_vec(::Type{Float32}) = Vec{2,Float32}

# Stage calls of a chain from `src` through `dsts` (one destination per stage). `mode`:
#   :single  S = 1; the vector width follows s, the first Float32 stage pairs butterflies
#   :full    S is a multiple of the complex lanes of the full vector
#   :half    otherwise (Float32 with odd S): one complex value per vector
function chain_body(T, Rs, σ, src, dsts, mode)
  K = length(Rs)
  n = prod(Rs; init = 1)
  VF, VH = full_vec(T), half_vec(T)
  body = Any[:(L = P.n), :(s = S)]
  sst = 1
  for i in 1:K
    r = Rs[i]
    dst = dsts[i]
    V = mode == :full ? VF : mode == :half ? VH : (sst % ncomplex(VF) == 0 ? VF : VH)
    if i == K
      push!(body, :(stage_last!($dst, $src, s, Val($r), Val($σ), $V)))
    elseif mode == :single && i == 1 && T == Float32 && (n ÷ r) % 2 == 0
      push!(body, :(m = L ÷ $r), :(stage_first32!($dst, $src, cptr(P.tw1), m, Val($r), Val($σ))),
            :(L = m), :(s *= $r))
    else
      push!(body, :(m = L ÷ $r), :(stage!($dst, $src, cptr(P.tw[$i]), m, s, Val($r), Val($σ), $V)),
            :(L = m), :(s *= $r))
    end
    src = dst
    sst *= r
  end
  body
end

"""
    run!(py, px, pw1, pw2, P::Kernel, S)

Transform S interleaved sequences of length `P.n` from `px` into `py` (which may equal `px`),
using the work buffers `pw1`, `pw2` (S·n complex values each).
"""
@generated function run!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::Chain{T,Rs,σ,K}, S::Int) where {T,Rs,σ,K}
  K == 0 && return :(py == px || unsafe_copyto!(py, px, 2S); nothing)
  # Out of place: the stages alternate between py and pw1 and end in py; px is only read.
  oop = [iseven(K - i) ? :py : :pw1 for i in 1:K]
  # In place: the first stage reads px, the last writes px, the ones between alternate between
  # the two work buffers. A single stage goes through pw1.
  ip = Any[:px for _ in 1:K]
  for i in K-1:-1:1
    ip[i] = isodd(K - i) ? :pw1 : :pw2
  end
  body(dsts, mode) = quote
    $(chain_body(T, Rs, σ, :px, dsts, mode)...)
  end
  dispatch(dsts) = quote
    if S == 1
      $(body(dsts, :single))
    elseif S % $(ncomplex(full_vec(T))) == 0
      $(body(dsts, :full))
    else
      $(body(dsts, :half))
    end
  end
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

# Lane by lane: gather, convolve, scatter.
function run!(py::Ptr{T}, px::Ptr{T}, ::Ptr{T}, ::Ptr{T}, P::Bluestein{T}, S::Int) where {T}
  n = P.n; buf = P.buf; M = length(buf)
  cx = Ptr{Complex{T}}(px); cy = Ptr{Complex{T}}(py)
  GC.@preserve P begin
    pb = cptr(buf); pw1 = cptr(P.work1); pw2 = cptr(P.work2)
    for i in 1:S
      @inbounds for j in 1:n
        buf[j] = unsafe_load(cx, i + S * (j - 1)) * P.chirp[j]
      end
      fill!(view(buf, n+1:M), zero(Complex{T}))
      run!(pb, pb, pw1, pw2, P.fwd, 1)
      buf .*= P.kernel
      run!(pb, pb, pw1, pw2, P.bwd, 1)
      @inbounds for k in 1:n
        unsafe_store!(cy, buf[k] * P.chirp[k], i + S * (k - 1))
      end
    end
  end
  nothing
end

kernel(::Type{T}, n::Int, σ::Int, S::Int = 1) where {T} =
  smooth(n) ? Chain{T}(n, σ, radices(n, S)) : Bluestein{T}(n, σ)
