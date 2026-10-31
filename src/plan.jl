# Plans: a Stockham chain for lengths whose prime factors are ≤ MAX_RADIX_PRIME, Bluestein's
# algorithm on top of two chains otherwise.

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
  work1::Vector{Complex{T}}
  work2::Vector{Complex{T}}
end

function Chain{T}(n::Int, σ::Int, rs = radices(n)) where {T}
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
  Chain{T,Tuple(rs),σ,length(rs)}(n, Tuple(tw), tw1, Vector{Complex{T}}(undef, n), Vector{Complex{T}}(undef, n))
end

full_vec(::Type{Float64}) = Vec{2,Float64}
full_vec(::Type{Float32}) = Vec{4,Float32}
half_vec(::Type{Float64}) = Vec{2,Float64}
half_vec(::Type{Float32}) = Vec{2,Float32}

# Stage calls for a chain from `src` through `dsts` (one destination per stage).
function chain_body(T, Rs, σ, src, dsts)
  K = length(Rs)
  n = prod(Rs; init = 1)
  VF, VH = full_vec(T), half_vec(T)
  body = Any[:(L = P.n), :(s = 1)]
  sst = 1
  for i in 1:K
    r = Rs[i]
    dst = dsts[i]
    V = sst % ncomplex(VF) == 0 ? VF : VH
    if i == K
      push!(body, :(stage_last!($dst, $src, s, Val($r), Val($σ), $V)))
    elseif i == 1 && T == Float32 && (n ÷ r) % 2 == 0
      push!(body, :(m = L ÷ $r), :(stage_first32!($dst, $src, P.tw1, m, Val($r), Val($σ))),
            :(L = m), :(s *= $r))
    else
      push!(body, :(m = L ÷ $r), :(stage!($dst, $src, P.tw[$i], m, s, Val($r), Val($σ), $V)),
            :(L = m), :(s *= $r))
    end
    src = dst
    sst *= r
  end
  body
end

# y ≠ x: stages alternate between y and work1 so that the last one writes y; x is only read.
@generated function apply_oop!(y::Vector{Complex{T}}, x::Vector{Complex{T}}, P::Chain{T,Rs,σ,K}) where {T,Rs,σ,K}
  K == 0 && return :(copyto!(y, x))
  dsts = [iseven(K - i) ? :y : :(P.work1) for i in 1:K]
  quote
    $(chain_body(T, Rs, σ, :x, dsts)...)
    y
  end
end

# In place: the first stage reads x, the last writes x, the stages between alternate between
# the two work buffers.
@generated function apply_ip!(x::Vector{Complex{T}}, P::Chain{T,Rs,σ,K}) where {T,Rs,σ,K}
  K == 0 && return :x
  if K == 1
    return quote
      $(chain_body(T, Rs, σ, :x, [:(P.work1)])...)
      copyto!(x, P.work1)
    end
  end
  dsts = Any[:x for _ in 1:K]
  for i in K-1:-1:1
    dsts[i] = isodd(K - i) ? :(P.work1) : :(P.work2)
  end
  quote
    $(chain_body(T, Rs, σ, :x, dsts)...)
    x
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
  # exponent j² mod 2n keeps the argument small
  chirp = [Complex{T}(cispi(σ * mod(j * j, 2n) / n)) for j in 0:n-1]
  b = zeros(Complex{T}, M)
  b[1] = conj(chirp[1])
  for j in 1:n-1
    b[j+1] = b[M-j+1] = conj(chirp[j+1])
  end
  fwd = Chain{T}(M, -1)
  bwd = Chain{T}(M, 1)
  apply_ip!(b, fwd)
  b ./= M
  Bluestein{T,typeof(fwd),typeof(bwd)}(n, chirp, b, fwd, bwd, zeros(Complex{T}, M))
end

function apply_oop!(y::Vector{Complex{T}}, x::Vector{Complex{T}}, P::Bluestein{T}) where {T}
  n = P.n; buf = P.buf
  @inbounds for j in 1:n
    buf[j] = x[j] * P.chirp[j]
  end
  fill!(view(buf, n+1:length(buf)), zero(Complex{T}))
  apply_ip!(buf, P.fwd)
  buf .*= P.kernel
  apply_ip!(buf, P.bwd)
  @inbounds for k in 1:n
    y[k] = buf[k] * P.chirp[k]
  end
  y
end

apply_ip!(x::Vector{Complex{T}}, P::Bluestein{T}) where {T} = apply_oop!(x, x, P)

kernel(::Type{T}, n::Int, σ::Int) where {T} = smooth(n) ? Chain{T}(n, σ) : Bluestein{T}(n, σ)
