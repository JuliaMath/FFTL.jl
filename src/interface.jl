# User interface. Plans are AbstractFFTs plans (`p * x`, `mul!`, `inv`, `\`), but FFTL does not
# add methods to `AbstractFFTs.plan_fft` for standard arrays: that would take over `fft` from
# FFTW.jl whenever both are loaded. FFTL's functions are called qualified (`FFTL.fft`).

const FFTLFloat = Union{Float32,Float64}

"""
    FFTLPlan{T,K}

Plan for a one-dimensional complex DFT of length `n` with sign `σ` (-1 forward, +1 backward,
unnormalised).
"""
mutable struct FFTLPlan{T<:FFTLFloat,K<:Kernel{T}} <: AbstractFFTs.Plan{Complex{T}}
  n::Int
  σ::Int
  kernel::K
  pinv::AbstractFFTs.Plan{Complex{T}}
  FFTLPlan{T,K}(n, σ, kernel) where {T,K} = new{T,K}(n, σ, kernel)
end

function FFTLPlan{T}(n::Int, σ::Int) where {T<:FFTLFloat}
  n >= 1 || throw(ArgumentError("length must be positive, got $n"))
  k = kernel(T, n, σ)
  FFTLPlan{T,typeof(k)}(n, σ, k)
end

Base.size(p::FFTLPlan) = (p.n,)
Base.ndims(::FFTLPlan) = 1
AbstractFFTs.fftdims(::FFTLPlan) = 1:1

function Base.show(io::IO, p::FFTLPlan{T}) where {T}
  dir = p.σ == -1 ? "forward" : "backward"
  print(io, "FFTL $dir plan for $(p.n) × Complex{$T}")
  k = p.kernel
  k isa Chain && print(io, ", radices ", radix_tuple(k))
  k isa Bluestein && print(io, ", Bluestein with length ", k.fwd.n)
end
radix_tuple(::Chain{T,Rs}) where {T,Rs} = Rs

check_size(p, x) = length(x) == p.n ||
  throw(DimensionMismatch("plan for length $(p.n) applied to an array of length $(length(x))"))

function LinearAlgebra.mul!(y::Vector{Complex{T}}, p::FFTLPlan{T}, x::Vector{Complex{T}}) where {T}
  check_size(p, x); check_size(p, y)
  y === x ? apply_ip!(y, p.kernel) : apply_oop!(y, x, p.kernel)
  y
end

function LinearAlgebra.mul!(y::AbstractVector{Complex{T}}, p::FFTLPlan{T}, x::AbstractVector{Complex{T}}) where {T}
  check_size(p, x); check_size(p, y)
  copyto!(y, mul!(similar(y, Complex{T}, p.n), p, collect(Complex{T}, x)))
end

Base.:*(p::FFTLPlan{T}, x::AbstractVector{<:Number}) where {T} =
  mul!(Vector{Complex{T}}(undef, p.n), p, convert(Vector{Complex{T}}, x))

AbstractFFTs.plan_inv(p::FFTLPlan{T}) where {T} =
  AbstractFFTs.ScaledPlan(FFTLPlan{T}(p.n, -p.σ), AbstractFFTs.normalization(T, (p.n,), 1:1))

"""
    plan_fft(x::AbstractVector{<:Complex}) / plan_bfft / plan_ifft

Plans for the forward, unnormalised backward and normalised inverse DFT of vectors like `x`
(Float32 or Float64 complex). Other element types are converted to ComplexF64.
"""
plan_fft(x::AbstractVector) = FFTLPlan{plan_type(x)}(length(x), -1)
plan_bfft(x::AbstractVector) = FFTLPlan{plan_type(x)}(length(x), 1)
plan_ifft(x::AbstractVector) = AbstractFFTs.ScaledPlan(plan_bfft(x), AbstractFFTs.normalization(plan_type(x), (length(x),), 1:1))

plan_type(x::AbstractVector{Complex{T}}) where {T<:FFTLFloat} = T
plan_type(x::AbstractVector{T}) where {T<:FFTLFloat} = T
plan_type(x::AbstractVector) = Float64

fft(x::AbstractVector) = plan_fft(x) * x
bfft(x::AbstractVector) = plan_bfft(x) * x
ifft(x::AbstractVector) = plan_ifft(x) * x
