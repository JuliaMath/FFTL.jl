# User interface. Plans are AbstractFFTs plans (`p * x`, `mul!`, `inv`, `\`), but FFTL does not
# add methods to `AbstractFFTs.plan_fft` for standard arrays: that would take over `fft` from
# FFTW.jl whenever both are loaded. FFTL's functions are called qualified (`FFTL.fft`).

const FFTLFloat = Union{Float32,Float64}

"""
    FFTLPlan{T,N}

Plan for the complex DFT of `N`-dimensional arrays of size `sz` along the dimensions `region`,
with sign `σ` (-1 forward, +1 backward, unnormalised). The dimensions are transformed one after
the other, each by a kernel that handles all lanes of the dimension (see kernels.jl).
"""
mutable struct FFTLPlan{T<:FFTLFloat,N,D,Ks<:Tuple} <: AbstractFFTs.Plan{Complex{T}}
  sz::NTuple{N,Int}
  region::NTuple{D,Int}
  σ::Int
  kernels::Ks
  work1::Vector{Complex{T}}
  work2::Vector{Complex{T}}
  pinv::AbstractFFTs.Plan{Complex{T}}
  FFTLPlan{T,N,D,Ks}(sz, region, σ, kernels, w1, w2) where {T,N,D,Ks} = new{T,N,D,Ks}(sz, region, σ, kernels, w1, w2)
end

function FFTLPlan{T}(sz::NTuple{N,Int}, region, σ::Int) where {T<:FFTLFloat,N}
  reg = Tuple(sort(unique(collect(Int, region))))
  all(d -> 1 <= d <= N, reg) || throw(ArgumentError("region $region is not within 1:$N"))
  all(>=(1), sz) || throw(ArgumentError("array size must be positive, got $sz"))
  # dimensions with the same length and stride class share the kernel
  cache = Dict{Tuple{Int,Bool},Kernel{T}}()
  stride(d) = prod(sz[1:d-1]; init = 1)
  kernels = Tuple(get!(() -> kernel(T, sz[d], σ, stride(d)), cache, (sz[d], stride(d) > LARGE_STRIDE))
                  for d in reg)
  nwork = maximum((prod(sz[1:d]) for d in reg); init = 0)
  FFTLPlan{T,N,length(reg),typeof(kernels)}(sz, reg, σ, kernels,
                                            Vector{Complex{T}}(undef, nwork), Vector{Complex{T}}(undef, nwork))
end

Base.size(p::FFTLPlan) = p.sz
Base.ndims(::FFTLPlan{T,N}) where {T,N} = N
AbstractFFTs.fftdims(p::FFTLPlan) = p.region

function Base.show(io::IO, p::FFTLPlan{T}) where {T}
  dir = p.σ == -1 ? "forward" : "backward"
  print(io, "FFTL $dir plan for $(join(p.sz, "×")) Complex{$T}, dims $(p.region)")
  for (d, k) in zip(p.region, p.kernels)
    print(io, "\n  dim $d: ")
    k isa Chain && print(io, "radices ", radix_tuple(k))
    k isa Bluestein && print(io, "Bluestein, convolution length ", k.fwd.n)
  end
end
radix_tuple(::Chain{T,Rs}) where {T,Rs} = Rs

check_size(p, x) = size(x) == p.sz ||
  throw(DimensionMismatch("plan for size $(p.sz) applied to an array of size $(size(x))"))

# Dimensions one after the other; the tuple of kernels is peeled so that each call is static.
function apply_dims!(y::Array{Complex{T}}, x::Array{Complex{T}}, p::FFTLPlan{T}, ks::Tuple, ds::Tuple) where {T}
  isempty(ks) && return y
  k, d = first(ks), first(ds)
  # loops, not prod(p.sz[1:d-1]): a tuple sliced by a run-time range has an unknown length
  S = 1; outer = 1
  for i in 1:d-1
    S *= p.sz[i]
  end
  for i in d+1:length(p.sz)
    outer *= p.sz[i]
  end
  n = p.sz[d]
  GC.@preserve x y p begin
    px = cptr(x); py = cptr(y); pw1 = cptr(p.work1); pw2 = cptr(p.work2)
    step = 2 * S * n * sizeof(T)
    for o in 0:outer-1
      run!(py + o * step, px + o * step, pw1, pw2, k, S)
    end
  end
  apply_dims!(y, y, p, Base.tail(ks), Base.tail(ds))
end

function LinearAlgebra.mul!(y::Array{Complex{T},N}, p::FFTLPlan{T,N}, x::Array{Complex{T},N}) where {T,N}
  check_size(p, x); check_size(p, y)
  isempty(p.region) && return copyto!(y, x)
  apply_dims!(y, x, p, p.kernels, p.region)
end

function LinearAlgebra.mul!(y::AbstractArray{Complex{T},N}, p::FFTLPlan{T,N}, x::AbstractArray{Complex{T},N}) where {T,N}
  check_size(p, x); check_size(p, y)
  copyto!(y, mul!(Array{Complex{T},N}(undef, p.sz), p, Array{Complex{T},N}(x)))
end

Base.:*(p::FFTLPlan{T,N}, x::AbstractArray{<:Number,N}) where {T,N} =
  mul!(Array{Complex{T},N}(undef, p.sz), p, convert(Array{Complex{T},N}, x))

AbstractFFTs.plan_inv(p::FFTLPlan{T}) where {T} =
  AbstractFFTs.ScaledPlan(FFTLPlan{T}(p.sz, p.region, -p.σ), AbstractFFTs.normalization(T, p.sz, p.region))

"""
    plan_fft(x::AbstractArray, region = 1:ndims(x)) / plan_bfft / plan_ifft

Plans for the forward, unnormalised backward and normalised inverse DFT along the dimensions
`region` of arrays like `x` (Float32 or Float64 complex; other element types are converted to
ComplexF64).
"""
plan_fft(x::AbstractArray, region = 1:ndims(x)) = FFTLPlan{plan_type(x)}(size(x), region, -1)
plan_bfft(x::AbstractArray, region = 1:ndims(x)) = FFTLPlan{plan_type(x)}(size(x), region, 1)
plan_ifft(x::AbstractArray, region = 1:ndims(x)) =
  AbstractFFTs.ScaledPlan(plan_bfft(x, region), AbstractFFTs.normalization(plan_type(x), size(x), region))

plan_type(::AbstractArray{Complex{T}}) where {T<:FFTLFloat} = T
plan_type(::AbstractArray{T}) where {T<:FFTLFloat} = T
plan_type(::AbstractArray) = Float64

fft(x::AbstractArray, region = 1:ndims(x)) = plan_fft(x, region) * x
bfft(x::AbstractArray, region = 1:ndims(x)) = plan_bfft(x, region) * x
ifft(x::AbstractArray, region = 1:ndims(x)) = plan_ifft(x, region) * x
