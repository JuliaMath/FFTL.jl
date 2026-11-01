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
  nthreads::Int
  tbuf::Vector{Vector{Complex{T}}}     # two work buffers per thread
  pinv::AbstractFFTs.Plan{Complex{T}}
  FFTLPlan{T,N,D,Ks}(sz, region, σ, kernels, w1, w2, nt, tbuf) where {T,N,D,Ks} =
    new{T,N,D,Ks}(sz, region, σ, kernels, w1, w2, nt, tbuf)
end

function FFTLPlan{T}(sz::NTuple{N,Int}, region, σ::Int; threads::Integer = Threads.nthreads()) where {T<:FFTLFloat,N}
  reg = Tuple(sort(unique(collect(Int, region))))
  all(d -> 1 <= d <= N, reg) || throw(ArgumentError("region $region is not within 1:$N"))
  all(>=(1), sz) || throw(ArgumentError("array size must be positive, got $sz"))
  nt = max(1, Int(threads))
  # dimensions with the same length and stride class share the kernel; a single long contiguous
  # transform on several threads uses the four-step algorithm
  cache = Dict{Tuple{Int,Bool},Kernel{T}}()
  function dim_kernel(d)
    S, outer = dim_extent(sz, d)
    S == 1 && outer == 1 && use_fourstep(sz[d], nt) && return FourStep{T}(sz[d], σ, nt)
    get!(() -> kernel(T, sz[d], σ, S), cache, (sz[d], S > LARGE_STRIDE))
  end
  kernels = Tuple(dim_kernel(d) for d in reg)
  nwork = maximum((prod(sz[1:d]) for d in reg); init = 0)
  ntwork = maximum((thread_work(sz, d, nt) for d in reg); init = 0)
  if length(reg) >= 2 && reg[1] == 1 && reg[2] == 2 && fuse_planes(sz[1], sz[2], dim_extent(sz, 2)[2], nt)
    ntwork = max(ntwork, sz[1] * sz[2])
  end
  tbuf = [Vector{Complex{T}}(undef, ntwork) for _ in 1:2nt]
  FFTLPlan{T,N,length(reg),typeof(kernels)}(sz, reg, σ, kernels, Vector{Complex{T}}(undef, nwork),
                                            Vector{Complex{T}}(undef, nwork), nt, tbuf)
end

Base.size(p::FFTLPlan) = p.sz
Base.ndims(::FFTLPlan{T,N}) where {T,N} = N
AbstractFFTs.fftdims(p::FFTLPlan) = p.region
AbstractFFTs.AdjointStyle(::FFTLPlan) = AbstractFFTs.FFTAdjointStyle()

function Base.show(io::IO, p::FFTLPlan{T}) where {T}
  dir = p.σ == -1 ? "forward" : "backward"
  print(io, "FFTL $dir plan for $(join(p.sz, "×")) Complex{$T}, dims $(p.region)")
  for (d, k) in zip(p.region, p.kernels)
    print(io, "\n  dim $d: ")
    k isa Chain && print(io, "radices ", radix_tuple(k))
    k isa Bluestein && print(io, "Bluestein, convolution length ", k.fwd.n)
    k isa FourStep && print(io, "four-step $(k.n1) × $(k.n2), radices ", radix_tuple(k.k1), " and ", radix_tuple(k.k2))
  end
end
radix_tuple(::Chain{T,Rs}) where {T,Rs} = Rs

check_size(p, x) = size(x) == p.sz ||
  throw(DimensionMismatch("plan for size $(p.sz) applied to an array of size $(size(x))"))

# Dimensions one after the other; the tuple of kernels is peeled so that each call is static.
# Dimensions 1 and 2 are fused per plane when a plane fits in cache (`run_planes!`).
function apply_dims!(y::Array{Complex{T}}, x::Array{Complex{T}}, p::FFTLPlan{T}, ks::Tuple, ds::Tuple) where {T}
  isempty(ks) && return y
  if length(ks) >= 2 && ds[1] == 1 && ds[2] == 2 && ks[1] isa Chain && ks[2] isa Chain
    n1, n2 = p.sz[1], p.sz[2]
    outer = dim_extent(p.sz, 2)[2]
    if fuse_planes(n1, n2, outer, p.nthreads)
      GC.@preserve x y p run_planes!(cptr(y), cptr(x), p, ks[1], ks[2], n1, n2, outer)
      return apply_dims!(y, y, p, Base.tail(Base.tail(ks)), Base.tail(Base.tail(ds)))
    end
  end
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
  GC.@preserve x y p run_dim!(cptr(y), cptr(x), p, k, S, n, outer)
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
  AbstractFFTs.ScaledPlan(FFTLPlan{T}(p.sz, p.region, -p.σ; threads = p.nthreads),
                          AbstractFFTs.normalization(T, p.sz, p.region))

"""
    plan_fft(x::AbstractArray, region = 1:ndims(x); threads = Threads.nthreads()) / plan_bfft / plan_ifft

Plans for the forward, unnormalised backward and normalised inverse DFT along the dimensions
`region` of arrays like `x` (Float32 or Float64 complex; other element types are converted to
ComplexF64). Transforms large enough to profit run on up to `threads` threads.
"""
plan_fft(x::AbstractArray, region = 1:ndims(x); kw...) = FFTLPlan{plan_type(x)}(size(x), region, -1; kw...)
plan_bfft(x::AbstractArray, region = 1:ndims(x); kw...) = FFTLPlan{plan_type(x)}(size(x), region, 1; kw...)
plan_ifft(x::AbstractArray, region = 1:ndims(x); kw...) =
  AbstractFFTs.ScaledPlan(plan_bfft(x, region; kw...), AbstractFFTs.normalization(plan_type(x), size(x), region))

plan_type(::AbstractArray{Complex{T}}) where {T<:FFTLFloat} = T
plan_type(::AbstractArray{T}) where {T<:FFTLFloat} = T
plan_type(::AbstractArray) = Float64

fft(x::AbstractArray, region = 1:ndims(x)) = plan_fft(x, region) * x
bfft(x::AbstractArray, region = 1:ndims(x)) = plan_bfft(x, region) * x
ifft(x::AbstractArray, region = 1:ndims(x)) = plan_ifft(x, region) * x
