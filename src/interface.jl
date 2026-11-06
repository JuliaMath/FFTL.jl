# User interface. Plans are AbstractFFTs plans (`p * x`, `mul!`, `inv`, `\`), but FFTL does not
# add methods to `AbstractFFTs.plan_fft` for standard arrays: that would take over `fft` from
# FFTW.jl whenever both are loaded. FFTL's functions are called qualified (`FFTL.fft`).

const FFTLFloat = Union{Float32,Float64}

# The transformed dimensions: sorted, without duplicates, each within 1:N (an integer is one
# dimension). The message is a literal: the run-time plans are compiled ahead of time.
function normregion(region, N::Int)
  reg = sort!(unique!(collect(Int, region isa Integer ? (region,) : region)))
  all(d -> 1 <= d <= N, reg) || throw(ArgumentError("region is not within the array dimensions"))
  reg
end

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
  reg = Tuple(normregion(region, N))
  nt = check_plan(sz, σ, threads)
  # dimensions with the same length and stride class share the kernel; a single long contiguous
  # transform on several threads uses the four-step algorithm
  cache = Dict{Tuple{Int,Int},Kernel{T}}()
  function dim_kernel(d)
    fourstep_dim(sz, d, nt) && return FourStep{T}(sz[d], σ, nt)
    bluestein_mt_dim(sz, d, nt) && return BluesteinMT{T}(sz[d], σ, nt)
    S, _ = dim_extent(sz, d)
    get!(() -> kernel(T, sz[d], σ, S), cache, (sz[d], S > LARGE_STRIDE ? S : 0))
  end
  kernels = Tuple(dim_kernel(d) for d in reg)
  nwork, ntwork = work_sizes(sz, reg, nt)
  tbuf = thread_buffers(T, ntwork, 2nt)
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
end

function Base.show(io::IO, ::MIME"text/plain", p::FFTLPlan{T}) where {T}
  show(io, p)
  for (d, k) in zip(p.region, p.kernels)
    print(io, "\n  dim $d: ")
    k isa Chain && print(io, "radices ", radix_tuple(k))
    k isa Bluestein && print(io, "Bluestein, convolution length ", k.fwd.n)
    k isa BluesteinMT && print(io, "Bluestein on threads, convolution length ", k.fwd.n)
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
  S, outer = dim_extent(p.sz, d)
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
`region` of arrays like `x`. Float32 and Float64 (real or complex) use the SIMD kernels, and
transforms large enough to profit run on up to `threads` threads. Other floating-point types
(BigFloat, Double64, Float128, …) are transformed in their own precision by a generic,
single-threaded path; Float16, integers and other numbers are converted to ComplexF64.

The methods are on AbstractFFTs's functions (Variant A): with FFTW.jl loaded, FFTW's methods
win for StridedArrays and FFTL catches the rest of its float range; FFTL stays reachable
qualified (`FFTL.plan_fft`).
"""
plan_fft(x::AbstractArray{<:Complex{<:FFTLFloat}}, region = 1:ndims(x); kws...) =
  FFTLPlan{plan_type(x)}(size(x), region, -1; kws...)
plan_bfft(x::AbstractArray{<:Complex{<:FFTLFloat}}, region = 1:ndims(x); kws...) =
  FFTLPlan{plan_type(x)}(size(x), region, 1; kws...)
# the normalisation over the plan's dimensions (the region without duplicates)
function plan_ifft(x::AbstractArray{<:Complex{<:FFTLFloat}}, region = 1:ndims(x); kws...)
  p = plan_bfft(x, region; kws...)
  AbstractFFTs.ScaledPlan(p, AbstractFFTs.normalization(plan_type(x), size(x), AbstractFFTs.fftdims(p)))
end
# The generic-float path: BigFloat, Double64, Float128, … in their own precision,
# single-threaded. These methods keep FFTL.plan_fft working for the wider float types; they
# are not part of the FFTW drop-in surface (FFTW does not handle these types).
plan_fft(x::AbstractArray{<:Complex{<:AbstractFloat}}, region = 1:ndims(x); kws...) =
  GPlan{elreal(x)}(size(x), region, -1)
plan_bfft(x::AbstractArray{<:Complex{<:AbstractFloat}}, region = 1:ndims(x); kws...) =
  GPlan{elreal(x)}(size(x), region, 1)
function plan_ifft(x::AbstractArray{<:Complex{<:AbstractFloat}}, region = 1:ndims(x); kws...)
  p = plan_bfft(x, region; kws...)
  AbstractFFTs.ScaledPlan(p, AbstractFFTs.normalization(elreal(x), size(x), AbstractFFTs.fftdims(p)))
end

# In-place: the same plans transform in place — mul! with x === y is safe (the kernels are
# line-atomic: every pass loads its line before storing it). The (x, region) forms override
# AbstractFFTs's generic wrappers, which compute out of place (their `*` does not mutate).
plan_fft!(x::AbstractArray{<:Complex{<:FFTLFloat}}, region = 1:ndims(x); kws...) = plan_fft(x, region; kws...)
plan_bfft!(x::AbstractArray{<:Complex{<:FFTLFloat}}, region = 1:ndims(x); kws...) = plan_bfft(x, region; kws...)
plan_ifft!(x::AbstractArray{<:Complex{<:FFTLFloat}}, region = 1:ndims(x); kws...) = plan_ifft(x, region; kws...)

fft!(x::AbstractArray{<:Complex{<:FFTLFloat}}, p::FFTLPlan) = mul!(x, p, x)
bfft!(x::AbstractArray{<:Complex{<:FFTLFloat}}, p::FFTLPlan) = mul!(x, p, x)
ifft!(x::AbstractArray{<:Complex{<:FFTLFloat}}, p::AbstractFFTs.ScaledPlan{<:Any,<:FFTLPlan}) = mul!(x, p, x)
fft!(x::AbstractArray{<:Complex{<:FFTLFloat}}, region = 1:ndims(x)) = fft!(x, plan_fft(x, region))
bfft!(x::AbstractArray{<:Complex{<:FFTLFloat}}, region = 1:ndims(x)) = bfft!(x, plan_bfft(x, region))
ifft!(x::AbstractArray{<:Complex{<:FFTLFloat}}, region = 1:ndims(x)) = ifft!(x, plan_ifft(x, region))

elreal(x::AbstractArray) = real(eltype(x))

plan_type(::AbstractArray{Complex{T}}) where {T<:FFTLFloat} = T
plan_type(::AbstractArray{T}) where {T<:FFTLFloat} = T
plan_type(::AbstractArray) = Float64

# The transform functions fft/bfft/ifft/rfft/brfft/irfft are AbstractFFTs's: its generic
# definitions call plan_fft/plan_bfft/plan_ifft/plan_rfft (the methods above) and re-exported
# from AbstractFFTs in FFTL.jl, so `using FFTL` provides them without own definitions.
