# Plans whose kernels are all run-time data: RChain for smooth lengths, Bluestein over RChain
# otherwise. Sizes, region and kernels live in vectors, so constructing a plan never builds a type
# from run-time values and executing it compiles nothing per size; this is the path for code
# compiled ahead of time (`juliac --trim`) and keeps the first plan of a new size cheap.
# Same strategies as FFTLPlan: four-step for one long line on several threads, dimensions 1 and
# 2 fused per plane when a plane fits in cache.

const RBluestein{T} = Bluestein{T,RChain{T},RChain{T}}
const RFourStep{T} = FourStep{T,RChain{T},RChain{T}}

"""
    RPlan{T}(sz, region, σ; threads = Threads.nthreads())

DFT plan with run-time kernels for complex arrays of size `sz` (any number of dimensions) along
the dimensions `region`, sign `σ` (-1 forward, +1 unnormalised backward). Applied like any FFTL
plan (`p * x`, `mul!(y, p, x)`, `inv(p)`); `execute!(p, py, px)` takes pointers.
"""
mutable struct RPlan{T<:FFTLFloat} <: AbstractFFTs.Plan{Complex{T}}
  sz::Vector{Int}
  region::Vector{Int}
  σ::Int
  chains::Vector{RChain{T}}                      # per dimension in region (smooth lengths)
  blue::Vector{Union{Nothing,RBluestein{T}}}     # per dimension: Bluestein, or nothing
  four::Union{Nothing,RFourStep{T}}              # one long contiguous line on several threads
  work1::Vector{Complex{T}}
  work2::Vector{Complex{T}}
  nthreads::Int
  tbuf::Vector{Vector{Complex{T}}}               # two work buffers per thread
  pinv::AbstractFFTs.Plan{Complex{T}}
  RPlan{T}(sz, region, σ, chains, blue, four, w1, w2, nt, tbuf) where {T} =
    new{T}(sz, region, σ, chains, blue, four, w1, w2, nt, tbuf)
end

function RPlan{T}(sz::AbstractVector{<:Integer}, region::AbstractVector{<:Integer}, σ::Integer;
                  threads::Integer = Threads.nthreads()) where {T<:FFTLFloat}
  dims = collect(Int, sz)
  N = length(dims)
  reg = normregion(region, N)
  nt = check_plan(dims, σ, threads)
  chains = RChain{T}[]
  blue = Union{Nothing,RBluestein{T}}[]
  # one long line (the other transformed dimensions have length 1): the four-step algorithm
  long = filter(d -> dims[d] > 1, reg)
  four = length(long) == 1 && fourstep_dim(T, dims, long[1], nt) ? FourStep{T}(dims[long[1]], Int(σ), nt, RChain) : nothing
  for d in reg
    n = dims[d]
    S, _ = dim_extent(dims, d)
    if smooth(n)
      push!(chains, RChain{T}(n, Int(σ), radices(n, S; esize = sizeof(Complex{T}))))
      push!(blue, nothing)
    else
      push!(chains, RChain{T}(1, Int(σ), Int[]))
      push!(blue, Bluestein{T}(n, Int(σ), RChain))
    end
  end
  nwork, ntwork = work_sizes(dims, reg, nt)
  RPlan{T}(dims, reg, Int(σ), chains, blue, four, Vector{Complex{T}}(undef, nwork), Vector{Complex{T}}(undef, nwork),
           nt, thread_buffers(T, ntwork, 2nt))
end

RPlan{T}(sz::Tuple, region, σ::Integer; kw...) where {T} = RPlan{T}(collect(Int, sz), collect(Int, region), σ; kw...)

"""
    execute!(p::RPlan{T}, py::Ptr{Complex{T}}, px::Ptr{Complex{T}})

Apply `p` to the array at `px` (column major, size `p.sz`) and write the result to `py`; `py`
may equal `px`. The caller keeps both arrays alive.
"""
function execute!(p::RPlan{T}, py::Ptr{Complex{T}}, px::Ptr{Complex{T}}) where {T}
  ry = Ptr{T}(py); rx = Ptr{T}(px)
  if isempty(p.region)
    py == px || unsafe_copyto!(py, px, prod(p.sz))
    return nothing
  end
  four = p.four
  if four !== nothing
    run_fourstep!(ry, rx, four, p.nthreads)
    return nothing
  end
  i0 = 1
  if length(p.region) >= 2 && p.region[1] == 1 && p.region[2] == 2 && p.blue[1] === nothing && p.blue[2] === nothing
    n1 = p.sz[1]; n2 = p.sz[2]
    outer = dim_extent(p.sz, 2)[2]
    if fuse_planes(n1, n2, outer, p.nthreads)
      run_planes!(ry, rx, p, p.chains[1], p.chains[2], n1, n2, outer)
      rx = ry
      i0 = 3
    end
  end
  for i in i0:length(p.region)
    d = p.region[i]
    S = 1; outer = 1
    for j in 1:d-1
      S *= p.sz[j]
    end
    for j in d+1:length(p.sz)
      outer *= p.sz[j]
    end
    src = i == 1 ? rx : ry
    b = p.blue[i]
    if b === nothing
      run_dim!(ry, src, p, p.chains[i], S, p.sz[d], outer)
    else
      run_dim!(ry, src, p, b, S, p.sz[d], outer)
    end
  end
  nothing
end

Base.size(p::RPlan) = Tuple(p.sz)
AbstractFFTs.fftdims(p::RPlan) = Tuple(p.region)
AbstractFFTs.AdjointStyle(::RPlan) = AbstractFFTs.FFTAdjointStyle()

Base.ndims(p::RPlan) = length(p.sz)
Base.show(io::IO, p::RPlan{T}) where {T} =
  print(io, "FFTL run-time ", p.σ < 0 ? "forward" : "backward", " plan for ", join(p.sz, "×"), " Complex{", T, "}, dims ", Tuple(p.region))

# Size check without building a vector per call (collect(size(x)) cost ~25 ns on small lines).
@inline has_size(a, sz::Vector{Int}) = ndims(a) == length(sz) && all(i -> size(a, i) == sz[i], eachindex(sz))

function LinearAlgebra.mul!(y::Array{Complex{T}}, p::RPlan{T}, x::Array{Complex{T}}) where {T}
  (has_size(x, p.sz) && has_size(y, p.sz)) ||
    throw(DimensionMismatch("plan size does not match the arrays"))
  GC.@preserve x y execute!(p, pointer(y), pointer(x))
  y
end

# other arrays (views, ...) through contiguous copies
LinearAlgebra.mul!(y::AbstractArray{Complex{T}}, p::RPlan{T}, x::AbstractArray{Complex{T}}) where {T} =
  copyto!(y, mul!(Array{Complex{T}}(undef, size(x)), p, Array{Complex{T}}(x)))

Base.:*(p::RPlan{T}, x::AbstractArray{<:Number}) where {T} =
  mul!(Array{Complex{T}}(undef, size(x)), p, convert(Array{Complex{T}}, x))

AbstractFFTs.plan_inv(p::RPlan{T}) where {T} =
  AbstractFFTs.ScaledPlan(RPlan{T}(p.sz, p.region, -p.σ; threads = p.nthreads),
                          AbstractFFTs.normalization(T, Tuple(p.sz), Tuple(p.region)))
