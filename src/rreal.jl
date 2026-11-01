# Run-time plans of the real transforms (r2c, c2r, r2r) for arrays of any number of dimensions:
# sizes, region and kernels are data (RChain, Bluestein over RChain), only the element type is
# a type parameter. They run the line kernels of real.jl and r2r.jl and are the real-data
# counterpart of RPlan for code compiled ahead of time (libfftl).

"""
    RRFFTPlan{T}(sz, region, σ; threads = Threads.nthreads())

r2c (σ = -1) or unnormalised c2r (σ = +1) plan for real arrays of size `sz` along `region`:
real along the first dimension of the region (half spectrum of n ÷ 2 + 1 values there), complex
along the others. Applied with `execute_r2c!(p, py, px)` or `execute_c2r!(p, py, px)` (pointers,
out of place; c2r leaves its input unchanged).
"""
mutable struct RRFFTPlan{T<:FFTLFloat}
  sz::Vector{Int}                     # size of the real array
  region::Vector{Int}
  σ::Int
  line::RealLine{T,RKern{T}}
  cplan::Union{Nothing,RPlan{T}}      # complex passes on the half spectrum
  S::Int
  offs_r::Vector{Int}
  offs_c::Vector{Int}
  nthreads::Int
  bufs::Vector{Vector{Complex{T}}}    # five per thread
  copy::Vector{Complex{T}}            # c2r: the input after the complex passes
end

function RRFFTPlan{T}(sz::AbstractVector{<:Integer}, region::AbstractVector{<:Integer}, σ::Integer;
                      threads::Integer = Threads.nthreads()) where {T<:FFTLFloat}
  dims = collect(Int, sz)
  reg = sort!(unique(collect(Int, region)))
  N = length(dims)
  isempty(reg) && throw(ArgumentError("a real transform needs at least one dimension"))
  all(d -> 1 <= d <= N, reg) || throw(ArgumentError("region is not within the array dimensions"))
  all(>=(1), dims) || throw(ArgumentError("array size must be positive"))
  (σ == 1 || σ == -1) || throw(ArgumentError("sign must be 1 or -1"))
  nt = max(1, Int(threads))
  d1 = reg[1]; n = dims[d1]
  line = RealLine{T}(n, Int(σ), RChain)
  csz = copy(dims); csz[d1] = nhalf(n)
  cplan = length(reg) > 1 ? RPlan{T}(csz, reg[2:end], Int(σ); threads = nt) : nothing
  S, offs_r, offs_c = line_layout(dims, d1)
  copybuf = Vector{Complex{T}}(undef, σ == 1 && cplan !== nothing ? prod(csz) : 0)
  RRFFTPlan{T}(dims, reg, Int(σ), line, cplan, S, offs_r, offs_c, nt, thread_buffers(T, n, 5nt), copybuf)
end

# r2c: px real array of size p.sz, py half spectrum
function execute_r2c!(p::RRFFTPlan{T}, py::Ptr{Complex{T}}, px::Ptr{T}) where {T}
  p.S == 1 ? r2c_contiguous!(py, px, p) : r2c_strided!(py, px, p)
  cp = p.cplan
  cp === nothing || execute!(cp, py, py)
  nothing
end

# c2r: px half spectrum, py real array of size p.sz
function execute_c2r!(p::RRFFTPlan{T}, py::Ptr{T}, px::Ptr{Complex{T}}) where {T}
  cp = p.cplan
  GC.@preserve p begin
    src = px
    if cp !== nothing
      src = pointer(p.copy)
      execute!(cp, src, px)
    end
    p.S == 1 ? c2r_contiguous!(py, src, p) : c2r_strided!(py, src, p)
  end
  nothing
end

"""
    RR2RPlan{T}(sz, kinds, region; threads = Threads.nthreads())

Real-to-real plan with run-time kernels for arrays of size `sz`, one `R2RKind` per dimension
of `region` (unnormalised, as `plan_r2r`). Applied with `execute_r2r!(p, py, px)`; `py` may
equal `px`.
"""
mutable struct RR2RPlan{T<:FFTLFloat}
  sz::Vector{Int}
  region::Vector{Int}
  kinds::Vector{R2RKind}
  lines::Vector{R2RLine{T,RInner{T}}}
  nthreads::Int
  bufs::Vector{R2RBuffers{T}}
end

function RR2RPlan{T}(sz::AbstractVector{<:Integer}, kinds::AbstractVector{R2RKind}, region::AbstractVector{<:Integer};
                     threads::Integer = Threads.nthreads()) where {T<:FFTLFloat}
  dims = collect(Int, sz)
  reg = collect(Int, region)
  N = length(dims)
  allunique(reg) && all(d -> 1 <= d <= N, reg) || throw(ArgumentError("region is not within the array dimensions"))
  all(>=(1), dims) || throw(ArgumentError("array size must be positive"))
  length(kinds) == length(reg) || throw(ArgumentError("one kind per transformed dimension"))
  lines = Vector{R2RLine{T,RInner{T}}}(undef, length(reg))
  nmax = 1
  for i in eachindex(reg)
    lines[i] = R2RLine{T}(kinds[i], dims[reg[i]], RChain)
    nmax = max(nmax, dims[reg[i]])
  end
  nt = max(1, Int(threads))
  bufs = Vector{R2RBuffers{T}}(undef, nt)
  for c in 1:nt
    bufs[c] = R2RBuffers{T}(nmax)
  end
  RR2RPlan{T}(dims, reg, collect(R2RKind, kinds), lines, nt, bufs)
end

function execute_r2r!(p::RR2RPlan{T}, py::Ptr{T}, px::Ptr{T}) where {T}
  if isempty(p.region)
    py == px || unsafe_copyto!(py, px, prod(p.sz))
    return nothing
  end
  src = px
  for i in eachindex(p.region)
    r2r_dim!(py, src, p.region[i], p.lines[i], p)
    src = py
  end
  nothing
end
