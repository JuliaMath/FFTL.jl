# Run-time plans of the real transforms (r2c, c2r, r2r) for arrays of any number of dimensions:
# sizes, region and kernels are data (RChain, Bluestein over RChain), only the element type is
# a type parameter. They run the line kernels of real.jl and r2r.jl and are the real-data
# counterpart of RPlan for code compiled ahead of time (libfftl).

"""
    RRFFTPlan{T}(sz, region, σ; threads = Threads.nthreads())

r2c (σ = -1) or unnormalised c2r (σ = +1) plan for real arrays of size `sz` along `region`:
real along the first dimension of the region (half spectrum of n ÷ 2 + 1 values there), complex
along the others. Applied with `execute_r2c!(p, py, px)` or `execute_c2r!(p, py, px)` (pointers;
out of place, c2r leaves its input unchanged). In place (`py` and `px` the same address) the
real array is padded to 2(n ÷ 2 + 1) values along that dimension, so that it fills the memory of
the half spectrum (FFTW's in-place layout); the padding is ignored by r2c and undefined after
c2r.
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
  rbuf::Vector{T}                     # in place with S > 1: the unpadded real array
end

function RRFFTPlan{T}(sz::AbstractVector{<:Integer}, region::AbstractVector{<:Integer}, σ::Integer;
                      threads::Integer = Threads.nthreads()) where {T<:FFTLFloat}
  dims = collect(Int, sz)
  N = length(dims)
  reg = normregion(region, N)
  isempty(reg) && throw(ArgumentError("a real transform needs at least one dimension"))
  all(>=(1), dims) || throw(ArgumentError("array size must be positive"))
  (σ == 1 || σ == -1) || throw(ArgumentError("sign must be 1 or -1"))
  nt = max(1, Int(threads))
  d1 = reg[1]; n = dims[d1]
  line = RealLine{T}(n, Int(σ), RChain)
  csz = copy(dims); csz[d1] = nhalf(n)
  cplan = length(reg) > 1 ? RPlan{T}(csz, reg[2:end], Int(σ); threads = nt) : nothing
  S, offs_r, offs_c = line_layout(dims, d1)
  copybuf = Vector{Complex{T}}(undef, σ == 1 && cplan !== nothing ? prod(csz) : 0)
  RRFFTPlan{T}(dims, reg, Int(σ), line, cplan, S, offs_r, offs_c, nt, thread_buffers(T, n, 5nt), copybuf, T[])
end

Base.show(io::IO, p::RRFFTPlan{T}) where {T} =
  print(io, "FFTL run-time ", p.σ < 0 ? "r2c" : "c2r", " plan for ", join(p.sz, "×"), " ", T, ", dims ", Tuple(p.region))

# r2c: px real array of size p.sz, py half spectrum
function execute_r2c!(p::RRFFTPlan{T}, py::Ptr{Complex{T}}, px::Ptr{T}) where {T}
  if UInt(py) == UInt(px)
    r2c_inplace!(p, py)
  else
    p.S == 1 ? r2c_contiguous!(py, px, p) : r2c_strided!(py, px, p)
  end
  cp = p.cplan
  cp === nothing || execute!(cp, py, py)
  nothing
end

# c2r: px half spectrum, py real array of size p.sz
function execute_c2r!(p::RRFFTPlan{T}, py::Ptr{T}, px::Ptr{Complex{T}}) where {T}
  UInt(py) == UInt(px) && return c2r_inplace!(p, px)
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

# Padded real layout of the in-place transforms: element j < n of line (i, o) at
# i + S (j + 2h o), h = n ÷ 2 + 1. With S = 1 a line and its half spectrum share their memory and
# are transformed where they are (each line kernel reads its input before writing); with S > 1
# the lines interleave, so the real array goes through the unpadded copy rbuf.
function padded_copy!(dst::Ptr{T}, src::Ptr{T}, p::RRFFTPlan{T}, topad::Bool) where {T}
  S = p.S; n = p.line.n; h2 = 2nhalf(n)
  outer = length(p.offs_r) ÷ S
  for o in 0:outer-1, j in 0:n-1, i in 0:S-1
    a = i + S * (j + n * o); b = i + S * (j + h2 * o)
    topad ? unsafe_store!(dst, unsafe_load(src, a + 1), b + 1) : unsafe_store!(dst, unsafe_load(src, b + 1), a + 1)
  end
  nothing
end

function r2c_inplace!(p::RRFFTPlan{T}, py::Ptr{Complex{T}}) where {T}
  if p.S == 1
    each_line(length(p.offs_c), p.line.n, line_tasks(p.line, p.nthreads)) do c, i
      local b = p.bufs
      local q = py + p.offs_c[i+1] * sizeof(Complex{T})
      r2c_line!(q, Ptr{T}(q), cptr(b[5c-4]), cptr(b[5c-3]), cptr(b[5c-2]), p.line)
    end
  else
    length(p.rbuf) == prod(p.sz) || resize!(p.rbuf, prod(p.sz))
    GC.@preserve p begin
      padded_copy!(pointer(p.rbuf), Ptr{T}(py), p, false)
      r2c_strided!(py, pointer(p.rbuf), p)
    end
  end
  nothing
end

function c2r_inplace!(p::RRFFTPlan{T}, px::Ptr{Complex{T}}) where {T}
  cp = p.cplan
  cp === nothing || execute!(cp, px, px)
  if p.S == 1
    each_line(length(p.offs_c), p.line.n, line_tasks(p.line, p.nthreads)) do c, i
      local b = p.bufs
      local q = px + p.offs_c[i+1] * sizeof(Complex{T})
      c2r_line!(Ptr{T}(q), q, cptr(b[5c-4]), cptr(b[5c-3]), cptr(b[5c-2]), p.line)
    end
  else
    length(p.rbuf) == prod(p.sz) || resize!(p.rbuf, prod(p.sz))
    GC.@preserve p begin
      c2r_strided!(pointer(p.rbuf), px, p)
      padded_copy!(Ptr{T}(px), pointer(p.rbuf), p, true)
    end
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
  check_r2r_region(reg, N)
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

Base.show(io::IO, p::RR2RPlan{T}) where {T} =
  print(io, "FFTL run-time r2r plan for ", join(p.sz, "×"), " ", T, ", ", join(("dim $d: $k" for (d, k) in zip(p.region, p.kinds)), ", "))

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
