# Batched 1D transforms along one dimension of a strided array: the operation an NFFT needs on
# the oversampled grid (boxes of a padded array, chosen and threaded by the caller).

# Kernels of run-time pass plans (`plan_pass_runtime`). Their type depends on the element type only:
# a smooth length runs on RChain, a leftover of large prime factors is a composite of an RChain and a
# resampled transform over RChain, and a single long line on several threads runs the four-step
# algorithm (RChain kernels). A library can therefore hold plans of any size in one container and
# compile them ahead of time (juliac --trim).
const RComposite{T} = Union{Composite{T,RChain{T},RsBatch{T,RChain{T},RTable{T}}},
                            Composite{T,RChain{T},RsBatch{T,RChain{T},RPoly{T}}}}
const RPassKern{T} = Union{RChain{T},RResample{T},RComposite{T},RFourStep{T}}

function runtime_kernel(::Type{T}, n::Int, σ::Int, S::Int) where {T}
  smooth(n) && return RChain{T}(n, σ, radices(n, S; esize = sizeof(Complex{T})))
  s, q = composite_split(T, n)
  s == 1 && return Resample{T}(n, σ, RChain)
  chain = RChain{T}(s, σ, radices(s, S * q; esize = sizeof(Complex{T})))
  rs = RsBatch{T}(q, σ, s)
  tw = cached(() -> Complex{T}.(unit_roots(σ, n)), CacheKey(:composite, T, n, σ, Int[]))::Vector{Complex{T}}
  Composite{T,typeof(chain),typeof(rs)}(n, s, q, chain, rs, tw, Vector{Complex{T}}(undef, S * n))
end

"""
    PassPlan{T}

Plan for the DFT along dimension `d` of all lines of arrays with a given size and strides
(dimension 1 contiguous). Along d = 1 each line is a contiguous chain; along d > 1 the lines are
the lanes: dimension 1 (and the following dimensions as long as they are contiguous with it) is
the vectorised lane index, the element stride is the stride of dimension d. `offs` lists the
offsets of the line groups (complex values from the first element).
"""
struct PassPlan{T<:FFTLFloat,N,K<:Kernel{T}}
  sz::NTuple{N,Int}
  st::NTuple{N,Int}
  d::Int
  σ::Int
  kernel::K
  lanes::Int                       # lanes per group (1 along d = 1)
  Sd::Int                          # element stride along d
  offs::Vector{Int}
  B::Int                           # lane block
  nthreads::Int
  bufs::Vector{Vector{Complex{T}}} # two per thread
end

# Memory offsets (in elements) of the lines of an array with size sz and strides st, one per
# index of the dimensions `outer` (first index fastest). Plain loops over vectors: the code does not
# depend on the number of dimensions at compile time (juliac --trim).
function line_offsets(sz::NTuple{N,Int}, st::NTuple{N,Int}, outer::Vector{Int}) where {N}
  offs = Int[0]
  for j in eachindex(outer)
    stride = st[outer[j]]; c = sz[outer[j]]; L = length(offs)
    next = Vector{Int}(undef, L * c)
    for i in 0:c-1, k in 1:L
      next[i * L + k] = offs[k] + i * stride
    end
    offs = next
  end
  offs
end

# Layout of the lines of a pass along dimension d (dimension 1 contiguous): lanes per group, the
# element stride along d, and the offsets of the groups.
function pass_layout(sz::NTuple{N,Int}, st::NTuple{N,Int}, d::Int) where {N}
  if d == 1
    lanes = 1; Sd = 1
    outer = collect(2:N)
  else
    # merge dimensions 1, 2, … into the lane index while they are contiguous and before d
    m = 1; L = sz[1]
    while m + 1 < d && st[m+1] == L
      m += 1; L *= sz[m]
    end
    lanes = L; Sd = st[d]
    outer = Int[i for i in m+1:N if i != d]
  end
  lanes, Sd, line_offsets(sz, st, outer)
end

function pass_checks(v::AbstractArray, d::Integer, sign::Integer)
  1 <= d <= ndims(v) || throw(ArgumentError("dimension $d is not within 1:$(ndims(v))"))
  sign in (-1, 1) || throw(ArgumentError("sign must be -1 or 1"))
  all(>=(1), size(v)) || throw(ArgumentError("array size must be positive"))
  (size(v, 1) == 1 || strides(v)[1] == 1) || throw(ArgumentError("dimension 1 must be contiguous (stride 1)"))
  nothing
end

# The plan with the kernel k of type K.
function pass_build(::Type{K}, v::StridedArray{Complex{T},N}, d::Int, σ::Int, nt::Int, k,
                    lanes::Int, Sd::Int, offs::Vector{Int}) where {K,T,N}
  sz = size(v); st = strides(v); n = sz[d]
  B = d == 1 ? 1 : lane_block(lanes, n, k isa RChain ? nt : 1)
  nw = d == 1 ? n : B * n
  PassPlan{T,N,K}(sz, st, d, σ, k, lanes, Sd, offs, B, nt, thread_buffers(T, nw, 2nt))
end

"""
    plan_pass(v::StridedArray{<:Complex}, d; sign = -1, threads = 1)

Plan for the batched 1D DFT along dimension `d` of arrays with the size and strides of `v`
(`sign = -1` forward, `+1` unnormalised backward). Apply with `mul!(w, p, v)` (w with the same
size and strides, `w === v` in place) or `apply!(p, v)`. With `threads > 1` the lines are split
among threads, and a single long line (d = 1) uses the four-step algorithm; callers that thread
over many arrays themselves keep the default 1.

The kernels are specialised to the length. For plans that a library holds in one container and
compiles ahead of time, see [`plan_pass_runtime`](@ref).
"""
function plan_pass(v::StridedArray{Complex{T},N}, d::Integer; sign::Integer = -1, threads::Integer = 1) where {T<:FFTLFloat,N}
  pass_checks(v, d, sign)
  sz = size(v); st = strides(v); n = sz[d]; nt = max(1, Int(threads)); σ = Int(sign)
  lanes, Sd, offs = pass_layout(sz, st, Int(d))
  # a single long line on several threads: four-step
  k = d == 1 && length(offs) == 1 && use_fourstep(n, nt) ? FourStep{T}(n, σ, nt) :
      d == 1 ? kernel(T, n, σ) : kernel(T, n, σ, Sd)
  pass_build(typeof(k), v, Int(d), σ, nt, k, lanes, Sd, offs)
end

"""
    plan_pass_runtime(v::StridedArray{<:Complex}, d; sign = -1, threads = 1)

The same transform as [`plan_pass`](@ref), with run-time kernels only (RChain, resampling and composite
over RChain, four-step over RChain). The type of the plan is `PassPlan{T,N,RPassKern{T}}`, the same for
every length, so that a library can hold plans of any size in one container and compile them ahead of
time (`juliac --trim`). The kernels are the same as those of `plan_pass` for lengths with run-time
kernels; `plan_pass` is faster for short lines.
"""
function plan_pass_runtime(v::StridedArray{Complex{T},N}, d::Integer; sign::Integer = -1, threads::Integer = 1) where {T<:FFTLFloat,N}
  pass_checks(v, d, sign)
  sz = size(v); st = strides(v); n = sz[d]; nt = max(1, Int(threads)); σ = Int(sign)
  lanes, Sd, offs = pass_layout(sz, st, Int(d))
  k::RPassKern{T} = d == 1 && length(offs) == 1 && use_fourstep(n, nt) ? FourStep{T}(n, σ, nt, RChain) :
      runtime_kernel(T, n, σ, d == 1 ? 1 : Sd)
  pass_build(RPassKern{T}, v, Int(d), σ, nt, k, lanes, Sd, offs)
end

function Base.show(io::IO, p::PassPlan{T}) where {T}
  print(io, "FFTL pass plan, dim $(p.d) of $(join(p.sz, "×")) Complex{$T}, strides $(p.st), ",
        p.σ == -1 ? "forward" : "backward", ", ", length(p.offs), " group(s) of ", p.lanes, " lane(s)")
end

function check_layout(p::PassPlan, v)
  size(v) == p.sz && strides(v) == p.st ||
    throw(DimensionMismatch("pass plan for size $(p.sz) and strides $(p.st) applied to size $(size(v)), strides $(strides(v))"))
end

function LinearAlgebra.mul!(w::StridedArray{Complex{T},N}, p::PassPlan{T,N}, v::StridedArray{Complex{T},N}) where {T,N}
  check_layout(p, v); check_layout(p, w)
  GC.@preserve v w p begin
    run_pass!(Ptr{T}(pointer(w)), Ptr{T}(pointer(v)), p)
  end
  w
end

"""
    apply!(p::PassPlan, v)

The transform of `p` applied to `v` in place.
"""
apply!(p::PassPlan, v) = mul!(v, p, v)

function run_pass!(py::Ptr{T}, px::Ptr{T}, p::PassPlan{T}) where {T}
  k = p.kernel; CT = 2 * sizeof(T)
  if k isa FourStep
    off = p.offs[1] * CT
    run_fourstep!(py + off, px + off, k, p.nthreads)
  elseif p.d == 1
    # lines in groups on the tasks (kernels with buffers of their own on one task)
    each_line(length(p.offs), p.sz[1], k isa RChain ? p.nthreads : 1) do c, i
      local off = p.offs[i + 1] * CT
      run!(py + off, px + off, cptr(p.bufs[2c-1]), cptr(p.bufs[2c]), k, 1)
    end
  else
    B = p.B; nb = cld(p.lanes, B)
    nitems = nb * length(p.offs)
    ntasks = (k isa RChain && p.nthreads > 1 && length(p.offs) * p.lanes * p.sz[p.d] >= TUNING[].thread_min_work) ? p.nthreads : 1
    parallel_items(nitems, ntasks) do c, it
      local g, b; g, b = divrem(it, nb)
      local i0 = b * B
      local off = (p.offs[g + 1] + i0) * CT
      run_lanes!(py + off, px + off, cptr(p.bufs[2c-1]), cptr(p.bufs[2c]), k, p.Sd, p.Sd, min(B, p.lanes - i0))
    end
  end
  nothing
end
