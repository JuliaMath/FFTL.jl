# Batched 1D transforms along one dimension of a strided array: the operation an NFFT needs on
# the oversampled grid (boxes of a padded array, pruned and threaded by the caller).

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

"""
    plan_pass(v::StridedArray{<:Complex}, d; sign = -1, threads = 1)

Plan for the batched 1D DFT along dimension `d` of arrays with the size and strides of `v`
(`sign = -1` forward, `+1` unnormalised backward). Apply with `mul!(w, p, v)` (w with the same
size and strides, `w === v` in place) or `apply!(p, v)`. With `threads > 1` the lines are split
among threads; callers that thread over many arrays themselves keep the default 1.
"""
function plan_pass(v::StridedArray{Complex{T},N}, d::Integer; sign::Integer = -1, threads::Integer = 1) where {T<:FFTLFloat,N}
  1 <= d <= N || throw(ArgumentError("dimension $d is not within 1:$N"))
  sign in (-1, 1) || throw(ArgumentError("sign must be -1 or 1"))
  sz = size(v); st = strides(v)
  (sz[1] == 1 || st[1] == 1) || throw(ArgumentError("dimension 1 must be contiguous (stride 1)"))
  n = sz[d]
  nt = max(1, Int(threads))
  if d == 1
    lanes = 1; Sd = 1
    outer = ntuple(i -> i + 1, N - 1)
  else
    # merge dimensions 1, 2, … into the lane index while they are contiguous and before d
    m = 1; L = sz[1]
    while m + 1 < d && st[m+1] == L
      m += 1; L *= sz[m]
    end
    lanes = L; Sd = st[d]
    outer = Tuple(i for i in m+1:N if i != d)
  end
  offs = [sum(((I[j] - 1) * st[outer[j]] for j in eachindex(outer)); init = 0)
          for I in CartesianIndices(ntuple(j -> sz[outer[j]], length(outer)))] |> vec
  k = d == 1 ? kernel(T, n, sign) : kernel(T, n, sign, Sd)
  B = d == 1 ? 1 : lane_block(lanes, n, k isa Chain ? nt : 1)
  nw = d == 1 ? n : B * n
  PassPlan{T,N,typeof(k)}(sz, st, d, sign, k, lanes, Sd, offs, B, nt,
                          [Vector{Complex{T}}(undef, nw) for _ in 1:2nt])
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
  if p.d == 1
    ntasks = (k isa Chain && p.nthreads > 1 && length(p.offs) * p.sz[1] >= THREAD_MIN_WORK) ? p.nthreads : 1
    ng = min(length(p.offs), BLOCKS_PER_THREAD * ntasks)
    parallel_items(ng, ntasks) do c, g
      for i in chunk_range(g + 1, ng, length(p.offs))
        local off = p.offs[i + 1] * CT
        run!(py + off, px + off, cptr(p.bufs[2c-1]), cptr(p.bufs[2c]), k, 1)
      end
    end
  else
    B = p.B; nb = cld(p.lanes, B)
    nitems = nb * length(p.offs)
    ntasks = (k isa Chain && p.nthreads > 1 && length(p.offs) * p.lanes * p.sz[p.d] >= THREAD_MIN_WORK) ? p.nthreads : 1
    parallel_items(nitems, ntasks) do c, it
      local g, b; g, b = divrem(it, nb)
      local i0 = b * B
      local off = (p.offs[g + 1] + i0) * CT
      run_lanes!(py + off, px + off, cptr(p.bufs[2c-1]), cptr(p.bufs[2c]), k, p.Sd, p.Sd, min(B, p.lanes - i0))
    end
  end
  nothing
end
