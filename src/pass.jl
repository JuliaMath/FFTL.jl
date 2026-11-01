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
struct PassPlan{T<:FFTLFloat,N,K<:Kernel{T},Pr<:Pruning}
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
  prune::Pr
end

"""
    plan_pass(v::StridedArray{<:Complex}, d; sign = -1, threads = 1, nonzero = nothing, needed = nothing)

Plan for the batched 1D DFT along dimension `d` of arrays with the size and strides of `v`
(`sign = -1` forward, `+1` unnormalised backward). Apply with `mul!(w, p, v)` (w with the same
size and strides, `w === v` in place) or `apply!(p, v)`. With `threads > 1` the lines are split
among threads, and a single long line (d = 1) uses the four-step algorithm; callers that thread
over many arrays themselves keep the default 1.

`nonzero` (ranges of indices along d, e.g. `(1:a, b:n)`) declares that each line is zero
outside these ranges; `needed` that only these outputs of each line are used (the others may
hold anything afterwards). The first resp. last stage then skips the zero inputs resp. the
unneeded outputs.
"""
function plan_pass(v::StridedArray{Complex{T},N}, d::Integer; sign::Integer = -1, threads::Integer = 1,
                   nonzero = nothing, needed = nothing) where {T<:FFTLFloat,N}
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
  # a single long line on several threads: four-step
  k = d == 1 && length(offs) == 1 && use_fourstep(n, nt) ? FourStep{T}(n, sign, nt) :
      d == 1 ? kernel(T, n, sign) : kernel(T, n, sign, Sd)
  B = d == 1 ? 1 : lane_block(lanes, n, k isa Chain ? nt : 1)
  nw = d == 1 ? n : B * n
  pr = k isa Chain ? pruning(k, nonzero, needed; lanes = d > 1) : NOPRUNE
  PassPlan{T,N,typeof(k),typeof(pr)}(sz, st, d, sign, k, lanes, Sd, offs, B, nt,
                                     [Vector{Complex{T}}(undef, nw) for _ in 1:2nt], pr)
end

# Segments of constant mask over units of `cu` consecutive indices 0…len-1; mask(range) gives the
# mask of one unit.
function segments(mask, len, cu)
  ranges = UnitRange{Int}[]; masks = Int[]
  for u0 in 0:cu:len-1
    r = u0:min(u0 + cu, len)-1
    mk = mask(r)
    if !isempty(masks) && masks[end] == mk
      ranges[end] = first(ranges[end]):last(r)
    else
      push!(ranges, r); push!(masks, mk)
    end
  end
  ranges, masks
end

inranges(ranges, n) = (b = falses(n); foreach(r -> (b[r] .= true), ranges); b)

"""
    pruning(chain, nonzero, needed; lanes)

Segments and masks of the first and last stage of `chain` for lines that are zero outside
`nonzero` and of which only `needed` is used (index ranges along the line; `nothing`: no
pruning). `lanes`: the chain runs in lane blocks (units of one butterfly); otherwise units follow
the vectorisation over butterflies of the single-line kernels.
"""
# Pruning only touches the first or last of K stages, so it saves at most about (1 - f)/K of a
# pass for a nonzero (needed) fraction f; below PRUNE_MIN_ZERO zeros the segment overhead
# outweighs that (M1: σ = 1.25 lines gained 0–3 %, short Float32 lines lost up to 13 %).
const PRUNE_MIN_ZERO = 1 / 3

function pruning(c::Chain{T,Rs}, nonzero, needed; lanes::Bool) where {T,Rs}
  n = c.n; K = length(Rs)
  K >= 2 || return NOPRUNE
  worth(ranges) = ranges !== nothing && 1 - count(inranges(ranges, n)) / n >= PRUNE_MIN_ZERO
  nonzero = worth(nonzero) ? nonzero : nothing
  needed = worth(needed) ? needed : nothing
  ladder = vec_ladder(T)
  fr = UnitRange{Int}[]; fm = Int[]; lr = UnitRange{Int}[]; lm = Int[]
  if nonzero !== nothing
    nz = inranges(nonzero, n)
    r1 = Rs[1]; m = n ÷ r1
    g = findfirst(V -> ncomplex(V) in (2, 4) && m % ncomplex(V) == 0, ladder)
    cu = lanes || g === nothing ? 1 : ncomplex(ladder[g])
    # input j of butterfly p is x[p + (j-1)m]; zero for the unit if zero for all its p
    fr, fm = segments(m, cu) do r
      sum((all(p -> !nz[p + (j - 1) * m + 1], r) ? 1 << (j - 1) : 0) for j in 1:r1)
    end
  end
  if needed !== nothing
    nd = inranges(needed, n)
    rK = Rs[K]; sl = n ÷ rK
    V = ladder[findfirst(V -> sl % ncomplex(V) == 0, ladder)]
    cu = lanes ? 1 : ncomplex(V)
    # output k of position q is y[q + (k-1)·sl]; needed for the unit if needed for any q
    lr, lm = segments(sl, cu) do r
      sum((any(q -> nd[q + (k - 1) * sl + 1], r) ? 1 << (k - 1) : 0) for k in 1:rK)
    end
  end
  Pruning{Tuple(fm),Tuple(lm)}(fr, lr)
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
    ntasks = (k isa Chain && p.nthreads > 1 && length(p.offs) * p.sz[1] >= THREAD_MIN_WORK) ? p.nthreads : 1
    ng = min(length(p.offs), BLOCKS_PER_THREAD * ntasks)
    parallel_items(ng, ntasks) do c, g
      for i in chunk_range(g + 1, ng, length(p.offs))
        local off = p.offs[i + 1] * CT
        run!(py + off, px + off, cptr(p.bufs[2c-1]), cptr(p.bufs[2c]), k, 1, p.prune)
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
      run_lanes!(py + off, px + off, cptr(p.bufs[2c-1]), cptr(p.bufs[2c]), k, p.Sd, p.Sd, min(B, p.lanes - i0), p.prune)
    end
  end
  nothing
end

# kernels without pruning support (Bluestein, four-step) ignore it
run!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::Kernel{T}, S::Int, ::Pruning) where {T} =
  run!(py, px, pw1, pw2, P, S)
run_lanes!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::Kernel{T}, Sx::Int, Sy::Int, B::Int, ::Pruning) where {T} =
  run_lanes!(py, px, pw1, pw2, P, Sx, Sy, B)
