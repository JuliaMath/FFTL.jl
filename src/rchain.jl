# Stockham chain with the radix sequence as run-time data. Same stages, twiddle tables and buffer
# schedule as `Chain` (plan.jl), but the stage sequence is a loop over plain code that selects
# the codelet for each stage from a fixed set (radix × sign × vector type). Nothing is compiled
# per length, and the execution path contains no call into a generated function whose body
# calls further generated functions, so it can be compiled ahead of time (`juliac --trim`).

"""
    RChain{T}(n, σ, rs = radices(n))

Stockham chain of length `n` and sign `σ` with the radix sequence `rs` stored as a vector.
"""
struct RChain{T} <: Kernel{T}
  n::Int
  σ::Int
  rs::Vector{Int}
  tw::Vector{Vector{Complex{T}}}      # stage i: w^(kp) at (R-1)p + k
  tw1::Vector{Complex{T}}             # first stage: w^(kp) at (k-1)m + p
end

RChain{T}(n::Int, σ::Int, rs::AbstractVector{Int} = radices(n)) where {T} =
  cached(() -> build_rchain(T, n, σ, rs), CacheKey(:rchain, T, n, σ, collect(Int, rs)))::RChain{T}

function build_rchain(::Type{T}, n::Int, σ::Int, rs::AbstractVector{Int}) where {T}
  prod(rs; init = 1) == n || throw(ArgumentError("radices do not multiply to the length"))
  all(in(RADICES), rs) || throw(ArgumentError("radix without a codelet in RADICES"))
  W = unit_roots(σ, n)
  tw = Vector{Complex{T}}[]
  L = n
  for r in rs
    push!(tw, stage_twiddles(T, W, r, L))
    L ÷= r
  end
  RChain{T}(n, σ, collect(Int, rs), tw, first_twiddles(T, W, isempty(rs) ? 1 : rs[1]))
end

"""
    RADICES

The radices `radices` produces for any length and stride (checked by the tests); `RChain` has a
codelet for each of them.
"""
const RADICES = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 29, 31)

# Codelet selection: nested branches on sign, vector type (index into vec_ladder(T)) and radix,
# generated here for both element types so that every call has constant type arguments.
for T in (Float32, Float64)
  ladder = vec_ladder(T)
  kinds = (
    (:rstage!,       (:py, :px, :pt, :m, :s),                     (r, σ, V) -> :(stage!(py, px, pt, m, s, Val($r), Val($σ), $V))),
    (:rstage_last!,  (:py, :px, :s),                              (r, σ, V) -> :(stage_last!(py, px, s, Val($r), Val($σ), $V))),
    (:rstage_group!, (:py, :px, :pt, :m),                         (r, σ, V) -> :(stage_first_group!(py, px, pt, m, Val($r), Val($σ), $V))),
    (:rstage_lanes!, (:py, :px, :pt, :m, :s0, :sx, :sy, :B),      (r, σ, V) -> :(stage_lanes!(py, px, pt, m, s0, sx, sy, B, Val($r), Val($σ), $V))),
    (:rstage_last_lanes!, (:py, :px, :s0, :sx, :sy, :B),          (r, σ, V) -> :(stage_last_lanes!(py, px, s0, sx, sy, B, Val($r), Val($σ), $V))))
  for (name, args, call) in kinds
    typed = [a in (:py, :px, :pt) ? :($a::Ptr{$T}) : :($a::Int) for a in args]
    byσ = map((-1, 1)) do σ
      byv = map(enumerate(ladder)) do (vi, V)
        # the grouped first stage exists for 2 or 4 complex values per vector only; run! never
        # selects another vector type for it, and code compiled ahead of time must not contain
        # the call (its generator would fail)
        name === :rstage_group! && !(ncomplex(V) in (2, 4)) && return :()
        byr = [:(R == $r && return $(call(r, σ, V))) for r in RADICES]
        :(if vi == $vi; $(byr...); end)
      end
      :(if σ == $σ; $(byv...); end)
    end
    @eval function $name($(typed...), R::Int, σ::Int, vi::Int)
      $(byσ...)
      throw(ArgumentError("no codelet for this radix, sign and vector type"))
    end
  end
end

# complex values per vector for each entry of vec_ladder(T), widest first, as constants (a map
# over the tuple of types is not folded by the compiler and would make the schedule dynamic)
for T in (Float32, Float64)
  @eval @inline ladder_ncomplex(::Type{$T}) = $(map(ncomplex, vec_ladder(T)))
end

function run!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T}, S::Int) where {T}
  rs = P.rs; K = length(rs); σ = P.σ; n = P.n
  if K == 0
    py == px || unsafe_copyto!(py, px, 2S)
    return nothing
  end
  ncs = ladder_ncomplex(T)
  single = S == 1
  # base: the widest complex count that divides S (as stride_dispatch in plan.jl)
  base = 1
  if !single
    for C in ncs
      if C > 1 && S % C == 0
        base = C
        break
      end
    end
  end
  # vector type of a grouped first stage: the widest with 2 or 4 complex values dividing m
  grp = 0
  for vi in eachindex(ncs)
    if ncs[vi] in (2, 4) && (n ÷ rs[1]) % ncs[vi] == 0
      grp = vi
      break
    end
  end
  inplace = py == px
  GC.@preserve P begin
    L = n; s = S; sst = 1; src = px
    for i in 1:K
      r = rs[i]
      dst = inplace ? (i == K ? px : (isodd(K - i) ? pw1 : pw2)) : (iseven(K - i) ? py : pw1)
      vi = 1
      while (base * sst) % ncs[vi] != 0
        vi += 1
      end
      if i == K
        rstage_last!(dst, src, s, r, σ, vi)
      elseif single && i == 1 && grp != 0 && ncs[vi] == 1
        m = L ÷ r
        rstage_group!(dst, src, cptr(P.tw1), m, r, σ, grp)
        L = m; s *= r
      else
        m = L ÷ r
        rstage!(dst, src, cptr(P.tw[i]), m, s, r, σ, vi)
        L = m; s *= r
      end
      src = dst
      sst *= r
    end
  end
  nothing
end

function run_lanes!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T}, Sx::Int, Sy::Int, B::Int) where {T}
  rs = P.rs; K = length(rs); σ = P.σ
  if K == 0
    lane_copy!(py, px, Sy, Sx, B, 1)
    return nothing
  end
  ncs = ladder_ncomplex(T)
  # widest vector type whose complex count divides B, else the narrowest
  vi = length(ncs)
  for j in 1:length(ncs)-1
    if B % ncs[j] == 0
      vi = j
      break
    end
  end
  GC.@preserve P begin
    if K == 1    # one stage loads all inputs of a butterfly before it stores: py may equal px
      rstage_last_lanes!(py, px, 1, Sx, Sy, B, rs[1], σ, vi)
      return nothing
    end
    L = P.n; s0 = 1; src = px; srcS = Sx
    for i in 1:K
      r = rs[i]
      if i == K
        rstage_last_lanes!(py, src, s0, srcS, Sy, B, r, σ, vi)
      else
        dst = isodd(i) ? pw1 : pw2
        m = L ÷ r
        rstage_lanes!(dst, src, cptr(P.tw[i]), m, s0, srcS, B, B, r, σ, vi)
        L = m; s0 *= r
        src = dst; srcS = B
      end
    end
  end
  nothing
end
