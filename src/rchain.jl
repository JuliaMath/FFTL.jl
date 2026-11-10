# Stockham chain with the radix sequence as run-time data. Same stages, twiddle tables, buffer
# schedule and vector-type rules as `Chain` (plan.jl), but the stage sequence is a loop over
# plain code that calls the codelet of each stage from a fixed set (radix × sign × vector type),
# selected by an integer index. Nothing is compiled per length, and the execution path contains
# no call into a generated function whose body calls further generated functions, so it can be
# compiled ahead of time (`juliac --trim`).

"""
    RADICES

The radices `radices` produces for any length and stride (checked by the tests); `RChain` has a
codelet for each of them.
"""
const RADICES = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 23, 29, 31)

# complex values per vector for each entry of vec_ladder(T), widest first, as constants (a map
# over the tuple of types is not folded by the compiler and would make the schedule dynamic)
for T in (Float32, Float64)
  @eval @inline ladder_ncomplex(::Type{$T}) = $(map(ncomplex, vec_ladder(T)))
end

# Position of each radix in RADICES (0 for radices without a codelet).
const RADIX_POS = let t = zeros(Int, maximum(RADICES))
  for (i, r) in enumerate(RADICES)
    t[r] = i
  end
  t
end

# Codelet index of (sign, vector type vi of vec_ladder(T), radix): one integer, so that the
# dispatch below is a single switch. Called per stage, so it is a table lookup.
@inline function codelet_index(::Type{T}, σ::Int, vi::Int, r::Int) where {T}
  ri = r <= length(RADIX_POS) ? RADIX_POS[r] : 0
  ri == 0 && throw(ArgumentError("radix without a codelet in RADICES"))
  ((σ == 1 ? 1 : 0) * length(ladder_ncomplex(T)) + vi - 1) * length(RADICES) + ri
end

# Radices whose codelets are compiled on first use, not with the package: they are 54 % of the
# statements of the stage kernels and rarely needed. The preference `lazy_radices` is a list of
# radices from RADICES; an empty list compiles every codelet with the package, which a trimmed build
# needs because juliac compiles nothing at run time.
using Preferences: @load_preference
const LAZY_RADICES = Tuple(Int.(@load_preference("lazy_radices", [17, 19, 23, 29, 31])))
all(in(RADICES), LAZY_RADICES) || error("LAZY_RADICES must be radices from RADICES")

# Whether the factorisation of n has a prime factor whose codelet is compiled on first use.
has_lazy_factor(n::Int) = any(in(LAZY_RADICES), keys(factorize(n)[1]))

# Codelet dispatch: one function per kind of stage call, a switch over the codelet index,
# generated for both element types so that every call has constant type arguments. Codelets of
# LAZY_RADICES are not in the switch; `<kind>_lazy` takes their index and calls the stage kernel
# through a callable that is created and compiled at the first call.
for T in (Float32, Float64)
  ladder = vec_ladder(T)
  LTYPES = Tuple(ladder)
  NR = length(RADICES); NL = length(ladder)
  kinds = (
    (:cstage!,       (:py, :px, :pt, :m, :s),                 (r, σ, V) -> :(stage!(py, px, pt, m, s, Val($r), Val($σ), $V))),
    (:cstage_range!, (:py, :px, :pt, :m, :s, :qlo, :qhi),     (r, σ, V) -> :(stage_range!(py, px, pt, m, s, qlo, qhi, 0, m, Val($r), Val($σ), $V))),
    (:cstage_last!,  (:py, :px, :s),                          (r, σ, V) -> :(stage_last!(py, px, s, Val($r), Val($σ), $V))),
    (:cstage_last_range!, (:py, :px, :s, :qlo, :qhi),         (r, σ, V) -> :(stage_last!(py, px, s, qlo, qhi, Val($r), Val($σ), $V))),
    (:cstage_group!, (:py, :px, :pt, :m),                     (r, σ, V) -> :(stage_first_group!(py, px, pt, m, Val($r), Val($σ), $V))),
    (:cstage_lanes!, (:py, :px, :pt, :m, :s0, :sx, :sy, :B),  (r, σ, V) -> :(stage_lanes!(py, px, pt, m, s0, sx, sy, B, Val($r), Val($σ), $V))),
    (:cstage_last_lanes!, (:py, :px, :s0, :sx, :sy, :B),      (r, σ, V) -> :(stage_last_lanes!(py, px, s0, sx, sy, B, Val($r), Val($σ), $V))))
  for (name, args, call) in kinds
    typed = [a in (:py, :px, :pt) ? :($a::Ptr{$T}) : :($a::Int) for a in args]
    cases = Any[]
    for (si, σ) in enumerate((-1, 1)), (vi, V) in enumerate(ladder), (ri, r) in enumerate(RADICES)
      # the grouped first stage exists for 2 or 4 complex values per vector only; the schedule
      # never selects another vector type for it, and code compiled ahead of time must not
      # contain the call (its generator would fail)
      name === :cstage_group! && !(ncomplex(V) in (2, 4)) && continue
      r in LAZY_RADICES && continue
      c = ((si - 1) * NL + vi - 1) * NR + ri
      push!(cases, :(c == $c && return $(call(r, σ, V))))
    end
    # Without lazy radices there is no run-time call at all (the trimmed library needs none).
    fallback = isempty(LAZY_RADICES) ? :(throw(ArgumentError("no codelet for this index"))) :
      :(return $(Symbol(name, :_lazy))($(args...), c))
    @eval function $name($(typed...), c::Int)
      $(cases...)
      $fallback
    end
    isempty(LAZY_RADICES) && continue
    # One callable singleton per (radix, sign, vector type): the first call of a codelet creates it
    # and compiles its kernel, later calls find it in a table by index. A call through the table
    # is a dynamic call of a single method, much cheaper than the dispatch of the stage kernel
    # on run-time `Val`s.
    callable = Symbol(name, :_, T, :_codelet)
    table = Symbol(name, :_, T, :_codelets)
    @eval struct $callable{R,σ,VT} end
    @eval function (::$callable{R,σ,VT})($(typed...)) where {R,σ,VT}
      return $(call(:R, :σ, :VT))
    end
    @eval const $table = Any[nothing for _ in 1:2 * $NL * $NR]
    # Index c = ((si - 1) * NL + vi - 1) * NR + ri, with ri = 1 … NR into RADICES.
    @eval @noinline function $(Symbol(name, :_, T, :_make))(c::Int)
      ri = (c - 1) % $NR + 1
      rest = (c - 1) ÷ $NR
      vi = rest % $NL + 1
      si = rest ÷ $NL + 1
      return $callable{RADICES[ri], si == 2 ? 1 : -1, $LTYPES[vi]}()
    end
    @eval @noinline function $(Symbol(name, :_lazy))($(typed...), c::Int)
      f = $table[c]
      f === nothing && (f = $table[c] = $(Symbol(name, :_, T, :_make))(c))
      return f($(args...))
    end
  end
end

# One stage call of a single line (S = 1), fixed when the chain is built.
const OP_MID = 1          # plain stage, q vectorised over s
const OP_MID_TAIL = 2     # q in vectors up to s - rem, the scalar remainder after it
const OP_LAST = 3
const OP_LAST_TAIL = 4
const OP_GROUP = 5        # first stage vectorised over the butterflies p

struct StageCall
  op::Int
  c::Int                  # codelet of the vectorised part
  c1::Int                 # codelet of the scalar remainder (one complex value per vector)
  m::Int
  s::Int
  rem::Int
end

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
  single::Vector{StageCall}           # the stage calls of a single line
end

# Either kind of Stockham chain: the compiled Chain (short smooth lengths) or the run-time RChain.
const AnyChain = Union{Chain,RChain}

RChain{T}(n::Int, σ::Int, rs::AbstractVector{Int} = radices(n)) where {T} =
  cached(() -> build_rchain(T, n, σ, rs), CacheKey(:rchain, T, n, σ, collect(Int, rs)))::RChain{T}

function build_rchain(::Type{T}, n::Int, σ::Int, rs::AbstractVector{Int}) where {T}
  all(in(RADICES), rs) || throw(ArgumentError("radix without a codelet in RADICES"))
  tw, tw1 = chain_tables(T, n, σ, rs)
  RChain{T}(n, σ, collect(Int, rs), tw, tw1, single_schedule(T, n, σ, collect(Int, rs)))
end

# The stage calls of a single line, with the rules of `chain_body` for S = 1: stage i takes the
# widest vector type whose complex count fits its stride s = R_1⋯R_(i-1), the remainder
# s mod C runs one complex value at a time; the first stage is grouped over the butterflies
# when its stride is 1.
function single_schedule(::Type{T}, n::Int, σ::Int, rs::Vector{Int}) where {T}
  ncs = ladder_ncomplex(T)
  nv = length(ncs)
  group = 0
  for vi in 1:nv
    if ncs[vi] in (2, 4)
      group = vi
      break
    end
  end
  calls = StageCall[]
  K = length(rs)
  L = n; s = 1
  for i in 1:K
    r = rs[i]
    vi = 1
    while ncs[vi] > s
      vi += 1
    end
    rem = s % ncs[vi]
    c = codelet_index(T, σ, vi, r); c1 = codelet_index(T, σ, nv, r)
    m = L ÷ r
    if i == K
      push!(calls, StageCall(rem == 0 ? OP_LAST : OP_LAST_TAIL, c, c1, m, s, rem))
    elseif i == 1 && group != 0 && ncs[vi] == 1
      push!(calls, StageCall(OP_GROUP, codelet_index(T, σ, group, r), c1, m, s, 0))
    else
      push!(calls, StageCall(rem == 0 ? OP_MID : OP_MID_TAIL, c, c1, m, s, rem))
    end
    L = m; s *= r
  end
  calls
end

# Destination of stage i of K. Out of place the stages alternate between py and pw1 and end in
# py; in place the first stage reads px, the last writes px, the ones between alternate between
# the two work buffers (a single stage loads all inputs of a butterfly before it stores).
@inline stage_dst(py, px, pw1, pw2, i, K, inplace) =
  inplace ? (i == K ? px : (isodd(K - i) ? pw1 : pw2)) : (iseven(K - i) ? py : pw1)

function run!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T}, S::Int) where {T}
  K = length(P.rs)
  if K == 0
    py == px || unsafe_copyto!(py, px, 2S)
    return nothing
  end
  if P.n <= SMALL_N && SMALL_INDEX[P.n] > 0
    i = SMALL_INDEX[P.n] + (P.σ == 1 ? 1 : 0)
    small_run!(py, px, pw1, pw2, P, i, S) && return nothing
  end
  S == 1 && return run_single!(py, px, pw1, pw2, P)
  ncs = ladder_ncomplex(T)
  nv = length(ncs)
  σ = P.σ
  # base: the widest complex count that divides S (as stride_dispatch in plan.jl)
  base = 1
  for C in ncs
    if C > 1 && S % C == 0
      base = C
      break
    end
  end
  inplace = py == px
  GC.@preserve P begin
    L = P.n; s = S; sst = 1; src = px
    for i in 1:K
      r = P.rs[i]
      dst = stage_dst(py, px, pw1, pw2, i, K, inplace)
      # base > 1: the widest vector type dividing base·sst (no remainder); base = 1 (odd S):
      # the widest one that fits sst, with the remainder of s split off at run time
      vi = 1
      if base > 1
        while (base * sst) % ncs[vi] != 0
          vi += 1
        end
      else
        while ncs[vi] > sst
          vi += 1
        end
      end
      c = codelet_index(T, σ, vi, r)
      rem = s % ncs[vi]
      if i == K
        if rem == 0
          cstage_last!(dst, src, s, c)
        else
          cstage_last_range!(dst, src, s, 0, s - rem, c)
          cstage_last_range!(dst, src, s, s - rem, s, codelet_index(T, σ, nv, r))
        end
      else
        m = L ÷ r
        pt = cptr(P.tw[i])
        if rem == 0
          cstage!(dst, src, pt, m, s, c)
        else
          cstage_range!(dst, src, pt, m, s, 0, s - rem, c)
          cstage_range!(dst, src, pt, m, s, s - rem, s, codelet_index(T, σ, nv, r))
        end
        L = m; s *= r
      end
      src = dst
      sst *= r
    end
  end
  nothing
end

function run_single!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T}) where {T}
  calls = P.single
  K = length(calls)
  inplace = py == px
  GC.@preserve P begin
    src = px
    for i in 1:K
      st = calls[i]
      dst = stage_dst(py, px, pw1, pw2, i, K, inplace)
      op = st.op
      if op == OP_MID
        cstage!(dst, src, cptr(P.tw[i]), st.m, st.s, st.c)
      elseif op == OP_GROUP
        cstage_group!(dst, src, cptr(P.tw1), st.m, st.c)
      elseif op == OP_LAST
        cstage_last!(dst, src, st.s, st.c)
      elseif op == OP_MID_TAIL
        pt = cptr(P.tw[i])
        cstage_range!(dst, src, pt, st.m, st.s, 0, st.s - st.rem, st.c)
        cstage_range!(dst, src, pt, st.m, st.s, st.s - st.rem, st.s, st.c1)
      else
        cstage_last_range!(dst, src, st.s, 0, st.s - st.rem, st.c)
        cstage_last_range!(dst, src, st.s, st.s - st.rem, st.s, st.c1)
      end
      src = dst
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
  if P.n <= SMALL_N && SMALL_INDEX[P.n] > 0
    i = SMALL_INDEX[P.n] + (P.σ == 1 ? 1 : 0)
    small_lanes_run!(py, px, pw1, pw2, P, i, Sx, Sy, B) && return nothing
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
      cstage_last_lanes!(py, px, 1, Sx, Sy, B, codelet_index(T, σ, vi, rs[1]))
      return nothing
    end
    L = P.n; s0 = 1; src = px; srcS = Sx
    for i in 1:K
      r = rs[i]
      c = codelet_index(T, σ, vi, r)
      if i == K
        cstage_last_lanes!(py, src, s0, srcS, Sy, B, c)
      else
        dst = isodd(i) ? pw1 : pw2
        m = L ÷ r
        cstage_lanes!(dst, src, cptr(P.tw[i]), m, s0, srcS, B, B, c)
        L = m; s0 *= r
        src = dst; srcS = B
      end
    end
  end
  nothing
end

# Small lengths: the stage loop of a single line runs as compiled code for each smooth length up
# to SMALL_N and both signs. The entries of a length are consecutive (σ = -1, then σ = +1); one
# table lookup per call finds them. The compiled chain uses the tables of the RChain, and its
# radix sequence is checked against the compiled one, so any other plan takes the stage loop.
const SMALL_N = 128
const SMALL_ENTRIES = [(n, σ) for n in 2:SMALL_N if smooth(n) && !has_lazy_factor(n) for σ in (-1, 1)]
const SMALL_INDEX = let t = zeros(Int, SMALL_N)
  for (i, (n, σ)) in enumerate(SMALL_ENTRIES)
    σ == -1 && (t[n] = i)
  end
  t
end

@noinline function small_chain!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T}, S::Int,
                              ::Type{C}, ::Val{K}) where {T,C<:Chain{T},K}
  c = C(P.n, ntuple(i -> P.tw[i], Val(K)), P.tw1)
  GC.@preserve P run!(py, px, pw1, pw2, c, S)
end

@noinline function small_lanes_chain!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T},
                                      Sx::Int, Sy::Int, B::Int, ::Type{C}, ::Val{K}) where {T,C<:Chain{T},K}
  c = C(P.n, ntuple(i -> P.tw[i], Val(K)), P.tw1)
  GC.@preserve P run_lanes!(py, px, pw1, pw2, c, Sx, Sy, B)
end

# Binary tree of compares over the entry index (a flat chain of ~120 compares costs ~50 ns per
# call). Each leaf checks the radix sequence of the RChain against the compiled one. `leaf` gives
# the statement of a leaf for the chain type C with K stages.
function small_tree(T, lo, hi, leaf)
  if lo == hi
    n, σ = SMALL_ENTRIES[lo]
    rs = Tuple(radices(n, 1; esize = sizeof(Complex{T})))
    K = length(rs)
    same = Any[:(i == $lo), :(length(P.rs) == $K)]
    for (j, r) in enumerate(rs)
      push!(same, :(P.rs[$j] == $r))
    end
    return :(if $(Expr(:&&, same...))
               $(leaf(:(Chain{$T,$rs,$σ,$K}), K))
               return true
             end)
  end
  mid = (lo + hi) ÷ 2
  :(if i <= $mid
      $(small_tree(T, lo, mid, leaf))
    else
      $(small_tree(T, mid + 1, hi, leaf))
    end)
end

for T in (Float32, Float64)
  @eval function small_run!(py::Ptr{$T}, px::Ptr{$T}, pw1::Ptr{$T}, pw2::Ptr{$T}, P::RChain{$T}, i::Int, S::Int)
    $(small_tree(T, 1, length(SMALL_ENTRIES),
                 (C, K) -> :(small_chain!(py, px, pw1, pw2, P, S, $C, Val($K))))
    )
    false
  end
  @eval function small_lanes_run!(py::Ptr{$T}, px::Ptr{$T}, pw1::Ptr{$T}, pw2::Ptr{$T}, P::RChain{$T},
                                  i::Int, Sx::Int, Sy::Int, B::Int)
    $(small_tree(T, 1, length(SMALL_ENTRIES),
                 (C, K) -> :(small_lanes_chain!(py, px, pw1, pw2, P, Sx, Sy, B, $C, Val($K))))
    )
    false
  end
end

# Rows of a plane: `count` single lines of the chain, row j at py + j·rowstride complex values.
# A compiled chain takes all rows in one call: one dispatch per plane instead of one per row.
@noinline function small_rows_chain!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T},
                                     count::Int, rowstride::Int, ::Type{C}, ::Val{K}) where {T,C<:Chain{T},K}
  c = C(P.n, ntuple(i -> P.tw[i], Val(K)), P.tw1)
  bytes = rowstride * 2 * sizeof(T)
  GC.@preserve P for j in 0:count-1
    run!(py + j * bytes, px + j * bytes, pw1, pw2, c, 1)
  end
  nothing
end

for T in (Float32, Float64)
  @eval function small_rows_run!(py::Ptr{$T}, px::Ptr{$T}, pw1::Ptr{$T}, pw2::Ptr{$T}, P::RChain{$T},
                                 i::Int, count::Int, rowstride::Int)
    $(small_tree(T, 1, length(SMALL_ENTRIES),
                 (C, K) -> :(small_rows_chain!(py, px, pw1, pw2, P, count, rowstride, $C, Val($K))))
    )
    false
  end
end

# Generic: one call of run! per row (any kernel), compiled chains through their dispatch.
function run_rows!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T}, count::Int, rowstride::Int) where {T}
  if P.n <= SMALL_N && SMALL_INDEX[P.n] > 0
    i = SMALL_INDEX[P.n] + (P.σ == 1 ? 1 : 0)
    small_rows_run!(py, px, pw1, pw2, P, i, count, rowstride) && return nothing
  end
  bytes = rowstride * 2 * sizeof(T)
  for j in 0:count-1
    run!(py + j * bytes, px + j * bytes, pw1, pw2, P, 1)
  end
  nothing
end

function run_rows!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::Kernel{T}, count::Int, rowstride::Int) where {T}
  bytes = rowstride * 2 * sizeof(T)
  for j in 0:count-1
    run!(py + j * bytes, px + j * bytes, pw1, pw2, P, 1)
  end
  nothing
end
