# Stockham chain with the radix sequence as run-time data: the stage sequence is a loop over
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
    (:cstage_small!, (:py, :px, :pt, :m),                     (r, σ, V) -> :(stage_small!(py, px, pt, m, Val($r), Val($σ), $V))),
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
      # stage_small! exists for the widest vector type and power-of-two radices
      name === :cstage_small! && !(vi == 1 && ispow2(r)) && continue
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
const OP_SMALL = 6        # first stage over the butterflies p on the widest vector (stage_small!)

struct StageCall{T}
  op::Int
  c::Int                  # codelet of the vectorised part
  c1::Int                 # codelet of the scalar remainder (one complex value per vector)
  m::Int
  s::Int
  rem::Int
  pt::Ptr{T}              # twiddle table of the stage (C_NULL for a last stage), see link_calls!
end

# One stage of a batch of lines or of a lane block: the vector type is chosen per call from the
# stride, so the record holds the codelet index of the widest vector type (the index of vector
# type vi is c + (vi - 1)·length(RADICES)) and the twiddle table.
struct BatchStage{T}
  c::Int
  pt::Ptr{T}              # C_NULL for the last stage
end

"""
    RChain{T}(n, σ, rs = radices(n))

Stockham chain of length `n` and sign `σ` with the radix sequence `rs` stored as a vector.
"""
mutable struct RChain{T} <: Kernel{T}
  const n::Int
  const σ::Int
  const rs::Vector{Int}
  const tw::Vector{Vector{Complex{T}}}      # stage i: w^(kp) at (R-1)p + k
  const tw1::Vector{Complex{T}}             # first stage: w^(kp) at (k-1)m + p
  const single::Vector{StageCall{T}}        # the stage calls of a single line
  const batch::Vector{BatchStage{T}}        # the stages of a batch or a lane block
  const one::Int                            # codelet of a single line that is one stage, else 0
  linked::UInt                              # address of the chain when its stage records were linked
end

RChain{T}(n::Int, σ::Int, rs::AbstractVector{Int} = radices(n)) where {T} =
  cached(() -> build_rchain(T, n, σ, rs), CacheKey(:rchain, T, n, σ, collect(Int, rs)))::RChain{T}

function build_rchain(::Type{T}, n::Int, σ::Int, rs::AbstractVector{Int}) where {T}
  all(in(RADICES), rs) || throw(ArgumentError("radix without a codelet in RADICES"))
  tw, tw1 = chain_tables(T, n, σ, rs)
  calls = single_schedule(T, n, σ, collect(Int, rs))
  batch = [BatchStage{T}(codelet_index(T, σ, 1, r), Ptr{T}(C_NULL)) for r in rs]
  P = RChain{T}(n, σ, collect(Int, rs), tw, tw1, calls, batch, length(calls) == 1 ? calls[1].c : 0, UInt(0))
  link_calls!(P)
  P
end

# The stage records carry the address of their twiddle table, so that a stage loop reads one
# flat vector and nothing else (a load through the chain inside the loop needs a GC frame, which
# costs more than a short stage). The tables belong to the chain and are never resized. The
# addresses are valid for the chain object they were taken for: a copy, or a chain that was stored
# in a package image and loaded into another process, is at another address than `linked` and is
# linked again before its records are used.
@inline linked(P::RChain) = P.linked == UInt(pointer_from_objref(P))

@noinline function link_calls!(P::RChain{T}) where {T}
  calls = P.single
  for i in eachindex(calls)
    st = calls[i]
    pt = st.op == OP_GROUP || st.op == OP_SMALL ? cptr(P.tw1) :
         st.op == OP_MID || st.op == OP_MID_TAIL ? cptr(P.tw[i]) : Ptr{T}(C_NULL)
    calls[i] = StageCall{T}(st.op, st.c, st.c1, st.m, st.s, st.rem, pt)
  end
  batch = P.batch
  for i in 1:length(batch)-1
    batch[i] = BatchStage{T}(batch[i].c, cptr(P.tw[i]))
  end
  P.linked = UInt(pointer_from_objref(P))
  nothing
end

# The stage calls of a single line: stage i takes the
# widest vector type whose complex count fits its stride s = R_1⋯R_(i-1), the remainder
# s mod C runs one complex value at a time; the first stage is grouped over the butterflies
# when its stride is 1, with stage_small! where that is wider.
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
  calls = StageCall{T}[]
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
    if i == 1 && small_first(ncs, group, r, n)
      push!(calls, StageCall{T}(OP_SMALL, codelet_index(T, σ, 1, r), c1, m, s, 0, Ptr{T}(C_NULL)))
    elseif i == K
      push!(calls, StageCall{T}(rem == 0 ? OP_LAST : OP_LAST_TAIL, c, c1, m, s, rem, Ptr{T}(C_NULL)))
    elseif i == 1 && group != 0 && ncs[vi] == 1
      push!(calls, StageCall{T}(OP_GROUP, codelet_index(T, σ, group, r), c1, m, s, 0, Ptr{T}(C_NULL)))
    else
      push!(calls, StageCall{T}(rem == 0 ? OP_MID : OP_MID_TAIL, c, c1, m, s, rem, Ptr{T}(C_NULL)))
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

# The single line is kept apart from the batch: the batch needs a GC frame, which a short single
# line must not pay for.
@inline run!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T}, S::Int) where {T} =
  S == 1 ? run_single!(py, px, pw1, pw2, P) : run_batch!(py, px, pw1, pw2, P, S)

@noinline function run_batch!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T}, S::Int) where {T}
  K = length(P.batch)
  if K == 0
    py == px || unsafe_copyto!(py, px, 2S)
    return nothing
  end
  linked(P) || link_calls!(P)
  pb = pointer(P.batch)
  ncs = ladder_ncomplex(T)
  nv = length(ncs)
  NR = length(RADICES)
  # base: the widest complex count that divides S
  base = 1
  for C in ncs
    if C > 1 && S % C == 0
      base = C
      break
    end
  end
  inplace = py == px
  L = P.n; s = S; sst = 1; src = px
  @inbounds for i in 1:K
    st = unsafe_load(pb, i)
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
    c = st.c + (vi - 1) * NR
    rem = s % ncs[vi]
    if i == K
      r = L
      if rem == 0
        cstage_last!(dst, src, s, c)
      else
        cstage_last_range!(dst, src, s, 0, s - rem, c)
        cstage_last_range!(dst, src, s, s - rem, s, st.c + (nv - 1) * NR)
      end
    else
      r = RADICES[(st.c - 1) % NR + 1]
      m = L ÷ r
      if rem == 0
        cstage!(dst, src, st.pt, m, s, c)
      else
        cstage_range!(dst, src, st.pt, m, s, 0, s - rem, c)
        cstage_range!(dst, src, st.pt, m, s, s - rem, s, st.c + (nv - 1) * NR)
      end
      L = m; s *= r
    end
    src = dst
    sst *= r
  end
  nothing
end

# A single line. One stage is one codelet call; several stages run the stage loop.
@inline function run_single!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T}) where {T}
  c = P.one
  if c != 0
    cstage_last!(py, px, 1, c)
    return nothing
  end
  run_calls!(py, px, pw1, pw2, P)
end

# The stage loop of a single line. The calls are read through a raw pointer, so that no reference
# is live across a stage (the chain itself is kept alive by the caller); a reference loaded from
# the chain here or passed in by the caller costs a GC frame. The codelet switches are inlined: a
# stage is one jump and one call.
@inline function stage_loop!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, pc::Ptr{StageCall{T}}, K::Int) where {T}
  inplace = py == px
  src = px
  for i in 1:K
    st = unsafe_load(pc, i)
    dst = stage_dst(py, px, pw1, pw2, i, K, inplace)
    op = st.op
    if op == OP_LAST
      @inline cstage_last!(dst, src, st.s, st.c)
    elseif op == OP_MID
      @inline cstage!(dst, src, st.pt, st.m, st.s, st.c)
    elseif op == OP_SMALL
      @inline cstage_small!(dst, src, st.pt, st.m, st.c)
    elseif op == OP_GROUP
      @inline cstage_group!(dst, src, st.pt, st.m, st.c)
    elseif op == OP_MID_TAIL
      @inline cstage_range!(dst, src, st.pt, st.m, st.s, 0, st.s - st.rem, st.c)
      @inline cstage_range!(dst, src, st.pt, st.m, st.s, st.s - st.rem, st.s, st.c1)
    else
      @inline cstage_last_range!(dst, src, st.s, 0, st.s - st.rem, st.c)
      @inline cstage_last_range!(dst, src, st.s, st.s - st.rem, st.s, st.c1)
    end
    src = dst
  end
  nothing
end

# The calls of a chain, linked.
@inline function linked_calls(P::RChain{T}) where {T}
  linked(P) || link_calls!(P)
  pointer(P.single)
end

@noinline function run_calls!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T}) where {T}
  K = length(P.single)
  if K == 0
    py == px || unsafe_copyto!(py, px, 2)
    return nothing
  end
  stage_loop!(py, px, pw1, pw2, linked_calls(P), K)
end

# `count` single lines, `bytes` apart.
@noinline function run_calls_rows!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T}, count::Int, bytes::Int) where {T}
  K = length(P.single)
  if K == 0
    for j in 0:count-1
      py == px || unsafe_copyto!(py + j * bytes, px + j * bytes, 2)
    end
    return nothing
  end
  pc = linked_calls(P)
  for j in 0:count-1
    stage_loop!(py + j * bytes, px + j * bytes, pw1, pw2, pc, K)
  end
  nothing
end

function run_lanes!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T}, Sx::Int, Sy::Int, B::Int) where {T}
  K = length(P.batch)
  if K == 0
    lane_copy!(py, px, Sy, Sx, B, 1)
    return nothing
  end
  linked(P) || link_calls!(P)
  pb = pointer(P.batch)
  ncs = ladder_ncomplex(T)
  # widest vector type whose complex count divides B, else the narrowest
  vi = length(ncs)
  @inbounds for j in 1:length(ncs)-1
    if B % ncs[j] == 0
      vi = j
      break
    end
  end
  NR = length(RADICES)
  off = (vi - 1) * NR
  if K == 1    # one stage loads all inputs of a butterfly before it stores: py may equal px
    cstage_last_lanes!(py, px, 1, Sx, Sy, B, unsafe_load(pb).c + off)
    return nothing
  end
  L = P.n; s0 = 1; src = px; srcS = Sx
  for i in 1:K
    st = unsafe_load(pb, i)
    c = st.c + off
    if i == K
      cstage_last_lanes!(py, src, s0, srcS, Sy, B, c)
    else
      dst = isodd(i) ? pw1 : pw2
      r = @inbounds RADICES[(st.c - 1) % NR + 1]
      m = L ÷ r
      cstage_lanes!(dst, src, st.pt, m, s0, srcS, B, B, c)
      L = m; s0 *= r
      src = dst; srcS = B
    end
  end
  nothing
end

# One single line per row.
function run_rows!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::RChain{T}, count::Int, rowstride::Int) where {T}
  bytes = rowstride * 2 * sizeof(T)
  c = P.one
  if c != 0
    for j in 0:count-1
      @inline cstage_last!(py + j * bytes, px + j * bytes, 1, c)
    end
  else
    run_calls_rows!(py, px, pw1, pw2, P, count, bytes)
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
