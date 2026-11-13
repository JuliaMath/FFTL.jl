# One stage kernel for every prime radix from LOOPED_RADIX_MIN on. The radix and its table of
# cosines and sines are arguments, so a further prime costs a table and no code. With h = (R-1)/2,
# S_j = a_j + a_(R-j), D_j = a_j - a_(R-j),
#   X_0 = a_0 + Σ_j S_j,
#   X_k, X_(R-k) = A_k ∓ i·B_k,  A_k = a_0 + Σ_j S_j cos(2π jk/R),  B_k = -σ Σ_j D_j sin(2π jk/R),
# the sums of the straight-line codelet of an odd prime (codegen.jl) as loops.

const LOOPED_RADIX_MIN = 17
# RADICES holds no composite radix from LOOPED_RADIX_MIN on.
looped(R::Int) = R >= LOOPED_RADIX_MIN

# Rows of a butterfly that one pass over its inputs computes: every row takes two registers.
const PRIME_BLOCK = (REG_BYTES == 64 || Sys.ARCH === :aarch64) ? 8 : 4

# The table of radix R and sign σ: cos(2π jk/R) at (j-1)h + k for j, k = 1…h, and -σ·sin(2π jk/R)
# h² further on, so that the rows k of one input pair j are contiguous.
function build_prime_table(::Type{T}, R::Int, σ::Int) where {T}
  h = (R - 1) ÷ 2
  vcat([T(cospi(2 * mod(j * k, R) / R)) for j in 1:h for k in 1:h],
       [T(-σ * sinpi(2 * mod(j * k, R) / R)) for j in 1:h for k in 1:h])
end

for T in (Float32, Float64)
  tables = Symbol(:PRIME_TABLES_, T)
  @eval const $tables = Dict{Tuple{Int,Int},Vector{$T}}((R, σ) => build_prime_table($T, R, σ)
                                                        for R in RADIX_PRIMES if looped(R) for σ in (-1, 1))
  @eval prime_table(::Type{$T}, R::Int, σ::Int) = $tables[(R, σ)]
end

# How a stage treats its outputs: no twiddles (a last stage), one twiddle per output k for all
# lanes, or the grouped first stage (kernels.jl): the lanes of a vector are the butterflies
# p, p+1, …, each with its own twiddle, and output k of lane l goes to R·l + k.
const PRIME_LAST = 0
const PRIME_MID = 1
const PRIME_GROUP = 2

@inline function prime_store!(X::VT, yb::Ptr{T}, ys::Int, k::Int, R::Int, mode::Int) where {T,VT}
  if mode == PRIME_GROUP
    for l in 1:ncomplex(VT)
      vstore(Vec{2,T}((X[2l-1], X[2l])), yb + (R * (l - 1) + k) * 2sizeof(T))
    end
  else
    vstore(X, yb + k * ys)
  end
  nothing
end

# The outputs k and R - k from A_k and B_k. tw is the address of the twiddle of output 1, tws the
# distance to the next one in the grouped table.
@inline function prime_out!(yb::Ptr{T}, ys::Int, tw::Ptr{T}, tws::Int, mode::Int, c::VT, s::VT, k::Int, R::Int) where {T,VT}
  Bt = 2sizeof(T)
  Xk = fcadd270(c, s); Xr = fcadd90(c, s)
  if mode == PRIME_MID
    Xk = ccmul_at(Xk, tw + (k - 1) * Bt); Xr = ccmul_at(Xr, tw + (R - k - 1) * Bt)
  elseif mode == PRIME_GROUP
    Xk = ccmul(Xk, vload(VT, tw + (k - 1) * tws)); Xr = ccmul(Xr, vload(VT, tw + (R - k - 1) * tws))
  end
  prime_store!(Xk, yb, ys, k, R, mode); prime_store!(Xr, yb, ys, R - k, R, mode)
  nothing
end
# The outputs of the rows k, k+1, … of a block, one per element of the tuples. The recursion over
# the tuples keeps the accumulators in registers: a loop with an index into them, and a closure
# over them, were both measured slower (up to 1.4× for a whole transform).
@inline prime_outs!(yb, ys, tw, tws, mode, ::Tuple{}, ::Tuple{}, k, R) = nothing
@inline function prime_outs!(yb, ys, tw, tws, mode, cs::Tuple, ss::Tuple, k, R)
  prime_out!(yb, ys, tw, tws, mode, first(cs), first(ss), k, R)
  prime_outs!(yb, ys, tw, tws, mode, Base.tail(cs), Base.tail(ss), k + 1, R)
end

# acc_b + v·t_b for the NB table values at p
@inline fma_rows(v::VT, p::Ptr{T}, acc::NTuple{NB,VT}) where {T,VT,NB} =
  ntuple(b -> muladd(v, unsafe_load(p, b), acc[b]), Val(NB))

# Rows k … k+NB-1 of one butterfly (and row 0 with the first block) in one pass over the inputs
# at xb + j·xs. The sums and differences are formed again in every pass: keeping them in a buffer
# costs more than the loads (the stores of whole registers are the slowest part). Not inlined: a
# pass is long enough for a call, and an inlined function keeps its typed code in the package
# image for every block size and vector type.
@noinline function prime_rows!(yb::Ptr{T}, ys::Int, xb::Ptr{T}, xs::Int, tab::Ptr{T}, tw::Ptr{T}, tws::Int, mode::Int,
                             ::Type{VT}, R::Int, h::Int, k::Int, ::Val{NB}) where {T,VT,NB}
  a0 = vload(VT, xb); acc = a0
  cs = ntuple(_ -> a0, Val(NB)); ss = ntuple(_ -> zero(VT), Val(NB))
  psin = tab + h * h * sizeof(T)
  for j in 1:h
    x = vload(VT, xb + j * xs); y = vload(VT, xb + (R - j) * xs)
    sj = x + y
    acc += sj
    o = ((j - 1) * h + k - 1) * sizeof(T)
    cs = fma_rows(sj, tab + o, cs)
    ss = fma_rows(x - y, psin + o, ss)
  end
  k == 1 && prime_store!(acc, yb, ys, 0, R, mode)
  prime_outs!(yb, ys, tw, tws, mode, cs, ss, k, R)
end

# One butterfly in blocks of at most PRIME_BLOCK rows of nearly equal size (q or q + 1, the
# first r blocks the larger ones).
@eval @inline function prime_butterfly!(yb, ys, xb, xs, tab, tw, tws, mode, ::Type{VT}, R, h, nblocks, q, r) where {VT}
  k = 1
  for b in 1:nblocks
    nb = q + (b <= r)
    Base.Cartesian.@nif $PRIME_BLOCK d -> (nb == d) d -> prime_rows!(yb, ys, xb, xs, tab, tw, tws, mode, VT, R, h, k, Val(d))
    k += nb
  end
  nothing
end

"""
    prime_stage!(py, px, pt, tab, R, m, s0, Sx, Sy, i0, i1, VT)

A Stockham stage of the prime radix `R` with the table `tab` (`prime_table`), in the form of the
lane stages (kernels.jl): input j of butterfly p at `i + Sx·(q0 + s0·(p + j·m))`, output k at
`i + Sy·(q0 + s0·(R·p + k))`, for q0 < s0 and the lanes i0 ≤ i < i1 in vectors of `VT`. A stage
vectorised over q is the case s0 = 1, Sx = Sy = s; a last stage has m = 1 and no twiddles
(`pt` null).
"""
@noinline function prime_stage!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, tab::Ptr{T}, R::Int, m::Int, s0::Int, Sx::Int, Sy::Int,
                                i0::Int, i1::Int, ::Type{VT}) where {T,VT}
  C = ncomplex(VT); Bt = 2sizeof(T); h = (R - 1) >> 1
  nblocks = cld(h, PRIME_BLOCK); q, r = divrem(h, nblocks)
  # In place (a single stage) the inputs of a butterfly must be read before its first store: one
  # block does that, several need a copy.
  py == px && nblocks > 1 && return prime_stage_inplace!(py, tab, R, s0, Sx, i0, i1, VT)
  xs = Sx * s0 * m * Bt; ys = Sy * s0 * Bt
  mode = pt == C_NULL ? PRIME_LAST : PRIME_MID
  for p in 0:m-1
    tw = pt + (R - 1) * p * Bt
    for q0 in 0:s0-1, i in i0:C:i1-1
      xb = px + (i + Sx * (q0 + s0 * p)) * Bt
      yb = py + (i + Sy * (q0 + s0 * R * p)) * Bt
      prime_butterfly!(yb, ys, xb, xs, tab, tw, 0, mode, VT, R, h, nblocks, q, r)
    end
  end
  nothing
end

# A single stage in place (m = 1, no twiddles): each butterfly from a copy of its inputs.
@noinline function prime_stage_inplace!(py::Ptr{T}, tab::Ptr{T}, R::Int, s0::Int, S::Int, i0::Int, i1::Int,
                                        ::Type{VT}) where {T,VT}
  C = ncomplex(VT); Bt = 2sizeof(T); h = (R - 1) >> 1; W = sizeof(VT)
  ys = S * s0 * Bt
  nblocks = cld(h, PRIME_BLOCK); q, r = divrem(h, nblocks)
  buf = Ref{NTuple{MAX_RADIX_PRIME,VT}}()
  GC.@preserve buf begin
    pbuf = Ptr{T}(Base.unsafe_convert(Ptr{NTuple{MAX_RADIX_PRIME,VT}}, buf))
    for q0 in 0:s0-1, i in i0:C:i1-1
      yb = py + (i + S * q0) * Bt
      for j in 0:R-1
        vstore(vload(VT, yb + j * ys), pbuf + j * W)
      end
      prime_butterfly!(yb, ys, pbuf, W, tab, Ptr{T}(C_NULL), 0, PRIME_LAST, VT, R, h, nblocks, q, r)
    end
  end
  nothing
end

# The butterflies p0 … p1-1 of the grouped first stage of a single line (s = 1, the twiddles pt in
# the grouped layout (k-1)m + p), ncomplex(VT) of them per vector.
@noinline function prime_group_range!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, tab::Ptr{T}, R::Int, m::Int, p0::Int, p1::Int,
                                      ::Type{VT}) where {T,VT}
  Bt = 2sizeof(T); h = (R - 1) >> 1
  nblocks = cld(h, PRIME_BLOCK); q, r = divrem(h, nblocks)
  for p in p0:ncomplex(VT):p1-1
    prime_butterfly!(py + R * p * Bt, 0, px + p * Bt, m * Bt, tab, pt + p * Bt, m * Bt, PRIME_GROUP, VT, R, h, nblocks, q, r)
  end
  nothing
end

# The grouped first stage: m - m % C butterflies in vectors, the rest one complex value at a time.
@inline function prime_group!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, tab::Ptr{T}, R::Int, m::Int, ::Type{VT}) where {T,VT}
  rem = m % ncomplex(VT)
  prime_group_range!(py, px, pt, tab, R, m, 0, m - rem, VT)
  rem == 0 || prime_group_range!(py, px, pt, tab, R, m, m - rem, m, Vec{2,T})
  nothing
end
