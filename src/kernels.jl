# Stockham stages. A stage of radix R with stride s and m = L/R (L the current length, s·L = n):
#   a_j = x[q + s(p + jm)],   y[q + s(Rp + k)] = (DFT_R a)_k · w^(kp),   w = exp(σ·2πi/L),
# for p = 0…m-1, q = 0…s-1. Twiddle tables are stored per stage, either indexed (R-1)p + k
# (broadcast over q) or, for a grouped first stage (2 or 4 complex values per vector), (k-1)m + p
# (contiguous in p).
# Kernels take raw pointers (to the real part of the first complex value); callers keep the
# arrays alive. Indices are zero-based offsets in units of complex values.
# This file builds the bodies of the stage kernels; kernel_methods.jl defines one method per
# (radix, sign, vector type) from them, with the arguments listed in STAGE_KERNELS.
#
# A transform along a dimension with stride S (batch of S interleaved transforms) is the same
# chain started with s = S: the S lanes become part of the contiguous index q.

@inline cptr(x::Array{Complex{T}}) where {T} = Ptr{T}(pointer(x))

# The statements of one butterfly of radix R in a stage body: the loads of the inputs from the
# addresses `ld(j)`, j = 0…R-1, the DFT, the statements `pre`, and the stores `st(X_k, k)`.
function butterfly(::Type{T}, R::Int, σ::Int, ::Type{VT}, ld, st, pre = ()) where {T,VT}
  stmts, a, X = codelet(R, σ)
  vcat([:($(a[j]) = vload($VT, $(ld(j - 1)))) for j in 1:R], stmts, collect(Any, pre), [st(X[k+1], k) for k in 0:R-1])
end

# General stage, vectorised over q: s must be a multiple of the complex lanes of VT.
# The body of stage! (q over 0:s-1) and of stage_range! (q over qlo:qhi-1).
function stage_body(::Type{T}, R::Int, σ::Int, ::Type{VT}, qlo = 0, qhi = :s) where {T,VT}
  C = ncomplex(VT); B = 2sizeof(T)
  st(X, k) = k == 0 ? :(vstore($X, yb0)) : :(vstore(ccmul_at($X, $(Symbol("w", k))), $(stream_addr(:yb, :yr, k))))
  twl = [:($(Symbol("w", k)) = pt + ($(R - 1) * p + $(k - 1)) * $(2sizeof(T))) for k in 1:R-1]
  quote
    $(stream_strides(:xr, R, :(s * m * $B))...)
    $(stream_strides(:yr, R, :(s * $B))...)
    for p in plo:phi-1
      $(twl...)
      for q in $qlo:$C:$qhi-1
        $(stream_bases(:xb, :xr, R, :(px + (q + s * p) * $B))...)
        $(butterfly(T, R, σ, VT, j -> stream_addr(:xb, :xr, j), st,
                    stream_bases(:yb, :yr, R, :(py + (q + s * $R * p) * $B)))...)
      end
    end
    nothing
  end
end

# base + i·2^e as one scaled-index address computation (a GEP over an element of 2^e bytes), so
# that the scaling stays in the addressing mode instead of becoming a value of its own.
for (e, ty) in enumerate(("i8", "i16", "i32", "i64"))
  ir = "define ptr @entry(ptr %0, i64 %1) #0 {\ntop:\n  %p = getelementptr $ty, ptr %0, i64 %1\n  ret ptr %p\n}\nattributes #0 = { alwaysinline }"
  @eval @inline scaled_ptr(b::Ptr{T}, i::Int, ::Val{$(e - 1)}) where {T} = Base.llvmcall(($ir, "entry"), Ptr{T}, Tuple{Ptr{T},Int}, b, i)
end

# The pointer p, opaque to the loop optimisations: an induction pointer passed through here is
# not rewritten into one pointer per stream.
@inline opaque_ptr(p::Ptr{T}) where {T} = Base.llvmcall(
  ("declare ptr @llvm.launder.invariant.group.p0(ptr)\ndefine ptr @entry(ptr %0) #0 {\ntop:\n  %p = call ptr @llvm.launder.invariant.group.p0(ptr %0)\n  ret ptr %p\n}\nattributes #0 = { alwaysinline }", "entry"),
  Ptr{T}, Tuple{Ptr{T}}, p)

# Address of stream j (zero-based) of streams r bytes apart: base + odd·r·2^e with odd in
# (1, 3, 5, 7) and one base per 8 streams, so that a few registers reach every stream.
function stream_addr(base, r, j)
  b = Symbol(base, j ÷ 8); k = j % 8
  k == 0 && return b
  e = trailing_zeros(k)
  :(scaled_ptr($b, $(Symbol(r, k >> e)), Val($e)))
end
stream_bases(base, r, R, first) =
  [i == 0 ? :($(Symbol(base, 0)) = opaque_ptr($first)) : :($(Symbol(base, i)) = scaled_ptr($(Symbol(base, i - 1)), $(Symbol(r, 1)), Val(3))) for i in 0:(R-1)÷8]
stream_strides(r, R, unit) =
  vcat(:($(Symbol(r, 1)) = $unit), [:($(Symbol(r, o)) = $o * $(Symbol(r, 1))) for o in (3, 5, 7) if o < R])

# Stage over a q range: a stride that is not a multiple of the complex lanes of VT runs the
# first s - s % C lanes in vectors of C complex values and the remainder one complex value at a
# time.
stage_range_body(::Type{T}, R::Int, σ::Int, ::Type{VT}) where {T,VT} = stage_body(T, R, σ, VT, :qlo, :qhi)

# Last stage (m = 1): all twiddles are 1.
function stage_last_body(::Type{T}, R::Int, σ::Int, ::Type{VT}) where {T,VT}
  C = ncomplex(VT); B = 2sizeof(T)
  quote
    $(stream_strides(:r, R, :(s * $B))...)
    for q in qlo:$C:qhi-1
      $(stream_bases(:xb, :r, R, :(px + q * $B))...)
      $(butterfly(T, R, σ, VT, j -> stream_addr(:xb, :r, j), (X, k) -> :(vstore($X, $(stream_addr(:yb, :r, k)))),
                  stream_bases(:yb, :r, R, :(py + q * $B)))...)
    end
    nothing
  end
end

# First stage with s = 1 for vectors of C = 2 or 4 complex values:
# butterflies p, …, p+C-1 share a register. Groups of C outputs X_k…X_(k+C-1) are transposed
# (complex value l of each output into one vector) so that each store writes C consecutive
# outputs of one butterfly; remaining outputs (R not a multiple of C) are stored one complex
# value at a time. Butterflies beyond a multiple of C (m % C) run one complex value at a time, with
# the same kernel on a vector of one complex value.
@inline function stage_first_group!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int,
                                    ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  rem = m % ncomplex(VT)
  stage_first_group_range!(py, px, pt, m, 0, m - rem, Val(R), Val(σ), VT)
  rem == 0 || stage_first_group_range!(py, px, pt, m, m - rem, m, Val(R), Val(σ), Vec{2,T})
  nothing
end

function stage_first_group_range_body(::Type{T}, R::Int, σ::Int, ::Type{VT}) where {T,VT}
  C = ncomplex(VT)
  C in (1, 2, 4) || error("the grouped first stage needs 1, 2 or 4 complex values per vector")
  stmts, a, X = codelet(R, σ)
  N = 2C; CB = 2sizeof(T)
  loads = [:($(a[j]) = vload($VT, px + (p + $(j - 1) * m) * $CB)) for j in 1:R]
  outs = Any[X[1]]
  twm = Any[]
  for k in 1:R-1
    b = Symbol("b", k)
    push!(twm, :($b = ccmul($(X[k+1]), vload($VT, pt + ($(k - 1) * m + p) * $CB))))
    push!(outs, b)
  end
  # complex value l of vectors u and v as one vector of two complex values
  pick(u, v, l) = :(shufflevector($u, $v, Val(($(2l), $(2l + 1), $(N + 2l), $(N + 2l + 1)))))
  st = Any[]
  k = 0
  while k + C <= R
    for l in 0:C-1
      z = C == 1 ? outs[k+1] : C == 2 ? pick(outs[k+1], outs[k+2], l) :
          :(shufflevector($(pick(outs[k+1], outs[k+2], l)), $(pick(outs[k+3], outs[k+4], l)),
                          Val((0, 1, 2, 3, 4, 5, 6, 7))))
      push!(st, :(vstore($z, py + ($R * (p + $l) + $k) * $CB)))
    end
    k += C
  end
  while k < R
    for l in 0:C-1
      push!(st, :(vstore(shufflevector($(outs[k+1]), Val(($(2l), $(2l + 1)))), py + ($R * (p + $l) + $k) * $CB)))
    end
    k += 1
  end
  quote
    for p in plo:$C:phi-1
      $(loads...)
      $(stmts...)
      $(twm...)
      $(st...)
    end
    nothing
  end
end

# Complex values of u ++ v (2C) in chunks of S, alternating u_0 v_0 u_1 v_1 …: the lane masks of
# the first C and the last C.
function chunk_masks(C, S)
  idx = [(isodd(i ÷ S) ? C : 0) + (i ÷ S ÷ 2) * S + i % S for i in 0:2C-1]
  lanes(ix) = Tuple(l for c in ix for l in (2c, 2c + 1))
  lanes(idx[1:C]), lanes(idx[C+1:2C])
end

# Interleave R = 2^r vectors in chunks of S complex values: pairs at S, then the lower and upper
# halves recursively at 2S; chunks of a whole vector are already in place. labs tracks
# (vector, complex value) of every complex value so the generator can check the permutation.
function ilv_net!(ex, ys, labs, S, C)
  (length(ys) == 1 || S >= C) && return ys, labs
  ml, mh = chunk_masks(C, S)
  lo = Any[]; hi = Any[]; llo = Any[]; lhi = Any[]
  for i in 1:2:length(ys)
    cat = vcat(labs[i], labs[i+1])
    push!(llo, cat[collect(ml[1:2:end]) .÷ 2 .+ 1]); push!(lhi, cat[collect(mh[1:2:end]) .÷ 2 .+ 1])
    a = gensym(:lo); b = gensym(:hi)
    push!(ex, :($a = shufflevector($(ys[i]), $(ys[i+1]), Val($ml))),
              :($b = shufflevector($(ys[i]), $(ys[i+1]), Val($mh))))
    push!(lo, a); push!(hi, b)
  end
  o1, l1 = ilv_net!(ex, lo, llo, 2S, C)
  o2, l2 = ilv_net!(ex, hi, lhi, 2S, C)
  vcat(o1, o2), vcat(l1, l2)
end

# Outputs X_k (lane p) in the order of y: vector t, lane l holds position tC + l = Rp + k.
function chunk_interleave!(ex, ys, C)
  R = length(ys)
  out, labs = ilv_net!(ex, ys, [[(k, l) for l in 0:C-1] for k in 0:R-1], 1, C)
  for t in 0:R-1, l in 0:C-1
    labs[t+1][l+1] == ((t * C + l) % R, (t * C + l) ÷ R) ||
      error("interleave of $R vectors of $C complex values is wrong")
  end
  out
end

# First stage of a single line (s = 1) at any vector width, for R a power of two and m a
# multiple of C: butterflies p, …, p+C-1 share a register (twiddles in the grouped first-stage
# table) and the R outputs are interleaved in registers into contiguous stores.
function stage_small_body(::Type{T}, R::Int, σ::Int, ::Type{VT}) where {T,VT}
  stmts, a, X = codelet(R, σ)
  C = ncomplex(VT); B = 2sizeof(T)
  loads = [:($(a[j]) = vload($VT, px + (p + $(j - 1) * m) * $B)) for j in 1:R]
  outs = Any[X[1]]
  twm = Any[]
  for k in 1:R-1
    b = Symbol("b", k)
    push!(twm, :($b = ccmul($(X[k+1]), vload($VT, pt + (p + $(k - 1) * m) * $B))))
    push!(outs, b)
  end
  shuf = Any[]
  ys = chunk_interleave!(shuf, outs, C)
  stores = [:(vstore($(ys[t+1]), py + ($R * p + $(t * C)) * $B)) for t in 0:R-1]
  quote
    for p in 0:$C:m-1
      $(loads...)
      $(stmts...)
      $(twm...)
      $(shuf...)
      $(stores...)
    end
    nothing
  end
end

# Stages on a block of B lanes (cache blocking of a dimension pass, and its unit of work among
# threads): element j of lane i
# is at i + Sx·j in the input and at i + Sy·j in the output, so a block can be read from the array
# (Sx = S) into a compact buffer (Sy = B) and back. The index q of the formulas above is
# i + S·q0 with i = 0…B-1, q0 = 0…s0-1. B must be a multiple of the complex lanes of VT.
function stage_lanes_body(::Type{T}, R::Int, σ::Int, ::Type{VT}) where {T,VT}
  C = ncomplex(VT); Bt = 2sizeof(T)
  twl = [:($(Symbol("w", k)) = cbroadcast($VT, pt + ($(R - 1) * p + $(k - 1)) * $(2sizeof(T)))) for k in 1:R-1]
  st(X, k) = k == 0 ? :(vstore($X, py + (i + Sy * o) * $Bt)) :
                      :(vstore(ccmul($X, $(Symbol("w", k))), py + (i + Sy * (o + $k * s0)) * $Bt))
  quote
    for p in plo:phi-1
      $(twl...)
      for q0 in 0:s0-1
        o = q0 + s0 * $R * p
        for i in 0:$C:B-1
          $(butterfly(T, R, σ, VT, j -> :(px + (i + Sx * (q0 + s0 * (p + $j * m))) * $Bt), st)...)
        end
      end
    end
    nothing
  end
end

function stage_last_lanes_body(::Type{T}, R::Int, σ::Int, ::Type{VT}) where {T,VT}
  C = ncomplex(VT); Bt = 2sizeof(T)
  quote
    for q0 in qlo:qhi-1
      for i in 0:$C:B-1
        $(butterfly(T, R, σ, VT, j -> :(px + (i + Sx * (q0 + s0 * $j)) * $Bt),
                    (X, k) -> :(vstore($X, py + (i + Sy * (q0 + s0 * $k)) * $Bt)))...)
      end
    end
    nothing
  end
end

# The stage kernels: name, the arguments between (py, px) and (Val(R), Val(σ), VT), and the body.
const STAGE_KERNELS = (
  (:stage!, (:pt, :m, :s, :plo, :phi), stage_body),
  (:stage_range!, (:pt, :m, :s, :qlo, :qhi, :plo, :phi), stage_range_body),
  (:stage_last!, (:s, :qlo, :qhi), stage_last_body),
  (:stage_first_group_range!, (:pt, :m, :plo, :phi), stage_first_group_range_body),
  (:stage_small!, (:pt, :m), stage_small_body),
  (:stage_lanes!, (:pt, :m, :s0, :Sx, :Sy, :B, :plo, :phi), stage_lanes_body),
  (:stage_last_lanes!, (:s0, :Sx, :Sy, :B, :qlo, :qhi), stage_last_lanes_body))

# Whether a kernel exists for vector type i of the ladder and radix R: the looped radices have
# one kernel of their own (primestage.jl), the grouped first stage takes at most 4 complex values
# per vector, stage_small! the widest vector type and a power-of-two radix.
kernel_exists(name::Symbol, ladder, i::Int, R::Int) =
  looped(R) ? false :
  name === :stage_first_group_range! ? ncomplex(ladder[i]) <= 4 :
  name === :stage_small! ? i == 1 && ispow2(R) : true
