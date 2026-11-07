# Stockham stages. A stage of radix R with stride s and m = L/R (L the current length, s·L = n):
#   a_j = x[q + s(p + jm)],   y[q + s(Rp + k)] = (DFT_R a)_k · w^(kp),   w = exp(σ·2πi/L),
# for p = 0…m-1, q = 0…s-1. Twiddle tables are stored per stage, either indexed (R-1)p + k
# (broadcast over q) or, for a grouped first stage (2 or 4 complex values per vector), (k-1)m + p
# (contiguous in p).
# Kernels take raw pointers (to the real part of the first complex value); callers keep the
# arrays alive. Indices are zero-based offsets in units of complex values.
#
# A transform along a dimension with stride S (batch of S interleaved transforms) is the same
# chain started with s = S: the S lanes become part of the contiguous index q.

@inline cptr(x::Array{Complex{T}}) where {T} = Ptr{T}(pointer(x))

# General stage, vectorised over q: s must be a multiple of the complex lanes of VT.
stage!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, s::Int, ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT} =
  stage!(py, px, pt, m, s, 0, m, Val(R), Val(σ), VT)

function stage_body(::Type{T}, R::Int, σ::Int, ::Type{VT}) where {T,VT}
  stmts, a, X = codelet(R, σ)
  C = ncomplex(VT); B = 2sizeof(T)
  loads = [:($(a[j]) = vload($VT, px + (q + s * (p + $(j - 1) * m)) * $B)) for j in 1:R]
  twl = [:($(Symbol("w", k)) = cbroadcast($VT, pt + ($(R - 1) * p + $(k - 1)) * $(2sizeof(T)))) for k in 1:R-1]
  stores = Any[:(vstore($(X[1]), py + o * $B))]
  for k in 1:R-1
    push!(stores, :(vstore(ccmul($(X[k+1]), $(Symbol("w", k))), py + (o + $k * s) * $B)))
  end
  quote
    for p in plo:phi-1
      $(twl...)
      for q in 0:$C:s-1
        $(loads...)
        $(stmts...)
        o = q + s * $R * p
        $(stores...)
      end
    end
    nothing
  end
end

@generated function stage!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, s::Int, plo::Int, phi::Int,
                           ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  stage_body(T, R, σ, VT)
end

# Stage for strides that are not a multiple of the complex lanes of VT (odd batch sizes): the
# first s - s % C lanes in vectors of C complex values, the scalar remainder one complex value
# at a time.
@inline function stage_split!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, s::Int,
                              ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  C = ncomplex(VT)
  rem = s % C
  if rem == 0
    stage!(py, px, pt, m, s, Val(R), Val(σ), VT)
  else
    stage_range!(py, px, pt, m, s, 0, s - rem, 0, m, Val(R), Val(σ), VT)
    stage_range!(py, px, pt, m, s, s - rem, s, 0, m, Val(R), Val(σ), Vec{2,T})
  end
end

# Stage over a q range: the kernel behind the split of a single line, whose q extent is known
# at generation time so the vectorised range and the scalar tail are emitted directly.
function stage_range_body(::Type{T}, R::Int, σ::Int, ::Type{VT}) where {T,VT}
  stmts, a, X = codelet(R, σ)
  C = ncomplex(VT); B = 2sizeof(T)
  loads = [:($(a[j]) = vload($VT, px + (q + s * (p + $(j - 1) * m)) * $B)) for j in 1:R]
  twl = [:($(Symbol("w", k)) = cbroadcast($VT, pt + ($(R - 1) * p + $(k - 1)) * $(2sizeof(T)))) for k in 1:R-1]
  stores = Any[:(vstore($(X[1]), py + o * $B))]
  for k in 1:R-1
    push!(stores, :(vstore(ccmul($(X[k+1]), $(Symbol("w", k))), py + (o + $k * s) * $B)))
  end
  quote
    for p in plo:phi-1
      $(twl...)
      for q in qlo:$C:qhi-1
        $(loads...)
        $(stmts...)
        o = q + s * $R * p
        $(stores...)
      end
    end
    nothing
  end
end

@generated function stage_range!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, s::Int, qlo::Int, qhi::Int,
                                 plo::Int, phi::Int, ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  stage_range_body(T, R, σ, VT)
end

# Last stage (m = 1): all twiddles are 1.
stage_last!(py::Ptr{T}, px::Ptr{T}, s::Int, ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT} =
  stage_last!(py, px, s, 0, s, Val(R), Val(σ), VT)

function stage_last_body(::Type{T}, R::Int, σ::Int, ::Type{VT}) where {T,VT}
  stmts, a, X = codelet(R, σ)
  C = ncomplex(VT); B = 2sizeof(T)
  loads = [:($(a[j]) = vload($VT, px + (q + s * $(j - 1)) * $B)) for j in 1:R]
  stores = [:(vstore($(X[k+1]), py + (q + $k * s) * $B)) for k in 0:R-1]
  quote
    for q in qlo:$C:qhi-1
      $(loads...)
      $(stmts...)
      $(stores...)
    end
    nothing
  end
end

@generated function stage_last!(py::Ptr{T}, px::Ptr{T}, s::Int, qlo::Int, qhi::Int,
                                ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  stage_last_body(T, R, σ, VT)
end

# Last stage for strides that are not a multiple of the complex lanes of VT (odd batch sizes):
# the first s - s % C lanes in vectors, the scalar remainder one complex value at a time.
@inline function stage_last_split!(py::Ptr{T}, px::Ptr{T}, s::Int,
                                   ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  C = ncomplex(VT)
  rem = s % C
  if rem == 0
    stage_last!(py, px, s, Val(R), Val(σ), VT)
  else
    stage_last!(py, px, s, 0, s - rem, Val(R), Val(σ), VT)
    stage_last!(py, px, s, s - rem, s, Val(R), Val(σ), Vec{2,T})
  end
end

# First stage with s = 1 for vectors of C = 2 or 4 complex values (m a multiple of C):
# butterflies p, …, p+C-1 share a register. Groups of C outputs X_k…X_(k+C-1) are transposed
# (complex value l of each output into one vector) so that each store writes C consecutive
# outputs of one butterfly; remaining outputs (R not a multiple of C) are stored one complex
# value at a time. Butterflies beyond a multiple of C (m % C) run one complex value at a time.
stage_first_group!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT} =
  stage_first_group!(py, px, pt, m, 0, m, Val(R), Val(σ), VT)

@inline function stage_first_group!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, plo::Int, phi::Int,
                                    ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  C = ncomplex(VT)
  rem = (phi - plo) % C
  if rem == 0
    stage_first_group_range!(py, px, pt, m, plo, phi, Val(R), Val(σ), VT)
  else
    stage_first_group_range!(py, px, pt, m, plo, phi - rem, Val(R), Val(σ), VT)
    stage_first_group_1!(py, px, pt, m, phi - rem, phi, Val(R), Val(σ), Vec{2,T})
  end
end

function stage_first_group_range_body(::Type{T}, R::Int, σ::Int, ::Type{VT}) where {T,VT}
  C = ncomplex(VT)
  C in (2, 4) || error("the grouped first stage needs 2 or 4 complex values per vector")
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
      z = C == 2 ? pick(outs[k+1], outs[k+2], l) :
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

@generated function stage_first_group_range!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, plo::Int, phi::Int,
                                             ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  stage_first_group_range_body(T, R, σ, VT)
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

# Outputs X_k (lanes p·s + q) in the order of y: vector t, lane l holds position tC + l.
function chunk_interleave!(ex, ys, s, C)
  R = length(ys)
  out, labs = ilv_net!(ex, ys, [[(k, l) for l in 0:C-1] for k in 0:R-1], s, C)
  for t in 0:R-1, l in 0:C-1
    pos = t * C + l
    labs[t+1][l+1] == ((pos % (s * R)) ÷ s, (pos ÷ (s * R)) * s + pos % s) ||
      error("chunk interleave of $R vectors at stride $s, $C complex values is wrong")
  end
  out
end

# Stage of a single line with stride s < C (C complex values per vector, s | C, R a power of
# two): vectorised over f = q + s·p, so each input is one contiguous load and the R outputs are
# interleaved in registers into contiguous stores. Twiddles per lane: w^(kp) at (k-1)ms + f,
# p = f ÷ s (the grouped first-stage table for s = 1). ms = m·s must be a multiple of C; s is a
# constant because the interleave depends on it.
function stage_small_body(::Type{T}, R::Int, σ::Int, ::Type{VT}, s::Int) where {T,VT}
  stmts, a, X = codelet(R, σ)
  C = ncomplex(VT); B = 2sizeof(T)
  loads = [:($(a[j]) = vload($VT, px + (f + $(j - 1) * ms) * $B)) for j in 1:R]
  outs = Any[X[1]]
  twm = Any[]
  for k in 1:R-1
    b = Symbol("b", k)
    push!(twm, :($b = ccmul($(X[k+1]), vload($VT, pt + (f + $(k - 1) * ms) * $B))))
    push!(outs, b)
  end
  shuf = Any[]
  ys = chunk_interleave!(shuf, outs, s, C)
  stores = [:(vstore($(ys[t+1]), py + ($R * f + $(t * C)) * $B)) for t in 0:R-1]
  quote
    for f in 0:$C:ms-1
      $(loads...)
      $(stmts...)
      $(twm...)
      $(shuf...)
      $(stores...)
    end
    nothing
  end
end

@generated function stage_small!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, ms::Int, ::Val{R}, ::Val{σ},
                                 ::Type{VT}, ::Val{s}) where {T,R,σ,VT,s}
  stage_small_body(T, R, σ, VT, s)
end

# Scalar tail of the grouped first stage: one complex value per vector, the twiddle table in
# the grouped layout (k-1)m + p, one store per output.
function stage_first_group_1_body(::Type{T}, R::Int, σ::Int, ::Type{VT}) where {T,VT}
  ncomplex(VT) == 1 || error("the scalar tail of the grouped first stage needs one complex value per vector")
  stmts, a, X = codelet(R, σ)
  CB = 2sizeof(T)
  loads = [:($(a[j]) = vload($VT, px + (p + $(j - 1) * m) * $CB)) for j in 1:R]
  twm = Any[]
  stores = Any[:(vstore($(X[1]), py + $R * p * $CB))]
  for k in 1:R-1
    b = Symbol("b", k)
    push!(twm, :($b = ccmul($(X[k+1]), vload($VT, pt + ($(k - 1) * m + p) * $CB))))
    push!(stores, :(vstore($b, py + ($R * p + $k) * $CB)))
  end
  quote
    for p in plo:phi-1
      $(loads...)
      $(stmts...)
      $(twm...)
      $(stores...)
    end
    nothing
  end
end

@generated function stage_first_group_1!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, plo::Int, phi::Int,
                                         ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  stage_first_group_1_body(T, R, σ, VT)
end

# Stages on a block of B lanes (cache blocking of a dimension pass, and its unit of work among
# threads): element j of lane i
# is at i + Sx·j in the input and at i + Sy·j in the output, so a block can be read from the array
# (Sx = S) into a compact buffer (Sy = B) and back. The index q of the formulas above is
# i + S·q0 with i = 0…B-1, q0 = 0…s0-1. B must be a multiple of the complex lanes of VT.
stage_lanes!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, s0::Int, Sx::Int, Sy::Int, B::Int,
             ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT} =
  stage_lanes!(py, px, pt, m, s0, Sx, Sy, B, 0, m, Val(R), Val(σ), VT)

function stage_lanes_body(::Type{T}, R::Int, σ::Int, ::Type{VT}) where {T,VT}
  stmts, a, X = codelet(R, σ)
  C = ncomplex(VT); Bt = 2sizeof(T)
  loads = [:($(a[j]) = vload($VT, px + (i + Sx * (q0 + s0 * (p + $(j - 1) * m))) * $Bt)) for j in 1:R]
  twl = [:($(Symbol("w", k)) = cbroadcast($VT, pt + ($(R - 1) * p + $(k - 1)) * $(2sizeof(T)))) for k in 1:R-1]
  stores = Any[:(vstore($(X[1]), py + (i + Sy * o) * $Bt))]
  for k in 1:R-1
    push!(stores, :(vstore(ccmul($(X[k+1]), $(Symbol("w", k))), py + (i + Sy * (o + $k * s0)) * $Bt)))
  end
  quote
    for p in plo:phi-1
      $(twl...)
      for q0 in 0:s0-1
        o = q0 + s0 * $R * p
        for i in 0:$C:B-1
          $(loads...)
          $(stmts...)
          $(stores...)
        end
      end
    end
    nothing
  end
end

@generated function stage_lanes!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, s0::Int, Sx::Int, Sy::Int,
                                 B::Int, plo::Int, phi::Int, ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  stage_lanes_body(T, R, σ, VT)
end

stage_last_lanes!(py::Ptr{T}, px::Ptr{T}, s0::Int, Sx::Int, Sy::Int, B::Int,
                  ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT} =
  stage_last_lanes!(py, px, s0, Sx, Sy, B, 0, s0, Val(R), Val(σ), VT)

function stage_last_lanes_body(::Type{T}, R::Int, σ::Int, ::Type{VT}) where {T,VT}
  stmts, a, X = codelet(R, σ)
  C = ncomplex(VT); Bt = 2sizeof(T)
  loads = [:($(a[j]) = vload($VT, px + (i + Sx * (q0 + s0 * $(j - 1))) * $Bt)) for j in 1:R]
  stores = [:(vstore($(X[k+1]), py + (i + Sy * (q0 + s0 * $k)) * $Bt)) for k in 0:R-1]
  quote
    for q0 in qlo:qhi-1
      for i in 0:$C:B-1
        $(loads...)
        $(stmts...)
        $(stores...)
      end
    end
    nothing
  end
end

@generated function stage_last_lanes!(py::Ptr{T}, px::Ptr{T}, s0::Int, Sx::Int, Sy::Int, B::Int, qlo::Int, qhi::Int,
                                      ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  stage_last_lanes_body(T, R, σ, VT)
end
