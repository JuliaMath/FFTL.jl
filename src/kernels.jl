# Stockham stages. A stage of radix R with stride s and m = L/R (L the current length, s·L = n):
#   a_j = x[q + s(p + jm)],   y[q + s(Rp + k)] = (DFT_R a)_k · w^(kp),   w = exp(σ·2πi/L),
# for p = 0…m-1, q = 0…s-1. Twiddle tables are stored per stage, either indexed (R-1)p + k
# (broadcast over q) or, for the first stage of a Float32 plan, (k-1)m + p (contiguous in p).
# Kernels take raw pointers (to the real part of the first complex value); callers keep the
# arrays alive. Indices are zero-based offsets in units of complex values.
#
# A transform along a dimension with stride S (batch of S interleaved transforms) is the same
# chain started with s = S: the S lanes become part of the contiguous index q.

@inline cptr(x::Array{Complex{T}}) where {T} = Ptr{T}(pointer(x))

# General stage, vectorised over q: s must be a multiple of the complex lanes of VT.
@generated function stage!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T},
                           m::Int, s::Int, ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  stmts, a, X = codelet(R, σ)
  C = ncomplex(VT); B = 2sizeof(T)
  loads = [:($(a[j]) = vload(VT, px + (q + s * (p + $(j - 1) * m)) * $B)) for j in 1:R]
  twl = [:($(Symbol("w", k)) = cconst(VT, unsafe_load(pt, $(2(R - 1)) * p + $(2k - 1)),
                                       unsafe_load(pt, $(2(R - 1)) * p + $(2k)))) for k in 1:R-1]
  stores = Any[:(vstore($(X[1]), py + o * $B))]
  for k in 1:R-1
    push!(stores, :(vstore(ccmul($(X[k+1]), $(Symbol("w", k))), py + (o + $k * s) * $B)))
  end
  quote
    for p in 0:m-1
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

# Last stage (m = 1): all twiddles are 1.
@generated function stage_last!(py::Ptr{T}, px::Ptr{T}, s::Int,
                                ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  stmts, a, X = codelet(R, σ)
  C = ncomplex(VT); B = 2sizeof(T)
  loads = [:($(a[j]) = vload(VT, px + (q + s * $(j - 1)) * $B)) for j in 1:R]
  stores = [:(vstore($(X[k+1]), py + (q + $k * s) * $B)) for k in 0:R-1]
  quote
    for q in 0:$C:s-1
      $(loads...)
      $(stmts...)
      $(stores...)
    end
    nothing
  end
end

# First stage with s = 1 and m even, for vectors of two complex values (VT = Vec{4,T}):
# butterflies p and p+1 share a register. The outputs X_k, X_{k+1} of both butterflies are
# regrouped by shuffles of the complex halves so that each store writes two consecutive outputs
# of one butterfly.
@generated function stage_first_pair!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T},
                                      m::Int, ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  stmts, a, X = codelet(R, σ)
  CB = 2sizeof(T)
  loads = [:($(a[j]) = vload(VT, px + (p + $(j - 1) * m) * $CB)) for j in 1:R]
  outs = Any[X[1]]
  twm = Any[]
  for k in 1:R-1
    b = Symbol("b", k)
    push!(twm, :($b = ccmul($(X[k+1]), vload(VT, pt + ($(k - 1) * m + p) * $CB))))
    push!(outs, b)
  end
  st = Any[]
  k = 0
  while k < R
    if k + 1 < R
      lo, hi = outs[k+1], outs[k+2]
      push!(st, :(vstore(shufflevector($lo, $hi, Val((0, 1, 4, 5))), py + ($R * p + $k) * $CB)))
      push!(st, :(vstore(shufflevector($lo, $hi, Val((2, 3, 6, 7))), py + ($R * (p + 1) + $k) * $CB)))
      k += 2
    else
      o = outs[k+1]
      push!(st, :(vstore(shufflevector($o, Val((0, 1))), py + ($R * p + $k) * $CB)))
      push!(st, :(vstore(shufflevector($o, Val((2, 3))), py + ($R * (p + 1) + $k) * $CB)))
      k += 1
    end
  end
  quote
    for p in 0:2:m-1
      $(loads...)
      $(stmts...)
      $(twm...)
      $(st...)
    end
    nothing
  end
end

# Stages on a block of B lanes (used to split one dimension among threads): element j of lane i
# is at i + Sx·j in the input and at i + Sy·j in the output, so a block can be read from the array
# (Sx = S) into a compact buffer (Sy = B) and back. The index q of the formulas above is
# i + S·q0 with i = 0…B-1, q0 = 0…s0-1. B must be a multiple of the complex lanes of VT.
@generated function stage_lanes!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, s0::Int, Sx::Int, Sy::Int,
                                 B::Int, ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  stmts, a, X = codelet(R, σ)
  C = ncomplex(VT); Bt = 2sizeof(T)
  loads = [:($(a[j]) = vload(VT, px + (i + Sx * (q0 + s0 * (p + $(j - 1) * m))) * $Bt)) for j in 1:R]
  twl = [:($(Symbol("w", k)) = cconst(VT, unsafe_load(pt, $(2(R - 1)) * p + $(2k - 1)),
                                       unsafe_load(pt, $(2(R - 1)) * p + $(2k)))) for k in 1:R-1]
  stores = Any[:(vstore($(X[1]), py + (i + Sy * o) * $Bt))]
  for k in 1:R-1
    push!(stores, :(vstore(ccmul($(X[k+1]), $(Symbol("w", k))), py + (i + Sy * (o + $k * s0)) * $Bt)))
  end
  quote
    for p in 0:m-1
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

@generated function stage_last_lanes!(py::Ptr{T}, px::Ptr{T}, s0::Int, Sx::Int, Sy::Int, B::Int,
                                      ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT}
  stmts, a, X = codelet(R, σ)
  C = ncomplex(VT); Bt = 2sizeof(T)
  loads = [:($(a[j]) = vload(VT, px + (i + Sx * (q0 + s0 * $(j - 1))) * $Bt)) for j in 1:R]
  stores = [:(vstore($(X[k+1]), py + (i + Sy * (q0 + s0 * $k)) * $Bt)) for k in 0:R-1]
  quote
    for q0 in 0:s0-1
      for i in 0:$C:B-1
        $(loads...)
        $(stmts...)
        $(stores...)
      end
    end
    nothing
  end
end
