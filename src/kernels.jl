# Stockham stages. A stage of radix R with stride s and m = L/R (L the current length, s·L = n):
#   a_j = x[q + s(p + jm)],   y[q + s(Rp + k)] = (DFT_R a)_k · w^(kp),   w = exp(σ·2πi/L),
# for p = 0…m-1, q = 0…s-1. Twiddle tables are stored per stage, either indexed (R-1)p + k
# (broadcast over q) or, for the first stage of a Float32 plan, (k-1)m + p (contiguous in p).
# Kernels take raw pointers (to the real part of the first complex value); callers keep the
# arrays alive. Indices are zero-based offsets in units of complex values.
#
# A transform along a dimension with stride S (batch of S interleaved transforms) is the same
# chain started with s = S: the S lanes become part of the contiguous index q.
#
# Pruned stages (zero inputs, unneeded outputs of a line): first stages take a range plo:phi-1 of
# p and a bit mask `zin` of inputs j that are zero there (not loaded, not computed with); last
# stages take a range of q (or q0) and a bit mask `need` of the outputs k to compute and store.
# The unpruned methods cover the full range with empty masks.

zero_input(zin, j) = (zin >> (j - 1)) & 1 == 1

@inline cptr(x::Array{Complex{T}}) where {T} = Ptr{T}(pointer(x))

# General stage, vectorised over q: s must be a multiple of the complex lanes of VT.
stage!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, s::Int, ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT} =
  stage!(py, px, pt, m, s, 0, m, Val(R), Val(σ), VT, Val(0))

@generated function stage!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, s::Int, plo::Int, phi::Int,
                           ::Val{R}, ::Val{σ}, ::Type{VT}, ::Val{zin}) where {T,R,σ,VT,zin}
  stmts, a, X = codelet(R, σ, zin)
  C = ncomplex(VT); B = 2sizeof(T)
  loads = [:($(a[j]) = vload(VT, px + (q + s * (p + $(j - 1) * m)) * $B)) for j in 1:R if !zero_input(zin, j)]
  twl = [:($(Symbol("w", k)) = cbroadcast(VT, pt + ($(R - 1) * p + $(k - 1)) * $(2sizeof(T)))) for k in 1:R-1]
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

# Last stage (m = 1): all twiddles are 1.
stage_last!(py::Ptr{T}, px::Ptr{T}, s::Int, ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT} =
  stage_last!(py, px, s, 0, s, Val(R), Val(σ), VT, Val((1 << R) - 1))

@generated function stage_last!(py::Ptr{T}, px::Ptr{T}, s::Int, qlo::Int, qhi::Int,
                                ::Val{R}, ::Val{σ}, ::Type{VT}, ::Val{need}) where {T,R,σ,VT,need}
  stmts, a, X = codelet(R, σ, 0, need)
  C = ncomplex(VT); B = 2sizeof(T)
  loads = [:($(a[j]) = vload(VT, px + (q + s * $(j - 1)) * $B)) for j in 1:R]
  stores = [:(vstore($(X[k+1]), py + (q + $k * s) * $B)) for k in 0:R-1 if X[k+1] !== nothing]
  quote
    for q in qlo:$C:qhi-1
      $(loads...)
      $(stmts...)
      $(stores...)
    end
    nothing
  end
end

# First stage with s = 1 for vectors of C = 2 or 4 complex values (m a multiple of C):
# butterflies p, …, p+C-1 share a register. Groups of C outputs X_k…X_(k+C-1) are transposed
# (complex value l of each output into one vector) so that each store writes C consecutive
# outputs of one butterfly; remaining outputs (R not a multiple of C) are stored one complex
# value at a time.
stage_first_group!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT} =
  stage_first_group!(py, px, pt, m, 0, m, Val(R), Val(σ), VT, Val(0))

@generated function stage_first_group!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, plo::Int, phi::Int,
                                       ::Val{R}, ::Val{σ}, ::Type{VT}, ::Val{zin}) where {T,R,σ,VT,zin}
  stmts, a, X = codelet(R, σ, zin)
  C = ncomplex(VT); N = 2C; CB = 2sizeof(T)
  loads = [:($(a[j]) = vload(VT, px + (p + $(j - 1) * m) * $CB)) for j in 1:R if !zero_input(zin, j)]
  outs = Any[X[1]]
  twm = Any[]
  for k in 1:R-1
    b = Symbol("b", k)
    push!(twm, :($b = ccmul($(X[k+1]), vload(VT, pt + ($(k - 1) * m + p) * $CB))))
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

# Stages on a block of B lanes (used to split one dimension among threads): element j of lane i
# is at i + Sx·j in the input and at i + Sy·j in the output, so a block can be read from the array
# (Sx = S) into a compact buffer (Sy = B) and back. The index q of the formulas above is
# i + S·q0 with i = 0…B-1, q0 = 0…s0-1. B must be a multiple of the complex lanes of VT.
stage_lanes!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, s0::Int, Sx::Int, Sy::Int, B::Int,
             ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT} =
  stage_lanes!(py, px, pt, m, s0, Sx, Sy, B, 0, m, Val(R), Val(σ), VT, Val(0))

@generated function stage_lanes!(py::Ptr{T}, px::Ptr{T}, pt::Ptr{T}, m::Int, s0::Int, Sx::Int, Sy::Int,
                                 B::Int, plo::Int, phi::Int, ::Val{R}, ::Val{σ}, ::Type{VT}, ::Val{zin}) where {T,R,σ,VT,zin}
  stmts, a, X = codelet(R, σ, zin)
  C = ncomplex(VT); Bt = 2sizeof(T)
  loads = [:($(a[j]) = vload(VT, px + (i + Sx * (q0 + s0 * (p + $(j - 1) * m))) * $Bt)) for j in 1:R if !zero_input(zin, j)]
  twl = [:($(Symbol("w", k)) = cbroadcast(VT, pt + ($(R - 1) * p + $(k - 1)) * $(2sizeof(T)))) for k in 1:R-1]
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

stage_last_lanes!(py::Ptr{T}, px::Ptr{T}, s0::Int, Sx::Int, Sy::Int, B::Int,
                  ::Val{R}, ::Val{σ}, ::Type{VT}) where {T,R,σ,VT} =
  stage_last_lanes!(py, px, s0, Sx, Sy, B, 0, s0, Val(R), Val(σ), VT, Val((1 << R) - 1))

@generated function stage_last_lanes!(py::Ptr{T}, px::Ptr{T}, s0::Int, Sx::Int, Sy::Int, B::Int, qlo::Int, qhi::Int,
                                      ::Val{R}, ::Val{σ}, ::Type{VT}, ::Val{need}) where {T,R,σ,VT,need}
  stmts, a, X = codelet(R, σ, 0, need)
  C = ncomplex(VT); Bt = 2sizeof(T)
  loads = [:($(a[j]) = vload(VT, px + (i + Sx * (q0 + s0 * $(j - 1))) * $Bt)) for j in 1:R]
  stores = [:(vstore($(X[k+1]), py + (i + Sy * (q0 + s0 * $k)) * $Bt)) for k in 0:R-1 if X[k+1] !== nothing]
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
