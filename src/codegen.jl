# Codelet generator: straight-line code for a DFT of size r,
#   X_k = Σ_j a_j · exp(σ·2πi·jk/r),  σ = -1 (forward) or +1 (backward),
# on complex SIMD vectors (see complexvec.jl). The generated statements refer to the vector type
# as `VT`, which the surrounding generated function binds.
#
# Terms carry a pending rotation: (sym, k) stands for sym·(-i)^k. A sum of two terms whose
# rotations differ by an odd k is one FCADD, by 2 a subtraction; rotations are materialised only
# at the output (`force`). Composite r = r1·r2 is split in registers (j = r2·j1 + j2,
# k = k1 + r1·k2):
#   X[k1 + r1 k2] = Σ_j2 W_r^(j2 k1) W_r2^(j2 k2) Σ_j1 a[r2 j1 + j2] W_r1^(j1 k1),  W_N = exp(σ·2πi/N).
# Odd primes use the symmetric form
#   X_k, X_{r-k} = A ± σ·i·B,  A = a_0 + Σ_j (a_j + a_{r-j}) cos(2πjk/r),  B = Σ_j (a_j - a_{r-j}) sin(2πjk/r).

mutable struct CodeGen
  stmts::Vector{Any}
  n::Int
  σ::Int
end
CodeGen(σ) = CodeGen(Any[], 0, σ)

function emit!(g::CodeGen, ex)
  g.n += 1
  s = Symbol("c", g.n)
  push!(g.stmts, :($s = $ex))
  s
end

rotate(t, k) = (t[1], mod(t[2] + k, 4))
# rotation units of one factor σ·i: -i = (-i)^1, +i = (-i)^3
unit(g::CodeGen) = g.σ == -1 ? 1 : 3

function plus!(g, a, b)
  d = mod(b[2] - a[2], 4)
  ex = d == 0 ? :($(a[1]) + $(b[1])) :
       d == 2 ? :($(a[1]) - $(b[1])) :
       d == 1 ? :(fcadd270($(a[1]), $(b[1]))) :      # a + (-i) b
                :(fcadd90($(a[1]), $(b[1])))         # a + i b
  (emit!(g, ex), a[2])
end
minus!(g, a, b) = plus!(g, a, rotate(b, 2))

function force!(g, t)
  t[2] == 0 && return t[1]
  t[2] == 2 && return emit!(g, :(-$(t[1])))
  t[2] == 1 && return emit!(g, :(fcadd270(zero(VT), $(t[1]))))
  emit!(g, :(fcadd90(zero(VT), $(t[1]))))
end

# t · W_N^e with constant e, N
function twiddle!(g, t, e, N)
  e = mod(e, N)
  e == 0 && return t
  (4e) % N == 0 && return rotate(t, unit(g) * (4e ÷ N))
  if (8e) % N == 0
    # W_N^e = exp(σ·iπ/4) · (σ·i)^((e8-1)/2),  exp(σ·iπ/4) = (1 + σ·i)/√2
    e8 = 8e ÷ N
    s = emit!(g, g.σ == -1 ? :(fcadd270($(t[1]), $(t[1]))) : :(fcadd90($(t[1]), $(t[1]))))
    return (emit!(g, :($(sqrt(0.5)) * $s)), mod(t[2] + unit(g) * ((e8 - 1) ÷ 2), 4))
  end
  c, s = cospi(2e / N), g.σ * sinpi(2e / N)
  (emit!(g, :(ccmul($(t[1]), cconst(VT, $c, $s)))), t[2])
end

function prime!(g, a)
  r = length(a); h = (r - 1) ÷ 2
  a = [force!(g, x) for x in a]
  sp = [emit!(g, :($(a[j+1]) + $(a[r-j+1]))) for j in 1:h]
  dm = [emit!(g, :($(a[j+1]) - $(a[r-j+1]))) for j in 1:h]
  X = Vector{Any}(undef, r)
  acc = sp[1]
  for j in 2:h
    acc = emit!(g, :($acc + $(sp[j])))
  end
  X[1] = (emit!(g, :($(a[1]) + $acc)), 0)
  for k in 1:h
    A = a[1]
    for j in 1:h
      A = emit!(g, :(muladd($(sp[j]), $(cospi(2j * k / r)), $A)))
    end
    B = emit!(g, :($(sinpi(2k / r)) * $(dm[1])))
    for j in 2:h
      B = emit!(g, :(muladd($(dm[j]), $(sinpi(2j * k / r)), $B)))
    end
    minus_iB = emit!(g, :(fcadd270($A, $B)))
    plus_iB = emit!(g, :(fcadd90($A, $B)))
    X[k+1] = (g.σ == -1 ? minus_iB : plus_iB, 0)
    X[r-k+1] = (g.σ == -1 ? plus_iB : minus_iB, 0)
  end
  X
end

first_factor(r) = r % 4 == 0 ? 4 : r % 2 == 0 ? 2 : first(p for p in 3:r if r % p == 0)

function dft!(g::CodeGen, a::Vector)
  r = length(a)
  r == 1 && return a
  if r == 2
    return [plus!(g, a[1], a[2]), minus!(g, a[1], a[2])]
  elseif r == 4
    t0 = plus!(g, a[1], a[3]); t1 = minus!(g, a[1], a[3])
    t2 = plus!(g, a[2], a[4]); t3 = rotate(minus!(g, a[2], a[4]), unit(g))
    return [plus!(g, t0, t2), plus!(g, t1, t3), minus!(g, t0, t2), minus!(g, t1, t3)]
  end
  r1 = first_factor(r)
  r1 == r && return prime!(g, a)
  r2 = r ÷ r1
  u = [dft!(g, [a[r2*j1+j2+1] for j1 in 0:r1-1]) for j2 in 0:r2-1]
  v = [[twiddle!(g, u[j2+1][k1+1], j2 * k1, r) for k1 in 0:r1-1] for j2 in 0:r2-1]
  X = Vector{Any}(undef, r)
  for k1 in 0:r1-1
    Y = dft!(g, [v[j2+1][k1+1] for j2 in 0:r2-1])
    for k2 in 0:r2-1
      X[k1+r1*k2+1] = Y[k2+1]
    end
  end
  X
end

"""
    codelet(r, σ) -> (stmts, inputs, outputs)

Statements computing a size-`r` DFT with sign `σ` from the input symbols `a1…ar` into the
returned output symbols.
"""
function codelet(r::Int, σ::Int)
  g = CodeGen(σ)
  a = [Symbol("a", j) for j in 1:r]
  X = dft!(g, [(s, 0) for s in a])
  out = [force!(g, x) for x in X]
  g.stmts, a, out
end
