# # Solving Poisson's equation with the DST
#
# Find ``u`` on the unit square with ``-\Delta u = f`` and ``u = 0`` on the boundary. Finite
# differences on the ``n × n`` interior grid turn this into a linear system with the five-point
# Laplacian. That matrix is diagonalised by the sine functions, which is exactly what the
# DST-I computes. So the system is solved by three steps: transform, divide by the eigenvalues,
# transform back.
#
# The test problem has a known answer, ``u(x, y) = \sin(\pi x)\sin(2\pi y)``, so the error is
# measured, not estimated. Then ``f = 5\pi^2 u``.

using FFTL, LinearAlgebra

# The eigenvalues of the 1D second difference with Dirichlet conditions are
# 2 - 2cos(jπh) over h², and the 2D eigenvalues are their sums, Λ_jk = λ_j + λ_k.
# Dividing the DST coefficients by Λ and transforming back solves the system; the inverse
# DST-I multiplies by 2(n+1) per dimension, which is why the result is divided by (2(n+1))².

function poisson_error(n)
  h = 1 / (n + 1)
  x = [i * h for i in 1:n]
  uex = [sin(π * xi) * sin(2π * yj) for xi in x, yj in x]
  f = 5π^2 .* uex

  λ1 = [(2 - 2cos(j * π * h)) / h^2 for j in 1:n]
  Λ = [λ1[j] + λ1[k] for j in 1:n, k in 1:n]

  F = FFTL.r2r(f, (FFTL.RODFT00, FFTL.RODFT00), (1, 2))
  F ./= Λ
  u = FFTL.r2r(F, (FFTL.RODFT00, FFTL.RODFT00), (1, 2))
  u ./= (2 * (n + 1))^2
  maximum(abs, u - uex)
end

# Halving the grid spacing should divide the error by four (second order):

errs = [poisson_error(n) for n in (31, 63, 127)]
for (n, e) in zip((31, 63, 127), errs)
  println("n = ", n, ", max error ", round(e; sigdigits = 3))
end
println("ratios: ", round.(errs[1:end-1] ./ errs[2:end]; digits = 2))

# ## The scaling
#
# The DST-I applied twice multiplies by ``2(n+1)``, which is why the inverse divides by
# ``(2(n+1))^2`` in two dimensions. `FFTL.r2r` does not normalise, so this factor is part of the
# algorithm, not of the library. The [cosine, sine and Hartley how-to](@ref "Cosine, sine and Hartley transforms")
# lists the factor for each kind.
