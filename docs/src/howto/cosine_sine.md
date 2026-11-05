# Cosine, sine and Hartley transforms

Real-to-real transforms map real data to real data. They are the transforms behind JPEG
(DCT-II), spectral Poisson solvers with Dirichlet or Neumann boundary conditions (DST-I, DCT-I),
and the Hartley transform. FFTL computes all eleven kinds, along any dimensions, each dimension
with its own kind.

## The kinds

The formulas are unnormalised. ``n`` is the length along the transformed dimension.

| constant | kind | ``y_k`` |
|:--|:--|:--|
| `REDFT00` | DCT-I | ``x_0 + (-1)^k x_{n-1} + 2\sum_{j=1}^{n-2} x_j \cos(\pi jk/(n-1))`` |
| `REDFT10` | DCT-II | ``2\sum_j x_j \cos(\pi(j+\tfrac12)k/n)`` |
| `REDFT01` | DCT-III | ``x_0 + 2\sum_{j\ge1} x_j \cos(\pi j(k+\tfrac12)/n)`` |
| `REDFT11` | DCT-IV | ``2\sum_j x_j \cos(\pi(j+\tfrac12)(k+\tfrac12)/n)`` |
| `RODFT00` | DST-I | ``2\sum_j x_j \sin(\pi(j+1)(k+1)/(n+1))`` |
| `RODFT10` | DST-II | ``2\sum_j x_j \sin(\pi(j+\tfrac12)(k+1)/n)`` |
| `RODFT01` | DST-III | ``(-1)^k x_{n-1} + 2\sum_{j\le n-2} x_j \sin(\pi(j+1)(k+\tfrac12)/n)`` |
| `RODFT11` | DST-IV | ``2\sum_j x_j \sin(\pi(j+\tfrac12)(k+\tfrac12)/n)`` |
| `DHT` | Hartley | ``\sum_j x_j (\cos(2\pi jk/n) + \sin(2\pi jk/n))`` |
| `R2HC` | real DFT, halfcomplex order | ``\operatorname{Re} F_k`` for ``k \le n/2``, then ``\operatorname{Im} F_k`` |
| `HC2R` | its inverse | ``\sum_k F_k e^{2\pi i jk/n}`` from halfcomplex input |

`R2HC` is the DFT ``F_k = \sum_j x_j e^{-2\pi i jk/n}`` of real data, stored in the same ``n``
reals as the input: the real parts for ``k \le n/2``, the imaginary parts for ``0 < k < n/2``
at position ``n - k``.

## Pairs and scale factors

The inverses are the same kind, or a partner kind, times a factor. For each kind, applying it
twice gives the input times the factor below, so the normalised inverse divides by it:

| kind | applied twice (or with its partner) gives |
|:--|:--|
| DCT-I (`REDFT00`) | ``2(n-1)`` |
| DST-I (`RODFT00`) | ``2(n+1)`` |
| DCT-II / DCT-III, DST-II / DST-III, DCT-IV, DST-IV | ``2n`` |
| DHT, `R2HC` with `HC2R` | ``n`` |

The example checks the self-inverse kinds directly:

```@example rr
using FFTL, LinearAlgebra

n = 9
x = [sin(0.7 * j) + 0.1 * j for j in 0:n-1]
for (kind, factor) in ((FFTL.REDFT00, 2(n - 1)), (FFTL.REDFT11, 2n),
                       (FFTL.RODFT00, 2(n + 1)), (FFTL.RODFT11, 2n), (FFTL.DHT, n))
  y = FFTL.r2r(FFTL.r2r(x, kind), kind) ./ factor
  println(rpad(string(kind), 10), maximum(abs, y - x) < 1e-12)
end
```

## The orthonormal DCT-II

`FFTL.dct` is the DCT-II scaled to an orthonormal map, and `FFTL.idct` its exact inverse. It is
the transform most users want for compression and for energy-preserving spectra:

```@example rr
c = randn(16)
D = FFTL.dct(c)
(norm(D) ≈ norm(c), FFTL.idct(D) ≈ c)
```

Its values are the unnormalised DCT-II divided by ``2\sqrt{n}`` for the zero frequency and by
``\sqrt{2n}`` for the others, which is the usual orthonormal scaling.

```@example rr
R = FFTL.r2r(c, FFTL.REDFT10)
D[1] ≈ R[1] / (2 * sqrt(16)), D[2] ≈ R[2] / sqrt(2 * 16)
```

## Several dimensions with different kinds

A region takes one kind per dimension. The example uses DCT-II along the rows and the real DFT
along the columns:

```@example rr
M = randn(8, 6)
Y = FFTL.r2r(M, (FFTL.REDFT10, FFTL.R2HC), (1, 2))
size(Y)
```

Real-to-real plans of the same size share the same rules as the complex ones: `plan_r2r` returns
a plan, `inv` of a plan gives its normalised inverse for the pairs above, and `plan_r2r_pass`
transforms one dimension of a strided array.
