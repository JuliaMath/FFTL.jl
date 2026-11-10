# Real data

Real input has a spectrum with Hermitian symmetry: ``X_{N-k} = \overline{X_k}``. Half of it is
redundant, and the real transforms return only the other half. That halves the work and the
memory.

```@example real
using FFTL, Random
Random.seed!(5)

x = randn(1000, 300)
X = rfft(x)
size(X)
```

The first transformed dimension is halved: ``1000 ÷ 2 + 1 = 501`` values for frequencies
``0 … 500``. The other dimensions keep their length. The complex array has the same type as a
complex transform's output.

## Going back

The inverse needs the length of the real array, because an odd and an even length can give the
same half spectrum. `irfft` takes it explicitly:

```@example real
irfft(X, 1000) ≈ x
```

```@example real
y = randn(999)
Y = rfft(y)
size(Y), irfft(Y, 999) ≈ y
```

`brfft` is the unnormalised inverse, the real counterpart of `bfft`. It differs from `irfft` by the
product of the transformed lengths, the number of elements in the region; for one dimension,
the length of the real array:

```@example real
y1 = randn(999)
Y1 = rfft(y1)
brfft(Y1, 999) ≈ 999 .* irfft(Y1, 999)
```

## Which dimension is halved

The half spectrum is taken along the first dimension of the region. To halve dimension 2 of a
matrix, put it first in the region:

```@example real
Z = randn(64, 128)
size(rfft(Z, (2, 1)))
```

## Plans

`plan_rfft`, `plan_irfft` and `plan_brfft` return plans with the same rules as the complex ones,
including `inv` for the normalised inverse. Use them when the same real size is transformed many
times; see [Plans and reuse](@ref "Plans and reuse").

Real Float32 and Float64 plans support strided inputs and outputs, including column blocks,
padded interiors and stepped views. Their shapes must match the plan; input and output strides
are independent and may change between calls. Warmed single-threaded `mul!` execution is
allocation-free. Out-of-place execution preserves the input and writes only the output view.

```@example real
using LinearAlgebra
source = randn(21, 6)
xv = view(source, 2:2:18, 2:5)
spectra = zeros(ComplexF64, 7, 8)
Xv = view(spectra, 2:6, 3:6)
p = plan_rfft(xv, 1; threads = 1)
mul!(Xv, p, xv)

result = zeros(22, 7)
yv = view(result, 3:2:19, 2:5)
q = plan_irfft(Xv, 9, 1; threads = 1)
mul!(yv, q, Xv)
yv ≈ xv
```
