# Real data

Real input has a spectrum with Hermitian symmetry: ``X_{N-k} = \overline{X_k}``. Half of it is
redundant, and the real transforms return only the other half. That halves the work and the
memory.

```@example real
using FFTL, Random
Random.seed!(5)

x = randn(1000, 300)
X = FFTL.rfft(x)
size(X)
```

The first transformed dimension is halved: ``1000 ÷ 2 + 1 = 501`` values for frequencies
``0 … 500``. The other dimensions keep their length. The complex array has the same type as a
complex transform's output.

## Going back

The inverse needs the length of the real array, because an odd and an even length can give the
same half spectrum. `irfft` takes it explicitly:

```@example real
FFTL.irfft(X, 1000) ≈ x
```

```@example real
y = randn(999)
Y = FFTL.rfft(y)
size(Y), FFTL.irfft(Y, 999) ≈ y
```

`brfft` is the unnormalised inverse, the real counterpart of `bfft`. It differs from `irfft` by the
product of the transformed lengths, which is the number of elements in the region. For one
dimension that is the length ``d``:

```@example real
y1 = randn(999)
Y1 = FFTL.rfft(y1)
FFTL.brfft(Y1, 999) ≈ 999 .* FFTL.irfft(Y1, 999)
```

## Which dimension is halved

The half spectrum is taken along the first dimension of the region. To halve dimension 2 of a
matrix, put it first in the region:

```@example real
Z = randn(64, 128)
size(FFTL.rfft(Z, (2, 1)))
```

## Plans

`plan_rfft`, `plan_irfft` and `plan_brfft` return plans with the same rules as the complex ones,
including `inv` for the normalised inverse. Use them when the same real size is transformed many
times; see [Plans and reuse](@ref "Plans and reuse").
