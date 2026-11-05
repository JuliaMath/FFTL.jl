# Transforms along dimensions

Every function takes a `region`: the dimensions to transform, as an integer or a collection of
integers. The default is all dimensions. Dimensions that are not in the region are batch
dimensions: each slice along them is an independent transform.

```@example dims
using FFTL, LinearAlgebra, Random
Random.seed!(4)

A = randn(ComplexF64, 32, 20, 6)
size(FFTL.fft(A)), size(FFTL.fft(A, 2))
```

Transforming along dimension 2 gives the same result as transforming each slice `A[:, :, k]`
along its second dimension:

```@example dims
B = FFTL.fft(A, 2)
slices = [FFTL.fft(A[:, :, k], 2) for k in 1:size(A, 3)]
all(B[:, :, k] ≈ slices[k] for k in 1:size(A, 3))
```

Regions can be several dimensions. The transform is separable, so transforming along 1 and then 3
equals transforming along both at once:

```@example dims
FFTL.fft(A, (1, 3)) ≈ FFTL.fft(FFTL.fft(A, 1), 3)
```

## Plans for a region

The plan is made for one size and one region; `plan_fft(A, (1, 3))` is the plan for that
combination. Dimensions that share the length and the stride class share a kernel, so a plan for
a 3D array costs little more than one for a 1D line.

## Passes: one dimension of a strided array

`FFTL.plan_pass` transforms along one dimension of a strided view, applied to all lines at once.
It is the building block of the transforms above and what NFFT.jl uses on its oversampled grid.
Two optional declarations let it skip work:

- `nonzero` says that each line is zero outside the given ranges of indices along `d`. The first
  stage then does not read the zeros.
- `needed` says that only the given output indices are used. The last stage computes only those.

The result on the declared outputs is identical to the full transform. Here, the input is zero
outside its first 16 of 64 entries, and only the first 8 outputs are needed:

```@example dims
n = 64
V = zeros(ComplexF64, n, 10)
V[1:16, :] .= randn(ComplexF64, 16, 10)

full = FFTL.fft(V, 1)
pass = FFTL.plan_pass(V, 1; nonzero = (1:16,), needed = (1:8,))
W = similar(V)
mul!(W, pass, V)
maximum(abs, W[1:8, :] - full[1:8, :])
```

Passes are single-threaded by default. A caller that already threads over many arrays keeps
`threads = 1`; see [Threads](@ref "Threads").

## Passes for real-to-real kinds

`FFTL.plan_r2r_pass(v, d, kind)` is the same for one real-to-real kind along one dimension (see
[Cosine, sine and Hartley transforms](@ref "Cosine, sine and Hartley transforms")).
