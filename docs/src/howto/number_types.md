# Number types

`Float32` and `Float64`, real or complex, take the fast path: SIMD codelets, threads, plans with
cached kernels. Other floating-point types take a generic path. It is slower, single-threaded
and has no SIMD, but it computes in the precision of the type.

```@example types
using FFTL, LinearAlgebra

x = big.(randn(ComplexF64, 24))
X = fft(x)
eltype(X)
```

```@example types
maximum(abs, ifft(X) - x) < big"1e-60"
```

The generic path covers the complex transforms, the real transforms (`rfft`, `irfft`,
`brfft`) and the real-to-real kinds, for any length and any number of dimensions. Tests check the round trip
for `BigFloat` to a few units in the last place of the type.

| input | computed as |
|:--|:--|
| `Float32`, `Float64`, `ComplexF32`, `ComplexF64` | fast path, the output has the same precision |
| `BigFloat`, `Double64` (DoubleFloats.jl), `Float128` (Quadmath.jl) and other `AbstractFloat`s | generic path, the output has the same precision |
| `Float16` | converted to `Float64` |
| integers, `Rational`s | converted to `Float64` (or `ComplexF64`) |

A `Vector{Int}` is therefore transformed as `ComplexF64`. To compute it in `BigFloat`, convert
the input first: `fft(big.(v))`.
