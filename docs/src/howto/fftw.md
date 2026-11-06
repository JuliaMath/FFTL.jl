# Coming from FFTW.jl

`using FFTL` can take the place of `using FFTW`. FFTL exports the names FFTW.jl exports, with
the same signs, scalings and keywords: the DFT functions `fft`, `bfft`, `ifft`, `rfft`, `brfft`,
`irfft`, the orthonormal `dct` and `idct`, their `plan_` and in-place forms, and AbstractFFTs's
helpers (`fftshift`, `fftfreq`, …). As in FFTW.jl, `FFTL.r2r`, the real-to-real kinds
(`FFTL.REDFT10`, …), the flags and `FFTL.set_num_threads` are used with the prefix; the kinds
have FFTW's names and FFTW's unnormalised definitions.

## The same transforms under the same names

| FFTW.jl | FFTL | remark |
|:--|:--|:--|
| `fft`, `bfft`, `ifft`, `rfft`, `brfft`, `irfft` and their plans | the same | same signs and scalings |
| `fft!`, `plan_fft!`, … | the same | the plans overwrite their argument |
| `dct`, `idct`, `plan_dct`, `dct!`, … | the same | orthonormal DCT-II and its inverse |
| `FFTW.r2r(x, FFTW.REDFT10)` | `FFTL.r2r(x, FFTL.REDFT10)` | all eleven kinds, same definitions |
| `plan_fft(x; flags = FFTW.MEASURE, timelimit = 10)` | `plan_fft(x; flags = FFTL.MEASURE, timelimit = 10)` | accepted, without effect (see below) |
| `plan_fft(x; num_threads = 4)` | the same, or `threads = 4` | |
| `FFTW.set_num_threads(n)` | `FFTL.set_num_threads(n)` | default for the plans made afterwards |
| wisdom (`export_wisdom`, `import_wisdom`) | not needed | plans do not measure |

The comparison runs with both packages loaded, so the names are qualified; FFTW.jl is not a
dependency of FFTL, so the example is shown, not run:

```julia
using FFTW, FFTL

x = randn(12, 7)
FFTL.dct(x) ≈ FFTW.dct(x)                                     # true
FFTL.r2r(x, FFTL.RODFT11, 1) ≈ FFTW.r2r(x, FFTW.RODFT11, 1)   # true, likewise for every kind
```

The test suite runs such comparisons for every transform, plan form and kind in both
precisions.

## Planning without flags

FFTW's planner can measure candidate algorithms at run time (`MEASURE`, `PATIENT`) and keep the
result as wisdom. FFTL has no such step: the radix sequence of a length comes from a fixed rule
that was derived from offline measurements, and the codelets are compiled for exactly that
sequence. A plan therefore gives the same algorithm in every session; the `flags` and
`timelimit` keywords are accepted so that FFTW.jl code runs unchanged, and have no effect. The cost that remains is compilation: the first plan of a new radix sequence compiles its
kernels, which [Precompilation](@ref "Precompilation") moves to package installation for the
common sizes.

## FFTL and FFTW.jl in one session

`fft`, `plan_fft`, `dct`, … are FFTL's own functions. A module that writes `using FFTL` gets
FFTL, also when a package it depends on loads FFTW.jl, and that package keeps getting FFTW.
In a module that loads both, the unqualified names are ambiguous and Julia asks for the
prefix: `FFTL.fft(x)` or `FFTW.fft(x)`.

Code that calls AbstractFFTs.jl's generic functions (`AbstractFFTs.fft`, or `fft` after
`using AbstractFFTs`) also runs on FFTL. With FFTW.jl loaded, those generic functions run on
FFTW for `StridedArray`s of `Float32` and `Float64` values, FFTW's own domain, and on FFTL for
the rest, for example `BigFloat` arrays. Since `AbstractFFTs` exports the same names, a module
that wants FFTL's functions without the prefix writes `import AbstractFFTs` rather than
`using AbstractFFTs` next to `using FFTL`.

## The generic interface

Every FFTL plan is an `AbstractFFTs.Plan` and supports `size`, `ndims`, `fftdims`,
`mul!(y, p, x)`, `p * x`, `inv(p)` and `p \ y`. Code written against `AbstractFFTs.Plan`
accepts an FFTL plan:

```@example fftw
using FFTL, LinearAlgebra
import AbstractFFTs

function roundtrip(p::AbstractFFTs.Plan, x)
  y = p * x
  p \ y
end

x = randn(ComplexF64, 32, 8)
p = plan_fft(x)
roundtrip(p, x) ≈ x
```

## What FFTL adds

- Number types beyond `Float32` and `Float64`, computed in their own precision
  ([Number types](@ref "Number types")).
- GPU arrays with the same functions ([GPU](gpu.md)).
- Passes over one dimension of a strided array, with pruning
  ([Transforms along dimensions](@ref "Passes: one dimension of a strided array")).
- A C library compiled from the Julia code ([libfftl](@ref "C library (libfftl)")).
