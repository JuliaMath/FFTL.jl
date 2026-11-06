# Coming from FFTW.jl

`using FFTL` can take the place of `using FFTW`. FFTL exports the same DFT functions, `fft`,
`bfft`, `ifft`, `rfft`, `brfft`, `irfft` and their `plan_` forms; they are the functions of
AbstractFFTs.jl, so code written against them runs unchanged. The real-to-real functions have
FFTW.jl's names and definitions and are called with the prefix, `FFTL.r2r` and `FFTL.dct`.

## The same transforms under the same names

| FFTW.jl | FFTL | remark |
|:--|:--|:--|
| `fft`, `bfft`, `ifft`, `rfft`, `brfft`, `irfft` and their plans | the same | same signs and scalings |
| `FFTW.r2r(x, FFTW.REDFT10)` | `FFTL.r2r(x, FFTL.REDFT10)` | all eleven kinds, same definitions |
| `FFTW.dct`, `FFTW.idct` | `FFTL.dct`, `FFTL.idct` | orthonormal DCT-II and its inverse |
| `plan_fft(x; flags = FFTW.MEASURE)` | `plan_fft(x)` | no `flags` keyword, see below |
| `plan_fft!(x)`, `fft!(x)` | `mul!(x, p, x)` with `p = plan_fft(x)` | in place through `mul!` |
| `FFTW.set_num_threads(n)` | `plan_fft(x; threads = n)` | per plan; default `Threads.nthreads()` |
| wisdom (`export_wisdom`, `import_wisdom`) | not needed | plans do not measure |

The example compares the two on the same data; FFTW.jl is not a dependency of FFTL, so it is
shown, not run:

```julia
using FFTW, FFTL

x = randn(12, 7)
FFTL.dct(x) ≈ FFTW.dct(x)                                    # true
FFTL.r2r(x, FFTL.RODFT11, 1) ≈ FFTW.r2r(x, FFTW.RODFT11, 1)  # true, likewise for every kind
```

## Planning without flags

FFTW's planner can measure candidate algorithms at run time (`MEASURE`, `PATIENT`) and keep the
result as wisdom. FFTL has no such step: the radix sequence of a length comes from a fixed rule
that was derived from offline measurements, and the codelets are compiled for exactly that
sequence. A plan therefore gives the same algorithm in every session, and there are no flags to
choose. The cost that remains is compilation: the first plan of a new radix sequence compiles its
kernels, which [Precompilation](@ref "Precompilation") moves to package installation for the
common sizes.

## The generic interface

Every FFTL plan is an `AbstractFFTs.Plan` and supports `size`, `ndims`, `fftdims`,
`mul!(y, p, x)`, `p * x`, `inv(p)` and `p \ y`. Code written against `AbstractFFTs.Plan`
accepts an FFTL plan:

```@example fftw
using FFTL, AbstractFFTs, LinearAlgebra

function roundtrip(p::AbstractFFTs.Plan, x)
  y = p * x
  p \ y
end

x = randn(ComplexF64, 32, 8)
p = plan_fft(x)
roundtrip(p, x) ≈ x
```

## FFTL and FFTW.jl in one session

Both packages add methods to the same AbstractFFTs functions, so they can be loaded together.
FFTW.jl's methods are the more specific ones: with both loaded, `fft` of a strided `Float32` or
`Float64` array runs on FFTW, and FFTL computes the cases FFTW.jl does not handle, such as
`BigFloat` and the other floating-point types. Load only FFTL to have FFTL compute the standard
arrays.

## What FFTL adds

- Number types beyond `Float32` and `Float64`, computed in their own precision
  ([Number types](@ref "Number types")).
- GPU arrays with the same functions ([GPU](gpu.md)).
- Passes over one dimension of a strided array, with pruning
  ([Transforms along dimensions](@ref "Passes: one dimension of a strided array")).
- A C library compiled from the Julia code ([libfftl](@ref "C library (libfftl)")).
