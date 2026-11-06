# Coming from FFTW.jl

FFTL uses the names, sign conventions and scalings of FFTW.jl. Code that calls FFTW.jl through
the generic AbstractFFTs.jl functions runs on FFTL, and code that calls FFTW.jl's own functions
moves over by changing the prefix. The real-to-real kinds have FFTW's names and FFTW's
unnormalised definitions, and `dct` is the same orthonormal DCT-II.

## The same transforms under the same names

| FFTW.jl | FFTL | remark |
|:--|:--|:--|
| `fft`, `bfft`, `ifft` | `FFTL.fft`, `FFTL.bfft`, `FFTL.ifft` | same signs and scalings |
| `rfft`, `brfft`, `irfft` | `FFTL.rfft`, `FFTL.brfft`, `FFTL.irfft` | half spectrum along the first dimension of the region |
| `r2r(x, FFTW.REDFT10)` | `FFTL.r2r(x, FFTL.REDFT10)` | all eleven kinds, same definitions |
| `dct`, `idct` | `FFTL.dct`, `FFTL.idct` | orthonormal DCT-II and its inverse |
| `plan_fft(x; flags = FFTW.MEASURE)` | `FFTL.plan_fft(x)` | no flags, see below |
| `plan_fft!(x)`, `fft!(x)` | `mul!(x, p, x)` with `p = FFTL.plan_fft(x)` | in place through `mul!` |
| `FFTW.set_num_threads(n)` | `FFTL.plan_fft(x; threads = n)` | per plan; default `Threads.nthreads()` |
| wisdom (`export_wisdom`, `import_wisdom`) | not needed | plans do not measure |

The example compares the two on the same data; FFTW.jl is not a dependency of FFTL, so it is
shown, not run:

```julia
using FFTW, FFTL

x = randn(12, 7)
FFTL.dct(x) ≈ FFTW.dct(x)                              # true
FFTL.r2r(x, FFTL.RODFT11, 1) ≈ FFTW.r2r(x, FFTW.RODFT11, 1)   # true, likewise for every kind
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
p = FFTL.plan_fft(x)
roundtrip(p, x) ≈ x
```

!!! note "FFTL and FFTW.jl in one session"
    FFTL is becoming the backend of the generic functions `AbstractFFTs.fft`, `plan_fft`, … for
    standard arrays. How the two packages behave when both are loaded is described here once
    that work is merged. The qualified calls `FFTL.fft` and `FFTW.fft` always select one of the
    two explicitly.

## What FFTL adds

- Number types beyond `Float32` and `Float64`, computed in their own precision
  ([Number types](@ref "Number types")).
- GPU arrays with the same functions ([GPU](gpu.md)).
- Passes over one dimension of a strided array, with pruning
  ([Transforms along dimensions](@ref "Passes: one dimension of a strided array")).
- A C library compiled from the Julia code ([libfftl](@ref "C library (libfftl)")).
