# AbstractFFTs

FFTL is a backend of [AbstractFFTs.jl](https://github.com/JuliaMath/AbstractFFTs.jl). Which
methods each plan type provides:

| method | complex plans | real plans (`rfft`, `brfft`, `irfft`) | real-to-real plans |
|:--|:--|:--|:--|
| `AbstractFFTs.Plan` supertype | ✓ | ✓ | ✓ |
| `size`, `ndims` | ✓ | ✓ | ✓ |
| `AbstractFFTs.fftdims` | ✓ | ✓ | ✓ |
| `mul!(y, p, x)` | ✓ | ✓ | ✓ |
| `p * x` | ✓ | ✓ | ✓ |
| `AbstractFFTs.plan_inv` (`inv(p)`, `p \ y`) | ✓ (`ScaledPlan`) | ✓ | ✓ |

Real plans are `RFFTLPlan` (`rfft`) and `BRFFTLPlan` (`brfft`, `irfft`); the complex ones are
`FFTLPlan`; real-to-real are `R2RPlan`.

## The generic functions

FFTL's exported `fft`, `plan_fft`, … are its own functions, not AbstractFFTs's. For generic
code, FFTL adds methods to AbstractFFTs's `plan_fft`, `plan_bfft`, `plan_fft!`, `plan_bfft!`,
`plan_rfft` and `plan_brfft` for arrays of floating-point numbers; AbstractFFTs derives the
transform functions, the normalised inverses and the promotion of real input from them. So
`AbstractFFTs.fft(x)` runs on FFTL when no other backend is loaded.

These methods are less specific than FFTW.jl's, which take `StridedArray`s of `Float32` and
`Float64` values. With FFTW.jl loaded, the generic functions therefore run on FFTW for those
arrays and on FFTL for the others (other floating-point types, non-strided arrays). Code that
wants FFTL calls FFTL's functions; see [Coming from FFTW.jl](@ref "Coming from FFTW.jl").

## Inverses

`inv(p)` of a forward complex plan is the normalised inverse, a `ScaledPlan` wrapping the
backward plan. For a real-to-real plan, `inv` applies the partner kind and the normalisation of
the kind; the factors are on [Cosine, sine and Hartley transforms](@ref "Cosine, sine and Hartley transforms").
