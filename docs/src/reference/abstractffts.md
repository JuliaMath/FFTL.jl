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

The generic functions `AbstractFFTs.fft`, `plan_fft`, `rfft`, … run on FFTL for standard arrays.
How FFTL and FFTW.jl behave when both are loaded is described on
[Coming from FFTW.jl](@ref "Coming from FFTW.jl").

## Inverses

`inv(p)` of a forward complex plan is the normalised inverse, a `ScaledPlan` wrapping the
backward plan. For a real-to-real plan, `inv` applies the partner kind and the normalisation of
the kind; the factors are on [Cosine, sine and Hartley transforms](@ref "Cosine, sine and Hartley transforms").
