# AbstractFFTs

FFTL implements the plan side of [AbstractFFTs.jl](https://github.com/JuliaMath/AbstractFFTs.jl).
Which methods each plan type provides:

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

## What FFTL does not implement

- **`AbstractFFTs.plan_fft` and `AbstractFFTs.fft` for standard arrays.** FFTL does not register
  itself as the backend of the generic functions, so `AbstractFFTs.fft(x)` needs another backend
  such as FFTW.jl loaded. The reason is in [Using FFTL next to FFTW.jl](@ref "Using FFTL next to FFTW.jl").

## Inverses

`inv(p)` of a forward complex plan is the normalised inverse, a `ScaledPlan` wrapping the
backward plan. For a real-to-real plan, `inv` applies the partner kind and the normalisation of
the kind; the factors are on [Cosine, sine and Hartley transforms](@ref "Cosine, sine and Hartley transforms").
