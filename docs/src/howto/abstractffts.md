# Using FFTL next to FFTW.jl

FFTL.jl does not replace FFTW.jl in the session. Both can be loaded at once, and each is called
by its own name:

```julia
using FFTW, FFTL

X1 = FFTW.fft(x)     # FFTW
X2 = FFTL.fft(x)     # FFTL
```

FFTL does not add methods to `AbstractFFTs.plan_fft` or `AbstractFFTs.fft` for standard arrays.
That is deliberate. `AbstractFFTs` is the interface package: it defines what a plan is and how
it is applied, and a backend supplies the methods that do the work. If FFTL registered itself
as the backend, loading it would silently change the meaning of `fft` in every package that uses
the generic functions, including FFTW.jl's. So FFTL's functions are qualified, and its plans
are ordinary `AbstractFFTs.Plan`s that the generic interface understands.

## What the generic interface gets

Every plan FFTL makes works with the `AbstractFFTs` protocol:

- `size`, `ndims` and `fftdims` (the transformed dimensions);
- `mul!(y, p, x)` and `p * x`;
- `inv(p)` and `p \ y` for the normalised inverse.

Code written against `AbstractFFTs.Plan` therefore accepts an FFTL plan:

```@example abs
using FFTL, AbstractFFTs, LinearAlgebra

function roundtrip(p::AbstractFFTs.Plan, x)
  y = p * x
  p \ y
end

x = randn(ComplexF64, 32, 8)
p = FFTL.plan_fft(x)
roundtrip(p, x) ≈ x
```

The same code works with an FFTW plan, since both are plans of the same interface.

## What it does not give

Calling the unqualified `AbstractFFTs.fft(x)` with only FFTL loaded finds no method and fails,
as it does with any `AbstractFFTs` without a backend. Use `FFTL.fft` or `FFTL.plan_fft`.
