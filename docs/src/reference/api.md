# Julia API

```@meta
CurrentModule = FFTL
```

FFTL does not export its functions. They are called qualified, `FFTL.fft` and so on, so that
the package can be loaded next to FFTW.jl. The names listed here are public.

## Complex transforms

```@docs
fft
bfft
ifft
plan_fft
```

`plan_bfft` and `plan_ifft` are documented with `plan_fft`.

## Real transforms

```@docs
rfft
brfft
irfft
plan_rfft
```

`plan_brfft` and `plan_irfft` are documented with `plan_rfft`.

## Real-to-real transforms

```@docs
dct
plan_r2r
R2RKind
```

`idct` is documented with `dct`, and `r2r` with `plan_r2r`.

The kinds, as constants: `REDFT00` (DCT-I), `REDFT10` (DCT-II), `REDFT01` (DCT-III), `REDFT11`
(DCT-IV), `RODFT00` (DST-I), `RODFT10` (DST-II), `RODFT01` (DST-III), `RODFT11` (DST-IV), `DHT`
(Hartley), `R2HC` and `HC2R` (halfcomplex). Their formulas are on
[Cosine, sine and Hartley transforms](@ref "Cosine, sine and Hartley transforms").

## Passes

Batched one-dimensional transforms of strided arrays, the operations NFFT.jl builds on.

```@docs
plan_pass
plan_r2r_pass
apply!
```

## GPU

Available after `using KernelAbstractions` and a GPU package.

```@docs
plan_ka
plan_ka_rfft
```

`plan_ka_brfft` and `plan_ka_r2r` are documented with `plan_ka_rfft`.

## Cache and configuration

```@docs
FFTL.clear_cache!
FFTL.cpu_caches
FFTL.set_cpu_caches!
FFTL.PRECOMPILE_WORKLOAD
```

## Plan types

Each plan is an `AbstractFFTs.Plan`. The types are listed for reference; their fields are not
part of the interface.

| type | returned by |
|:--|:--|
| `FFTLPlan` | `plan_fft`, `plan_bfft`, `plan_ifft` (wrapped in `ScaledPlan`) |
| `RFFTLPlan` | `plan_rfft` |
| `BRFFTLPlan` | `plan_brfft`, `plan_irfft` (wrapped in `ScaledPlan`) |
| `R2RPlan` | `plan_r2r` |
| `PassPlan` | `plan_pass` |
| `R2RPassPlan` | `plan_r2r_pass` |
| `GPlan`, `GRPlan`, `GR2RPlan` | the same functions for `BigFloat` and other number types |

The plans of the generic and the GPU paths follow the same protocol; see
[AbstractFFTs](@ref "AbstractFFTs").
