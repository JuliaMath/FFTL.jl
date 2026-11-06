# Julia API

```@meta
CurrentModule = FFTL
```

The names listed here are public. FFTL exports the DFT functions `fft`, `bfft`, `ifft`, `rfft`,
`brfft`, `irfft` and their `plan_` forms; they are the functions of AbstractFFTs.jl, to which
FFTL adds methods ([AbstractFFTs](@ref "AbstractFFTs")). All other names are used with the
prefix, `FFTL.r2r`, `FFTL.dct` and so on.

## Complex transforms

The transform functions are AbstractFFTs.jl's; each makes the plan below and applies it.

| function | computes |
|:--|:--|
| `fft(x, region = 1:ndims(x))` | forward DFT along `region`, unnormalised; equal to `plan_fft(x, region) * x` |
| `bfft(x, region = 1:ndims(x))` | backward DFT, unnormalised: `bfft(fft(x))` is `x` times the number of transformed elements |
| `ifft(x, region = 1:ndims(x))` | normalised inverse: `ifft(fft(x)) ≈ x` |

```@docs
plan_fft
```

`plan_bfft` and `plan_ifft` are documented with `plan_fft`.

## Real transforms

| function | computes |
|:--|:--|
| `rfft(x, region = 1:ndims(x))` | DFT of real `x`, the half spectrum along the first dimension of `region` (`n ÷ 2 + 1` values) |
| `brfft(X, n, region = 1:ndims(X))` | unnormalised inverse of `rfft`; `n` is the length of the real array along the first dimension of `region` |
| `irfft(X, n, region = 1:ndims(X))` | normalised inverse: `irfft(rfft(x), size(x, region[1])) ≈ x` |

```@docs
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

## Run-time plans

Plans whose radix sequences and sizes are data, for programs compiled with `juliac --trim`; see
[Compiling ahead of time](@ref "Compiling ahead of time").

```@docs
RPlan
execute!
RRFFTPlan
RR2RPlan
```

`execute_r2c!`, `execute_c2r!` and `execute_r2r!` are documented with the plans they apply.

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
