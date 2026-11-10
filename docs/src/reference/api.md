# Julia API

```@meta
CurrentModule = FFTL
```

The names listed here are public. FFTL exports the names FFTW.jl exports: the DFT and DCT
functions with their `plan_` and in-place forms, and AbstractFFTs.jl's helpers (`fftshift`,
`fftfreq`, …). All other names are used with the prefix, `FFTL.r2r`, `FFTL.REDFT10` and so on.
The exported functions are FFTL's own; the generic functions of AbstractFFTs.jl reach them
through the methods described on [AbstractFFTs](@ref "AbstractFFTs").

## Complex transforms

```@docs
fft
bfft
ifft
plan_fft
plan_fft!
InplacePlan
```

`plan_bfft` and `plan_ifft` are documented with `plan_fft`; `fft!`, `bfft!`, `ifft!`,
`plan_bfft!` and `plan_ifft!` with `plan_fft!`.

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
plan_dct
plan_r2r
R2RKind
```

`idct` is documented with `dct`; `plan_idct`, `plan_dct!`, `plan_idct!`, `dct!` and `idct!` with
`plan_dct`; `r2r` with `plan_r2r`.

The kinds, as constants: `REDFT00` (DCT-I), `REDFT10` (DCT-II), `REDFT01` (DCT-III), `REDFT11`
(DCT-IV), `RODFT00` (DST-I), `RODFT10` (DST-II), `RODFT01` (DST-III), `RODFT11` (DST-IV), `DHT`
(Hartley), `R2HC` and `HC2R` (halfcomplex). Their formulas are on
[Cosine, sine and Hartley transforms](@ref "Cosine, sine and Hartley transforms").

## Passes

Batched one-dimensional transforms of strided arrays, the operations NFFT.jl builds on.

```@docs
plan_pass
plan_pass_runtime
plan_r2c_pass
plan_r2c_pass_runtime
plan_c2r_pass
plan_c2r_pass_runtime
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
FFTL.set_num_threads
FFTL.ESTIMATE
FFTL.clear_cache!
FFTL.cpu_caches
FFTL.set_cpu_caches!
FFTL.PRECOMPILE_WORKLOAD
FFTL.REG_BYTES
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
| `OrthoDCTPlan` | `plan_dct`, `plan_idct` |
| `InplacePlan` | `plan_fft!`, `plan_bfft!`, `plan_ifft!`, `plan_dct!`, `plan_idct!` |
| `PassPlan` | `plan_pass` |
| `R2RPassPlan` | `plan_r2r_pass` |
| `GPlan`, `GRPlan`, `GR2RPlan` | the same functions for `BigFloat` and other number types |

The plans of the generic and the GPU paths follow the same protocol; see
[AbstractFFTs](@ref "AbstractFFTs").
