# libfftl C API

The complete header is [`fftl.h`](@ref "C header (fftl.h)"). This page lists the functions and
the limits; the header has the exact contracts.

## Functions

| function | purpose |
|:--|:--|
| `fftl_version(buf, len)` | the library version as `"major.minor.patch"` |
| `fftl_plan_dft(rank, dims, transform_rank, region, precision, sign, threads)` | complex DFT plan, `sign` −1 forward or +1 backward |
| `fftl_plan_dft_r2c(rank, dims, transform_rank, region, precision, threads)` | real input, half spectrum out |
| `fftl_plan_dft_c2r(rank, dims, transform_rank, region, precision, threads)` | half spectrum in, real output, unnormalised |
| `fftl_plan_r2r(rank, dims, transform_rank, region, kinds, precision, threads)` | real-to-real transforms, one kind per dimension |
| `fftl_plan_execute(plan, out, in)` | apply a plan of any of the four kinds; `out == in` runs in place |
| `fftl_plan_destroy(plan)` | free a plan |
| `fftl_last_error(buf, len)` | the message of the last failed call on this thread |

The kinds are `FFTL_REDFT00` … `FFTL_REDFT11` (DCT-I to DCT-IV), `FFTL_RODFT00` … `FFTL_RODFT11`
(DST-I to DST-IV), `FFTL_DHT`, `FFTL_R2HC` and `FFTL_HC2R`.

## Limits

| | limit |
|:--|:--|
| rank | any, at least 1 |
| dimension lengths | any, at least 1; lengths with large prime factors use Bluestein's algorithm |
| transform rank | 0 to the rank; the region is strictly ascending, 0-based |
| precision | `FFTL_PRECISION_DOUBLE` (`Complex{Float64}`, 16 bytes) and `FFTL_PRECISION_SINGLE` (`Complex{Float32}`, 8 bytes) |
| sign | −1 or +1 for complex plans |
| normalisation | none; the backward transform multiplies by the number of transformed elements |
| threads | the value is stored per plan; the worker pool is fixed when the library is built (`FFTL_THREADS`). More threads than workers run on the workers that exist |
| memory | bounded by the available memory and by 64-bit element offsets |
| concurrency | different plans execute in parallel; one plan on one thread at a time; destroying a plan during its execution is not allowed |
| number types | `float` and `double` only; no GPU, no generic number types |

## Build and size

On macOS (arm64), the build output in `libfftl/build/lib` is about 23 MB: the library and the Julia
runtime it embeds. No Julia installation is needed to run a program that links it. The build
needs Julia 1.12 or newer with `juliac`. See [C library (libfftl)](@ref "C library (libfftl)")
for the build variables.

## Errors

Constructors return a negative handle on failure; `fftl_plan_execute` and `fftl_plan_destroy`
return −1 on a bad handle or a failed execution. The reason is in `fftl_last_error`, which keeps
its message until the next failed call on the same thread.
