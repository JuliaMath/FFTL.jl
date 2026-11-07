# Platforms and limits

Which transform works on which number type and platform. ✓ means implemented and covered by the
test suite; ◐ means implemented but not tuned or not tested on that platform; – means not
available.

## Transforms

| transform | CPU, `Float32`/`Float64` | CPU, other `AbstractFloat`s (`BigFloat`, `Double64`, `Float128`) | GPU (KernelAbstractions) | libfftl |
|:--|:--|:--|:--|:--|
| complex DFT (`fft`, `bfft`, `ifft`) | ✓ threads, SIMD | ✓ one thread | ✓ tuned on Metal, measured on CUDA | ✓ |
| real input `rfft`, `irfft`, `brfft` | ✓ | ✓ | ✓ at about complex speed | ✓ `r2c`, `c2r`, also in place and split |
| DCT-II, the orthonormal `dct`/`idct` | ✓ | ✓ | ✓ | ✓ |
| DCT-I, DCT-III, DCT-IV | ✓ | ✓ | ◐ partly tuned | ✓ |
| DST-II | ✓ | ✓ | ✓ | ✓ |
| DST-I, DST-III, DST-IV | ✓ | ✓ | ◐ partly tuned | ✓ |
| DHT, `R2HC`, `HC2R` | ✓ | ✓ | ◐ not tuned | ✓ |
| batched 1D passes (`plan_pass`, `plan_r2r_pass`) with pruning | ✓ | – | – | – |
| any rank and any subset of dimensions | ✓ | ✓ | ✓ | ✓ |
| any length (Bluestein above prime factor 31) | ✓ | ✓ | ✓ | ✓ |

## Platforms

| platform | status |
|:--|:--|
| Apple M1 Max (arm64, macOS) | development machine; codelets use the `FCADD`/`FCMLA` complex instructions; GPU via Metal |
| AMD EPYC 7763 (Zen 3, x86-64 with AVX2, Linux) | x86-64 development machine |
| AMD EPYC 7702 (Zen 2, x86-64 with AVX2, Linux) | x86-64 machine of the [benchmarks](@ref "Benchmarks") |
| Intel Core i7-1185G7 (Tiger Lake, x86-64 with AVX-512, Linux) | 512-bit kernels tested; radix rules not retuned for AVX-512 |
| Linux and macOS in CI | quick tests on every push to `main` and on pull requests; full tests weekly |
| NVIDIA RTX A6000 | GPU via CUDA, measured against cuFFT |
| AMD, Intel GPUs | not tested |
| Windows | not tested |

## Julia versions

| | version |
|:--|:--|
| the package | Julia 1.10 and newer (weekly CI on 1.10) |
| the libfftl build | Julia 1.12 and newer with `juliac` |

## Limits

| quantity | limit |
|:--|:--|
| element types, complex | `ComplexF32`, `ComplexF64`; any `Complex{T}` with `T` a floating-point type on the generic path |
| element types, real | `Float32`, `Float64`, and every other `AbstractFloat` on the generic path; `Float16` and integers are converted |
| rank | any |
| length of a dimension | any; memory is the limit |
| threads | `Threads.nthreads()` by default; a plan may use fewer |
| transforms on the GPU | out of place; a plan holds about two buffers of the array's size |
| libfftl precision | `float` and `double` (FFTW3's `fftw_` and `fftwf_` functions; no `fftwl_`, `fftwq_`) |
| libfftl interface | FFTW3's C, legacy Fortran and Fortran 2003 interfaces; layouts other than one contiguous array are copied |
| libfftl threads | fixed at build time |

Numbers that are not in this table, such as the largest transform that fits on a particular
machine, depend on the machine and are reported with the [benchmarks](@ref "Benchmarks").
