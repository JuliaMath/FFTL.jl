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
| batched 1D passes (`plan_pass`, `plan_r2r_pass`) | ✓ | – | – | – |
| any rank and any subset of dimensions | ✓ | ✓ | ✓ | ✓ |
| any length | ✓ prime factors above 73 resampled (`Float64`: Bluestein's algorithm for single lines on aarch64, everywhere with a preference, see [Accuracy](@ref "Accuracy")) | ✓ prime factors above 73 by Bluestein's algorithm | ✓ prime factors above 31 by Rader's or Bluestein's algorithm | ✓ as on the CPU |

## Accuracy

The error of a transform, as the largest error over the largest output against a reference in
higher precision, in units of ε of the element type (`eps(Float64)` ≈ 2.2·10⁻¹⁶,
`eps(Float32)` ≈ 1.2·10⁻⁷). Measured on an Apple M1 Max and an AMD EPYC 7702 with random inputs,
for one dimension; in several dimensions the errors of the dimensions add up.

| lengths | `Float64` | `Float32` |
|:--|:--|:--|
| prime factors up to 73 (codelets) | 0.8–2.1 ε | 0.7–1.8 ε |
| prime factors above 73, default (resampling) | up to about 25 ε | up to about 2.5 ε |
| prime factors above 73, preference `"bluestein"` | 2–4 ε | (resampling, as above) |
| other floating-point types (generic path) | a few ε of the type | |

Resampling in `Float64` trades accuracy for speed: the deconvolution amplifies the rounding of its
grid transform (see [Algorithms](@ref "Lengths with large prime factors: resampling")). Where 25 ε, about
5·10⁻¹⁵, is too much, the preference `float64_large_primes = "bluestein"` selects Bluestein's
algorithm for every `Float64` transform of such a length, at 1.1–1.6 times the time:

```julia
using Preferences, FFTL
Preferences.set_preferences!(FFTL, "float64_large_primes" => "bluestein"; force = true)
```

The next load recompiles FFTL; [`FFTL.FLOAT64_LARGE_PRIMES`](@ref) reports the setting. The tests
check these bounds for both settings.

## Platforms

| platform | status |
|:--|:--|
| Apple M1 Max (arm64, macOS) | development machine; codelets use the `FCADD`/`FCMLA` complex instructions; GPU via Metal |
| AMD EPYC 7763 (Zen 3, x86-64 with AVX2, Linux) | x86-64 development machine and the x86-64 machine of the [benchmarks](@ref "Benchmarks"); hosts the RTX A6000 |
| AMD EPYC 7702 (Zen 2, x86-64 with AVX2, Linux) | x86-64 development machine (thresholds, tests) |
| Intel Core i7-1185G7 (Tiger Lake, x86-64 with AVX-512, Linux) | 512-bit kernels tested; radix rules not retuned for AVX-512 |
| Linux and macOS in CI | quick tests on every push to `main` and on pull requests; full tests weekly |
| NVIDIA RTX A6000 | GPU via CUDA, measured against cuFFT |
| AMD, Intel GPUs | not tested |
| Windows | not tested |

### Code compiled for other CPUs

FFTL compiles its kernels for the CPU it is loaded on: AVX2 or AVX-512 on x86-64, the
`FCADD`/`FCMLA` complex instructions on aarch64. Code compiled ahead of time for other CPUs, such
as a PackageCompiler app or sysimage with `cpu_target = "generic"`, needs the preference
`cpu_features = "baseline"`, which limits the kernels to the instructions every CPU of the
platform has:

```julia
using Preferences, FFTL
Preferences.set_preferences!(FFTL, "cpu_features" => "baseline"; force = true)
```

On x86-64 the kernels then use 16-byte vectors (about 1.5 times the time of AVX-512 for a real 3D
transform in `Float32`). On Apple silicon `FCADD`/`FCMLA` stay in use, so the target must be
`apple-m1` or later. [`FFTL.CPU_FEATURES`](@ref) reports the setting; CI runs the tests with it on
Linux.

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
