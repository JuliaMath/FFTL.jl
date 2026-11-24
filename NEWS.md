# News

## Unreleased

- `cpu_features = "baseline"` leaves out `FCADD`/`FCMLA` on Apple silicon too, so the kernels
  compile for any target, `generic` and WebAssembly included (#75). Code compiled for `apple-m1`
  or later keeps `"host"`: without these instructions transforms take 1.2 to 2.4 times as long.

## 0.3.4

- `rfft`, `brfft`, `irfft` and their plans halve the first dimension of `region`, as AbstractFFTs
  specifies, also when `region` is not sorted: `rfft(x, (2, 1, 3))` halves dimension 2. Before,
  FFTL sorted the region and halved its smallest dimension. This holds for the CPU, generic
  (`BigFloat`), run-time and KernelAbstractions/GPU plans (#70).
- The cache sizes are read on Windows too (`GetLogicalProcessorInformationEx`); before, Windows
  always used the defaults. The x86 default for the L2 per core, used where no query works, is
  256 KiB instead of 512 KiB (#71).
- Citation metadata (`CITATION.cff`) for GitHub and Zenodo (#68, #69).

## 0.3.3

- Real-to-real transforms of complex arrays, as in FFTW.jl: `FFTL.r2r`, `dct`, `idct` and their
  plans transform the real and imaginary parts separately. On the CPU both parts go through one
  real plan over the interleaved layout. Before, the plan was built and failed with an
  `InexactError` when applied (#63).
- `FFTL.r2r!` and `FFTL.plan_r2r!`, the in-place forms of FFTW.jl (#63).
- `mul!` with r2r plans accepts views and reinterpreted arrays; dense ones are transformed without
  a copy (#63).
- A single line takes the four-step algorithm on several threads only where it is faster than one
  thread: on aarch64 from 128 KiB with at least 2 threads, on x86 from 2^15 values with at least
  6 threads (new `TUNING` fields `fourstep_min_bytes`, `fourstep_min_threads`). Before, it started
  at 2^13 values with 2 threads and was up to 11 times slower than one thread there (#65).
- Documentation: page on the other pure-Julia FFT packages (#64).

## 0.3.2

- AVX-512 CPUs that Julia compiles for an older CPU name without 512-bit registers (Julia 1.13
  compiles for `znver3` on a Zen 4 Ryzen 5 PRO 215): FFTL failed to precompile with "LLVM ERROR:
  Do not know how to split the result of this operator!". FFTL now uses 32-byte kernels there, as
  with `simd_bytes = 32` (new constant `FFTL.HAS_ZMM`), and the ±i step of the 512-bit kernels
  uses the generic `fma` instead of the target intrinsic `llvm.x86.avx512.vfmaddsub.*.512`, which
  LLVM cannot split (#59).
- The x86 kernels take AVX2, AVX-512 and the 512-bit registers from Julia's code generation target,
  checked on compiled code, instead of the CPU and a list of CPU names. Julia started with `-C` for
  a target below the CPU no longer aborts precompilation (`-C x86-64` on an AVX2 CPU did). CI tests
  on Windows and with `-C x86-64` and `-C haswell` (#60).

## 0.3.1

- The run-time pass plans (`plan_pass_runtime`, `plan_r2c_pass_runtime`, `plan_c2r_pass_runtime`)
  compile with `juliac --trim=safe` again; in 0.3.0 the new kernels for large prime factors left
  calls the trim verifier could not resolve. A trim probe of these plans now runs in the libfftl
  workflow (#57).
- Documentation: benchmark panels of the 0.3.0 campaign (#56).

## 0.3.0

- Lengths with prime factors above 73: the large-prime part is resampled onto a smooth grid
  (one transform of about twice its length) instead of Bluestein's algorithm (two such
  transforms); batched over the lanes of a composite length, and on threads for a long line (#29).
  Faster on most such lengths. `Float64` on aarch64 keeps Bluestein's algorithm for single lines.
- Accuracy for such lengths: up to about 25 ε in `Float64` (largest error over the largest output,
  about 5·10⁻¹⁵; FFTL 0.2: 2–4 ε), up to 2.5 ε in `Float32` (more accurate than before). The
  preference `float64_large_primes = "bluestein"` restores Bluestein's algorithm for every `Float64`
  transform of such a length, at 1.1–1.6 times the time on most of them (#47, #49).
- Four-step transforms of long lines with cache buffers; the row copy and the non-temporal output
  only with 64-byte vectors (#33).
- Preference `cpu_features = "baseline"`: kernels with only the instructions every CPU of the
  platform has, for code compiled ahead of time for other CPUs, such as a PackageCompiler app
  with `cpu_target = "generic"` (#53).
- The precompile workload no longer calls `randn`, so loading FFTL does not need Random (#48, #50).
- libfftl: `fftl_version` reports the version of the package the library is built from.
- Documentation: Benchmarks in the main menu, an accuracy section, two tutorials on any length
  and on precision (#51, #52).

## 0.2.1

- Julia 1.10 is supported again (#45).
