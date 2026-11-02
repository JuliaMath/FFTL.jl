# FFTL.jl — Fastest Fourier Transform in Lübeck

[![CI](https://github.com/JuliaMath/FFTL.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaMath/FFTL.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/JuliaMath/FFTL.jl/graph/badge.svg?branch=main)](https://codecov.io/gh/JuliaMath/FFTL.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A pure-Julia fast Fourier transform for the CPU and, experimentally, GPUs: complex, real-input
and real-to-real (DCT, DST, Hartley, halfcomplex) transforms of `Float32` and `Float64` data, any
number of dimensions, any length, multithreaded; other float types through a generic path; a C
library built with `juliac --trim`. Work in progress; the interface may still change.

## Usage

```julia
using FFTL, LinearAlgebra

x = randn(ComplexF64, 1000, 1000)
y = FFTL.fft(x)                 # all dimensions
y1 = FFTL.fft(x, 1)             # along dimension 1 only
x2 = FFTL.ifft(y)               # normalised inverse; FFTL.bfft is the unnormalised backward DFT

p = FFTL.plan_fft(x; threads = 8)
mul!(y, p, x)                   # out of place, or mul!(x, p, x) in place
x3 = p \ y                      # inverse through the plan

r = randn(1000, 1000)
R = FFTL.rfft(r)                # half spectrum along dimension 1 (501 × 1000)
r2 = FFTL.irfft(R, 1000)
c = FFTL.dct(r)                 # orthonormal DCT-II; FFTL.idct is its inverse
h = FFTL.r2r(r, (FFTL.REDFT10, FFTL.R2HC))   # unnormalised real-to-real, one kind per dimension
```

Plans use `threads = Threads.nthreads()` by default; transforms below about 2^15 values run on
one thread regardless. Inside code that is already threaded, pass `threads = 1`.

A plan owns its work buffers: one plan must not run on several tasks at the same time (make one
plan per task; plans of the same size share their twiddle tables, so this is cheap).

Plans are `AbstractFFTs.Plan`s. FFTL does not add methods to `AbstractFFTs.fft`/`plan_fft` for
standard arrays, so it can be loaded next to FFTW.jl; call it qualified as `FFTL.fft`,
`FFTL.plan_fft`, … . The first plan for a new radix sequence compiles its kernels (milliseconds
for the sizes of the precompile workload, a fraction of a second to a few seconds otherwise);
later plans of the same size take microseconds and transforms never compile.

`FFTL.plan_pass(v, d; nonzero, needed)` transforms all lines of a strided view along one dimension,
optionally skipping zero inputs and unneeded outputs (pruning); `FFTL.plan_r2r_pass` does the same
for real-to-real kinds. These are the passes NFFT.jl uses.

Other floating-point types (`BigFloat`, `Double64`, `Float128`, …) are transformed in their own
precision by a generic, single-threaded path without SIMD: complex, real (`rfft`, `irfft`) and
real-to-real transforms, any length and dimensions, accurate to a few ulps of the type (tested
with `BigFloat`; `Double64` and `Float128` measured at the same level). `Float16`, integers and
other numbers are converted to `ComplexF64`/`Float64`.

### GPUs

With KernelAbstractions.jl and a GPU package loaded, the same functions work on GPU arrays
(experimental; `Float32`, `Float64` where the GPU supports it). Complex transforms use tuned
multi-stage kernels; `rfft`/`irfft` run at about the speed of a complex transform of the same
size, real-to-real transforms (`r2r`, all kinds) go through a twiddled complex DFT of twice the
length and are not tuned yet:

```julia
using FFTL, Metal, KernelAbstractions          # or CUDA, AMDGPU, oneAPI

xd = MtlArray(randn(ComplexF32, 1024, 1024))
yd = FFTL.fft(xd)
p = FFTL.plan_fft(xd, 2)                       # mul!(yd, p, xd) needs yd !== xd
Rd = FFTL.rfft(MtlArray(randn(Float32, 1024, 1024)))
```

Kernels are queued on the device without synchronisation (copying the result to the host waits
for them; for timing call `KernelAbstractions.synchronize(backend)`). GPU transforms are out of
place, a plan holds about two array-sized device buffers, and making a plan for a new size
generates and compiles its kernels. `threads` is ignored on GPUs.

`FFTL.plan_ka(backend, T, dims, σ; region)`, `plan_ka_rfft`, `plan_ka_brfft` and `plan_ka_r2r` make
plans for any KernelAbstractions backend, including the CPU. The GPU kernels come from the same codelet generator: each kernel computes one
pass of large radix (up to 4096 points) in workgroup memory, a transform is one to five passes per
dimension. Pass and workgroup sizes are tuned on an Apple M1 Max, where Float32 transforms are
faster than Metal.jl's FFT in nearly all measured cases; other GPUs work but are not tuned for
yet.

### Precompilation

FFTL ships a precompile workload that removes the compilation cost of the first plan and
the first transform. The default (`"lean"`) covers the common radices (4, 5, 8, 9, 10 and
16, plus Bluestein) and the compile-time and run-time plan paths for both element types. A
broader workload (`"broad"`) additionally covers the lengths 10^5, 2^20, 128, 360, 243,
the sizes 256×256 and 64×64×64, and a four-step case, at the cost of a larger package image.
Select it with a preference; the next load recompiles FFTL:

```julia
using Preferences, FFTL
Preferences.set_preferences!(FFTL, "precompile_workload" => "broad"; force = true)
```

`"none"` disables the workload. On an M1 Max the lean workload takes 32 s of precompilation
and 96 MB of package image; a first plan and transform of a covered size then take a few
milliseconds instead of about 4 s.

### C library

`libfftl/` builds FFTL into a C library with `juliac --trim=safe` (Julia ≥ 1.12): complex DFT,
r2c/c2r (also in place, FFTW's padded layout) and all real-to-real kinds, any rank, both
precisions, threads; interface in `libfftl/include/fftl.h`, build with `libfftl/build.sh`
(≈ 14 MB library plus ≈ 10 MB Julia runtime on macOS). Experimental; a CI job builds and tests
it on Linux and macOS.

## How it works

- **Generated codelets.** A small symbolic generator writes straight-line code for DFTs of
  size 2–31 (composite sizes split in registers, odd primes in the symmetric form). Codelets are
  generated with `@generated` for the radix, element type and SIMD width actually used.
- **Complex numbers in SIMD registers.** On aarch64 CPUs with the ARMv8.3 complex-number
  extension (all Apple silicon), additions with a factor ±i and complex multiplications use the
  `FCADD` and `FCMLA` instructions; elsewhere the same operations are built from shuffles.
- **Stockham chains.** A transform is a sequence of radix stages (mostly 4 and 8) without bit
  reversal. The sequence is part of the plan's type, so it compiles into one function. The radices
  are chosen by a fixed rule, without measurements.
- **Multidimensional transforms** run the same chains along every dimension: a dimension with
  stride S is a chain started with stride S, the S neighbouring columns forming the vectorised
  index. No transposition.
- **Lengths with prime factors above 31** use Bluestein's algorithm.
- **Threads** split each dimension over slices or blocks of columns; long 1D transforms use the
  four-step algorithm.

## Tests

`] test FFTL` runs a quick selection that covers every feature (about 1.5 min). Set
`FFTL_TEST_FULL=true` for the full lists and `FFTL_TEST_VERBOSE=true` for the time of every test
set. CI runs the quick tests on Linux and macOS for pushes to `main` and pull requests, the full
lists and the oldest supported Julia (1.10) once a week, and builds and tests the C library.

## License

MIT.
