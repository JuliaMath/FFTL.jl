# FFTL.jl — Fastest Fourier Transform in Lübeck

A pure-Julia fast Fourier transform for the CPU: complex `Float32` and `Float64` data, any number
of dimensions, any length, multithreaded. Work in progress; the interface may still change.

## Usage

```julia
using FFTL

x = randn(ComplexF64, 1000, 1000)
y = FFTL.fft(x)                 # all dimensions
y1 = FFTL.fft(x, 1)             # along dimension 1 only
x2 = FFTL.ifft(y)               # normalised inverse; FFTL.bfft is the unnormalised backward DFT

p = FFTL.plan_fft(x; threads = 8)
mul!(y, p, x)                   # out of place, or mul!(x, p, x) in place
x3 = p \ y                      # inverse through the plan
```

Plans are `AbstractFFTs.Plan`s. FFTL does not add methods to `AbstractFFTs.fft`/`plan_fft` for
standard arrays, so it can be loaded next to FFTW.jl; call it qualified as `FFTL.fft`,
`FFTL.plan_fft`, … . The first plan for a new length compiles its kernels (a fraction of a second);
later plans and all transforms run without compilation.

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

## License

MIT.
