![FFTL.jl](assets/fftl-wide.svg)

# FFTL.jl

*Fastest Fourier Transform in Lübeck*: a fast Fourier transform written in Julia. Complex, real
and real-to-real transforms of any length and any number of dimensions, in `Float32`, `Float64`
and other floating-point types, on several threads with SIMD on the CPU, through
KernelAbstractions on GPUs, and as a C library ([`libfftl`](@ref "C library (libfftl)"))
compiled from the same Julia code.

!!! warning "Experimental"
    FFTL.jl is under development and was recently registered. The interface can still change; the
    documented parts are the ones tested in CI.

## Background

FFTL was written by a coding agent under the guidance of a domain expert. Julia makes a library
of this scope possible in one language: the compiler generates the codelets for the radix,
element type and SIMD width in use, the same generator serves the GPU kernels, and `juliac`
turns the package into a C library. A coding agent makes it possible to implement, test and
measure the many details such a library needs. The library builds on decades of work on fast
Fourier transforms, above all on FFTW; [FFTW and FFTL](@ref "FFTW and FFTL") describes that
history.

FFTL is meant to be developed further by the Julia community, as one FFT that the ecosystem
shares. [Contribute](@ref "Contribute") says where help is most useful.

## Scope / Coverage

**Fourier and discrete cosine transforms written in Julia, with wide support for number types and CPU and GPU execution.**

- Complex, real-input and all eleven real-to-real kinds (DCT-I to DCT-IV, DST-I to DST-IV,
  Hartley, halfcomplex), along any set of dimensions of any rank.
- Any length: smooth lengths through generated codelets, lengths with large prime factors
  resampled onto a smooth grid, no padding to powers of two.
- `BigFloat` and other floating-point types in their own precision, not converted to `Float64`.
- GPUs through KernelAbstractions.jl, with the same functions.
- Multithreading where the transform is large enough, and the four-step algorithm for one long line
  (see [Threads](@ref "Threads")).

**A C library made from the Julia code.** [libfftl](@ref "C library (libfftl)") is the package
compiled with `juliac --trim`, with FFTW3's C and Fortran interfaces, so programs written for
FFTW3 compile and link against it without source changes. No transform is written in C, and a program that links the
library needs no Julia installation. The run-time plans it is built from are public, so Julia
programs that use them can be trimmed as well.

**Entry points for special variants.** [Passes](@ref "Passes: one dimension of a strided array")
transform one dimension of a strided array. Libraries that assemble their own transforms from such
passes, such as NFFT.jl, use them.

**The interface of the ecosystem.** FFTL exports FFTW.jl's names with FFTW.jl's conventions and
scalings, so `using FFTL` can take the place of `using FFTW`, and it is a backend of
AbstractFFTs.jl's generic functions (see [Coming from FFTW.jl](@ref "Coming from FFTW.jl")).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaMath/FFTL.jl")
```

## Quick start

```@example home
using FFTL, LinearAlgebra

x = randn(ComplexF64, 1000, 1000)
y = fft(x)                  # all dimensions, unnormalised
x2 = ifft(y)                # normalised inverse
x2 ≈ x
```

```@example home
p = plan_fft(x)             # a plan holds the work buffers and tables
mul!(y, p, x)               # apply it, as often as you like
y ≈ fft(x)
```

```@example home
r = randn(1000, 1000)
R = rfft(r)                 # half spectrum along dimension 1: 501 × 1000
irfft(R, 1000) ≈ r
```

```@example home
C = dct(r)                  # orthonormal DCT-II; idct inverts it
idct(C) ≈ r
```

## Further reading

| If you want to | read |
|:--|:--|
| run your first transform | [Your first transform](@ref "Your first transform") |
| see the library used on real problems | [Tutorials](@ref "Spectrum of a sampled signal") |
| move code from FFTW.jl | [Coming from FFTW.jl](@ref "Coming from FFTW.jl") |
| solve a specific task: plans, real data, threads, GPUs, C | [How-to Guides](@ref "Plans and reuse") |
| understand the mathematics and the algorithms | [Explanation](@ref "The discrete Fourier transform") |
| look up a function, a kind, a limit | [Reference](@ref "Julia API") |

## Design in one paragraph

A plan is a Julia object that carries the radix sequence, the twiddle tables and the work
buffers for one size and one set of dimensions. Radix sequences are chosen by a fixed rule, and
the codelets they run are generated for the radix, element type and SIMD width actually used.
Long transforms run on several threads, and lengths with large prime factors are resampled onto a
smooth grid. The details are in [Algorithms](@ref "Algorithms").

## Package status

| | |
|:--|:--|
| Julia | 1.10 and newer (CI: 1.10 and current) |
| License | MIT |
| Interface | AbstractFFTs.jl backend; plans are `AbstractFFTs.Plan`s with `mul!`, `*`, `\` and `inv` |
| Platforms | Linux and macOS (CI); Apple M1 Max is the development machine |
| GPU | experimental, via KernelAbstractions.jl |
| C library | experimental, `juliac --trim`, see [libfftl](@ref "C library (libfftl)") |

The complete matrix of transforms, number types and platforms is in
[Platforms and limits](@ref "Platforms and limits").
