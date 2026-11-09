# FFTW and FFTL

FFTW, the "Fastest Fourier Transform in the West" by Matteo Frigo and Steven G. Johnson, is one
of the most important pieces of numerical software in existence. For more than a quarter of a
century it has been the FFT that scientific computing relies on, and the reference every new FFT
is measured against, including this one. FFTL's name is a nod to it. This page describes what
FFTW achieved and what FFTL takes from it.

## A short history

The fast Fourier transform entered computing with the paper of Cooley and Tukey in 1965 [^1]; the
idea goes back to Gauss, as Heideman, Johnson and Burrus showed in 1984 [^2]. For three decades
afterwards, a fast Fourier transform on a new machine meant hand-written code for that machine.

FFTW changed this. Frigo and Johnson started it at MIT in the late 1990s with an idea that was
new for numerical libraries: instead of one implementation tuned by hand, FFTW combines many
small, automatically generated kernels and lets a planner choose, at run time and on the actual
machine, the combination that is fastest there ("FFTW: an adaptive software architecture for
the FFT", ICASSP 1998 [^3]). The kernels, the codelets, come from `genfft`, a special-purpose compiler
written in OCaml that derives straight-line code for small DFTs symbolically, simplifies it and
schedules it for the register file [^4]. The PLDI
paper received the conference's Most Influential Paper Award ten years later, and FFTW received
the J. H. Wilkinson Prize for Numerical Software in 1999.

FFTW 3 [^5] added SIMD codelets, the real-to-real transforms with their eleven kinds, multithreading and the
guru interface for arbitrary strides and batches. Along the way, Johnson and Frigo found a
split-radix variant with fewer arithmetic operations than any power-of-two algorithm known since
1968 [^6].

FFTW's interface became a standard of its own [^7]. MATLAB's `fft` functions are based on FFTW, and
vendor libraries offer FFTW-compatible interfaces so that existing programs can switch to them.
Julia shipped FFTW in its standard library until version 0.7; since then FFTW.jl provides it,
and the conventions of FFTW.jl are the conventions of the Julia ecosystem.

That a library designed in the late 1990s is still the benchmark for every new FFT on every new
processor is the clearest measure of what its two authors accomplished. It is an exceptional
achievement, sustained over decades.

## What FFTL takes from FFTW

FFTL is written from FFTW's papers and documentation; FFTW's source code was not used. From
these, FFTL takes:

- **Generated codelets.** FFTL's symbolic generator writes straight-line DFT code for small radices,
  the idea of `genfft`. FFTL lets Julia's compiler do this when a radix and element type are
  first used, instead of generating C code ahead of time (see [Algorithms](@ref "Algorithms")).
- **The real-to-real transforms.** The eleven kinds, their names (`REDFT10`, `RODFT00`, …), their
  unnormalised definitions and the halfcomplex order are FFTW's, so that code and formulas
  written for FFTW carry over unchanged.
- **The interface.** Signs, scalings and function names follow FFTW.jl. libfftl, the C library,
  implements FFTW3's C and Fortran interfaces from the FFTW manual; the interface is FFTW's
  design.

## Where FFTL goes another way

FFTL has no run-time planner. The radix sequence of a length comes from a fixed rule found by
offline searches on the development machines, and the sequence is part of the plan's type, so
Julia compiles the whole transform for it. Plans are therefore reproducible and need no wisdom,
at the price of compiling each new radix sequence once. Stockham chains take the place of the
decompositions FFTW's planner chooses, multidimensional transforms run without transposition, and
one code base serves CPUs, GPUs and the C library. On the measured machines FFTL is faster in many
cases and FFTW in others ([Benchmarks](@ref "Benchmarks")); FFTW is always measured at its best
setting (`MEASURE`, built with native CPU flags).

## References

[^1]: J. W. Cooley and J. W. Tukey, "An algorithm for the machine calculation of complex Fourier series", *Mathematics of Computation* 19 (1965), 297–301.

[^2]: M. T. Heideman, D. H. Johnson and C. S. Burrus, "Gauss and the history of the fast Fourier transform", *IEEE ASSP Magazine* 1(4) (1984), 14–21.

[^3]: M. Frigo and S. G. Johnson, "FFTW: an adaptive software architecture for the FFT", *Proc. ICASSP* 1998, vol. 3, 1381–1384.

[^4]: M. Frigo, "A fast Fourier transform compiler", *Proc. ACM SIGPLAN PLDI* 1999, 169–180.

[^5]: M. Frigo and S. G. Johnson, "The design and implementation of FFTW3", *Proceedings of the IEEE* 93(2) (2005), 216–231.

[^6]: S. G. Johnson and M. Frigo, "A modified split-radix FFT with fewer arithmetic operations", *IEEE Transactions on Signal Processing* 55(1) (2007), 111–119.

[^7]: FFTW website and manual: [fftw.org](https://www.fftw.org).
