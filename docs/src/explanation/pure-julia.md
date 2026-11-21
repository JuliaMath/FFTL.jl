# Other pure-Julia FFTs

FFTL is not the only FFT written in Julia. The packages on this page have goals of their own and
are not part of the [Benchmarks](@ref "Benchmarks"), which compare FFTL with the full-featured
reference libraries that it would replace as a dependency (FFTW, ducc0, vDSP, cuFFT, VkFFT).

## FFTA.jl

[FFTA.jl](https://github.com/JuliaMath/FFTA.jl) ("Fastest Fourier Transform in my Apartment") is
built for genericity. It transforms arrays of any element type that provides a few basic
operations, so the same call computes an FFT of `Complex{BigFloat}` numbers in high precision, of
symbolic variables from Symbolics.jl (which shows the formula of a transform), or of a small
`SVector` from StaticArrays.jl. Its code base is small and readable.

## FFTJ.jl

[FFTJ.jl](https://github.com/oscardssmith/FFTJ.jl) is an experimental FFT built on SIMD.jl and the
AbstractFFTs interface. Several of its ideas have been contributed to FFTL, among them the batched resampling of lengths with a smooth part and a large prime factor (see
[Lengths with a smooth part](@ref "Lengths with a smooth part")).

## PureFFT.jl and PureFFTGPU.jl

[PureFFT.jl](https://github.com/el-oso/PureFFT.jl) started as an investigation of where the speed
of FFTW and RustFFT comes from and whether pure Julia can reach it, and aims to become a
general-purpose, dependency-free FFT library. It transforms complex data of any length, with
Rader's and Bluestein's algorithms for primes, has real transforms, and offers autotuned plans.
Its documentation explains in detail which technique contributes what to the speed.

[PureFFTGPU.jl](https://github.com/el-oso/PureFFTGPU.jl) is its companion for GPUs with the same
interface: kernels written in Julia with KernelAbstractions.jl, with vendor-specific kernels as
package extensions, rather than a wrapper around the vendor library.
