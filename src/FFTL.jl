"""
    FFTL

Fastest Fourier Transform in Lübeck: a pure-Julia FFT with generated SIMD codelets.

    p = FFTL.plan_fft(x)     # x::Vector{ComplexF64} or Vector{ComplexF32}
    y = p * x
    FFTL.fft(x), FFTL.bfft(x), FFTL.ifft(x)
"""
module FFTL

using SIMD
using LinearAlgebra
import AbstractFFTs

include("complexvec.jl")
include("codegen.jl")
include("kernels.jl")
include("factor.jl")
include("plan.jl")
include("threads.jl")
include("interface.jl")

end
