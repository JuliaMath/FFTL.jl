"""
    FFTL

Fastest Fourier Transform in Lübeck: a pure-Julia FFT with generated SIMD codelets.

    p = FFTL.plan_fft(x)     # x::Vector{ComplexF64} or Vector{ComplexF32}
    y = p * x
    FFTL.fft(x), FFTL.bfft(x), FFTL.ifft(x)

GPU arrays (with KernelAbstractions and a GPU package such as Metal or CUDA loaded) get plans
from the KernelAbstractions extension through the same functions; see [`plan_ka`](@ref).
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
include("rchain.jl")
include("threads.jl")
include("interface.jl")
include("rplan.jl")
include("pass.jl")
include("real.jl")
include("r2r.jl")
include("ka.jl")

end
