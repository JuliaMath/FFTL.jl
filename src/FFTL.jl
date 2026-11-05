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

include("limits.jl")
include("tuning.jl")
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
include("rreal.jl")
include("generic.jl")
include("ka.jl")
include("precompile.jl")

# The API (not exported: FFTL is used qualified next to FFTW.jl). `public` exists from Julia 1.11.
@static if VERSION >= v"1.11"
  eval(Expr(:public, :fft, :bfft, :ifft, :rfft, :brfft, :irfft, :r2r, :dct, :idct,
            :plan_fft, :plan_bfft, :plan_ifft, :plan_rfft, :plan_brfft, :plan_irfft, :plan_r2r,
            :R2RKind, :REDFT00, :REDFT10, :REDFT01, :REDFT11, :RODFT00, :RODFT10, :RODFT01, :RODFT11,
            :DHT, :R2HC, :HC2R,
            :plan_pass, :plan_r2r_pass, :apply!,
            :RPlan, :RRFFTPlan, :RR2RPlan, :execute!, :execute_r2c!, :execute_c2r!, :execute_r2r!,
            :plan_ka, :plan_ka_rfft, :plan_ka_brfft, :plan_ka_r2r,
            :cpu_caches, :set_cpu_caches!, :clear_cache!, :PRECOMPILE_WORKLOAD))
end

end
