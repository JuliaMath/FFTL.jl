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
# The transform and planning functions are AbstractFFTs's functions: FFTL adds methods to them
# (less specific than FFTW.jl's) and re-exports the same bindings, so `using FFTL` is a drop-in
# replacement for `using FFTW` and both packages can be loaded together without conflicts.
using AbstractFFTs: fft, bfft, ifft, rfft, brfft, irfft
import AbstractFFTs: plan_fft, plan_bfft, plan_ifft, plan_rfft, plan_brfft, plan_irfft

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

# The drop-in surface is exported (Variant A): the DFT names are AbstractFFTs's functions with
# FFTL's methods (re-exported above), so `using FFTL` replaces `using FFTW` and both packages
# can be loaded together. The internals stay public-not-exported. `public` exists from Julia 1.11.
export fft, bfft, ifft, rfft, brfft, irfft,
       plan_fft, plan_bfft, plan_ifft, plan_rfft, plan_brfft, plan_irfft
@static if VERSION >= v"1.11"
  eval(Expr(:public, :r2r, :dct, :idct, :plan_r2r,
            :R2RKind, :REDFT00, :REDFT10, :REDFT01, :REDFT11, :RODFT00, :RODFT10, :RODFT01, :RODFT11,
            :DHT, :R2HC, :HC2R,
            :plan_pass, :plan_r2r_pass, :apply!,
            :RPlan, :RRFFTPlan, :RR2RPlan, :execute!, :execute_r2c!, :execute_c2r!, :execute_r2r!,
            :plan_ka, :plan_ka_rfft, :plan_ka_brfft, :plan_ka_r2r,
            :cpu_caches, :set_cpu_caches!, :clear_cache!, :PRECOMPILE_WORKLOAD))
end

end
