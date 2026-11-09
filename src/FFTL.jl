"""
    FFTL

Fastest Fourier Transform in Lübeck: a pure-Julia FFT with generated SIMD codelets.

    p = plan_fft(x)     # x::Vector{ComplexF64} or Vector{ComplexF32}
    y = p * x
    fft(x), bfft(x), ifft(x)

GPU arrays (with KernelAbstractions and a GPU package such as Metal or CUDA loaded) get plans
from the KernelAbstractions extension through the same functions; see [`plan_ka`](@ref).
"""
module FFTL

using SIMD
using LinearAlgebra
# Only the module name: `fft`, `plan_fft`, … are FFTL's own functions, so `using FFTL` always
# reaches FFTL, whichever other FFT package is loaded elsewhere. AbstractFFTs's functions are
# extended only qualified (src/abstractffts.jl). Never `using AbstractFFTs` or
# `import AbstractFFTs: fft`: either one can make FFTL's `fft` the same function as
# AbstractFFTs's.
import AbstractFFTs
# AbstractFFTs's helpers, re-exported as FFTW.jl does (the same bindings, so no conflicts).
using AbstractFFTs: fftdims, fftshift, ifftshift, fftshift!, ifftshift!, Frequencies, fftfreq, rfftfreq

include("limits.jl")
include("tuning.jl")
include("complexvec.jl")
include("codegen.jl")
include("kernels.jl")
include("factor.jl")
include("plan.jl")
include("rchain.jl")
include("kernel_methods.jl")
include("threads.jl")
include("interface.jl")
include("rplan.jl")
include("pass.jl")
include("real.jl")
include("r2r.jl")
include("rreal.jl")
include("realpass.jl")
include("generic.jl")
include("abstractffts.jl")
include("ka.jl")
include("precompile.jl")

# The exported names are FFTW.jl's exports, so that `using FFTL` can take the place of
# `using FFTW`. The DFT and DCT functions are FFTL's own; the helpers are AbstractFFTs's. The
# rest of the API is public, used with the prefix (`FFTL.r2r`, `FFTL.REDFT10`, …), as in FFTW.jl.
# `public` exists from Julia 1.11.
export fft, bfft, ifft, rfft, brfft, irfft,
       plan_fft, plan_bfft, plan_ifft, plan_rfft, plan_brfft, plan_irfft,
       fft!, bfft!, ifft!, plan_fft!, plan_bfft!, plan_ifft!,
       dct, idct, dct!, idct!, plan_dct, plan_idct, plan_dct!, plan_idct!,
       AbstractFFTs, fftdims, fftshift, ifftshift, fftshift!, ifftshift!, Frequencies, fftfreq, rfftfreq
@static if VERSION >= v"1.11"
  eval(Expr(:public, :r2r, :plan_r2r, :set_num_threads, :ESTIMATE, :MEASURE, :PATIENT, :EXHAUSTIVE,
            :R2RKind, :REDFT00, :REDFT10, :REDFT01, :REDFT11, :RODFT00, :RODFT10, :RODFT01, :RODFT11,
            :DHT, :R2HC, :HC2R,
            :plan_pass, :plan_r2r_pass, :plan_r2c_pass, :plan_c2r_pass, :apply!,
            :RPlan, :RRFFTPlan, :RR2RPlan, :execute!, :execute_r2c!, :execute_c2r!, :execute_r2r!,
            :plan_ka, :plan_ka_rfft, :plan_ka_brfft, :plan_ka_r2r,
            :cpu_caches, :set_cpu_caches!, :clear_cache!, :PRECOMPILE_WORKLOAD))
end

end
