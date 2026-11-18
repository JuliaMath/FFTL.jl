# Trim probe for the run-time pass plans (plan_pass_runtime, plan_r2c_pass_runtime,
# plan_c2r_pass_runtime), the way a library such as an NFFT holds them: plans of any length in one
# typed vector, applied with mul!. Built by libfftl/probe/build.sh with juliac --trim=safe; the
# build fails if any call on these paths is unresolved.

using FFTL
using FFTL: plan_pass_runtime, plan_r2c_pass_runtime, plan_c2r_pass_runtime
using LinearAlgebra

# Complex passes along every dimension of an n1×n2×n3 array; the result is the Frobenius norm of the
# transformed array, so that the work cannot be optimised away.
function complex_passes(::Type{T}, n1::Int, n2::Int, n3::Int, nt::Int) where {T<:FFTL.FFTLFloat}
  x = ones(Complex{T}, n1, n2, n3)
  plans = [plan_pass_runtime(x, d; sign = -1, threads = nt) for d in 1:3]
  for p in plans
    mul!(x, p, x)
  end
  sqrt(sum(abs2, x))
end

# r2c along dimension 1, then c2r back.
function real_passes(::Type{T}, n1::Int, n2::Int, nt::Int) where {T<:FFTL.FFTLFloat}
  x = ones(T, n1, n2)
  f = plan_r2c_pass_runtime(x, 1; threads = nt)
  y = zeros(Complex{T}, n1 ÷ 2 + 1, n2)
  mul!(y, f, x)
  b = plan_c2r_pass_runtime(y, 1; n = n1, threads = nt)
  mul!(x, b, y)
  sqrt(sum(abs2, x))
end

Base.@ccallable function fftl_probe_complex64(n1::Cint, n2::Cint, n3::Cint, nt::Cint)::Cdouble
  complex_passes(Float64, Int(n1), Int(n2), Int(n3), Int(nt))
end

Base.@ccallable function fftl_probe_complex32(n1::Cint, n2::Cint, n3::Cint, nt::Cint)::Cdouble
  Float64(complex_passes(Float32, Int(n1), Int(n2), Int(n3), Int(nt)))
end

Base.@ccallable function fftl_probe_real64(n1::Cint, n2::Cint, nt::Cint)::Cdouble
  real_passes(Float64, Int(n1), Int(n2), Int(nt))
end

Base.@ccallable function fftl_probe_real32(n1::Cint, n2::Cint, nt::Cint)::Cdouble
  Float64(real_passes(Float32, Int(n1), Int(n2), Int(nt)))
end
