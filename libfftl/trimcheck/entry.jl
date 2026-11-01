# Trim check for FFTL's public plan/execute path (T1 in notes/coordination.md), built with
# juliac --trim=safe: one entry point per case so the verifier errors can be attributed, and
# two isolation cases whose plan is built at build time (top-level code is not verified, so
# fftl_trim_exec_* counts only the execution path and fftl_trim_plan_* only the construction
# path). Expected today (findings 1 and 4 in notes/libfftl.md): the construction does not
# resolve (radix sequence in the plan type, runtime Core.apply_type), and the execution
# through run! does not either (its generated body calls the @generated stage codelets);
# the stage codelets themselves resolve. Phase 2 target: 0 errors through RPlan.

using FFTL
using LinearAlgebra

# plan and execute in one entry: the default FFTL flow for a caller
Base.@ccallable function fftl_trim_1d_f64(n::Cint)::Cint
  x = Vector{ComplexF64}(undef, n)
  y = similar(x)
  p = FFTL.plan_fft(x)
  mul!(y, p, x)
  0
end

Base.@ccallable function fftl_trim_1d_f32(n::Cint)::Cint
  x = Vector{ComplexF32}(undef, n)
  y = similar(x)
  p = FFTL.plan_fft(x)
  mul!(y, p, x)
  0
end

Base.@ccallable function fftl_trim_2d_f64()::Cint
  x = Array{ComplexF64}(undef, 8, 12)
  y = similar(x)
  p = FFTL.plan_fft(x, (1, 2))
  mul!(y, p, x)
  0
end

Base.@ccallable function fftl_trim_rfft_f64(n::Cint)::Cint
  x = Vector{Float64}(undef, n)
  y = Vector{ComplexF64}(undef, n ÷ 2 + 1)
  p = FFTL.plan_rfft(x)
  mul!(y, p, x)
  0
end

Base.@ccallable function fftl_trim_r2r_f64(n::Cint)::Cint
  x = Vector{Float64}(undef, n)
  y = similar(x)
  p = FFTL.plan_r2r(x, FFTL.REDFT00)
  mul!(y, p, x)
  0
end

# construction only: the plan is used, the execution is kept out of the entry
Base.@ccallable function fftl_trim_plan_f64(n::Cint)::Cint
  p = FFTL.plan_fft(Vector{ComplexF64}(undef, n))
  Cint(p.nt)
end

# execution only: the plan comes from build-time top-level code, which the verifier skips
const P_EXEC = FFTL.plan_fft(Vector{ComplexF64}(undef, 128))
const X_EXEC = Vector{ComplexF64}(undef, 128)
const Y_EXEC = similar(X_EXEC)

Base.@ccallable function fftl_trim_exec_f64()::Cint
  mul!(Y_EXEC, P_EXEC, X_EXEC)
  0
end

# threaded execution only (Claude Code's review point 4 of fftl.h): the plan comes from
# build-time top-level code with threads = 2, the entry runs the spawn path. The runtime of
# the trimmed library starts with one Julia thread (--jl-option threads=1); the check is
# static, whether Threads.@spawn and friends resolve.
const P_THR = FFTL.plan_fft(Array{ComplexF64}(undef, 33, 1024); threads = 2)
const X_THR = Array{ComplexF64}(undef, 33, 1024)
const Y_THR = similar(X_THR)

Base.@ccallable function fftl_trim_exec_threaded_f64()::Cint
  mul!(Y_THR, P_THR, X_THR)
  0
end
