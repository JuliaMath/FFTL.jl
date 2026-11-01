# Trim check for FFTL's public plan/execute path and the run-time plan path (RChain / RPlan),
# built with juliac --trim=safe: one entry point per case so the verifier errors can be
# attributed, and isolation cases whose plan is built at build time (top-level code is not
# verified, so fftl_trim_exec_* counts only the execution path and fftl_trim_plan_* only the
# construction path).
#
# Two groups of entries:
# 1. The public API of the package (plan_fft & friends, type-specialised chains in the plan
#    type). Baseline of the T1 report (142 errors on the Mac, Julia 1.13.2): construction and
#    the run! dispatch layer do not resolve; phase 2 does not change this path.
# 2. The run-time path that libfftl binds (RChain / RPlan, phase 2): the target is 0 errors,
#    and the boundary (src/libfftl.jl) builds around exactly these calls. fftl_trim_first_group
#    isolates the one codelet family that the phase-3 build found unresolved (see
#    REPORT-phase3.md and the Requests in notes/coordination.md).

using FFTL
using FFTL: RChain, run!
using LinearAlgebra
using SIMD

# --- 1. public plan/execute path (unchanged by phase 2; the T1 baseline) -------------

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
  Cint(p.nthreads)
end

# execution only: the plan comes from build-time top-level code, which the verifier skips
const P_EXEC = FFTL.plan_fft(Vector{ComplexF64}(undef, 128))
const X_EXEC = Vector{ComplexF64}(undef, 128)
const Y_EXEC = similar(X_EXEC)

Base.@ccallable function fftl_trim_exec_f64()::Cint
  mul!(Y_EXEC, P_EXEC, X_EXEC)
  0
end

# threaded execution only (fftl.h review point 4): the plan comes from build-time top-level
# code with threads = 2, the entry runs the spawn path.
const P_THR = FFTL.plan_fft(Array{ComplexF64}(undef, 33, 1024); threads = 2)
const X_THR = Array{ComplexF64}(undef, 33, 1024)
const Y_THR = similar(X_THR)

Base.@ccallable function fftl_trim_exec_threaded_f64()::Cint
  mul!(Y_THR, P_THR, X_THR)
  0
end

# --- 2. run-time plan path (phase 2: RChain / RPlan; what libfftl calls) ----------------

# RChain construction + execution in one entry (256 = (4, 8, 8), grouped first stage expected)
Base.@ccallable function fftl_trim_rchain_f64(n::Cint)::Cint
  c = RChain{Float64}(Int(n), -1)
  x = Vector{ComplexF64}(undef, n)
  y = similar(x)
  w1 = similar(x)
  w2 = similar(x)
  GC.@preserve x y w1 w2 c begin
    run!(Ptr{Float64}(pointer(y)), Ptr{Float64}(pointer(x)), Ptr{Float64}(pointer(w1)),
         Ptr{Float64}(pointer(w2)), c, 1)
  end
  0
end

# RPlan construction + execution in one entry: the exact call pattern of fftl_plan_dft and
# fftl_plan_execute (1D f64)
Base.@ccallable function fftl_trim_rplan_f64(n::Cint)::Cint
  x = Vector{ComplexF64}(undef, n)
  y = similar(x)
  p = FFTL.RPlan{Float64}([Int(n)], [1], -1)
  GC.@preserve x y FFTL.execute!(p, pointer(y), pointer(x))
  0
end

# RPlan construction + execution, Bluestein length (97 prime)
Base.@ccallable function fftl_trim_rplan_bluestein_f64(n::Cint)::Cint
  x = Vector{ComplexF64}(undef, n)
  y = similar(x)
  p = FFTL.RPlan{Float64}([Int(n)], [1], -1)
  GC.@preserve x y FFTL.execute!(p, pointer(y), pointer(x))
  0
end

# RPlan 2D over both dimensions
Base.@ccallable function fftl_trim_rplan2d_f64()::Cint
  x = Array{ComplexF64}(undef, 8, 12)
  y = similar(x)
  p = FFTL.RPlan{Float64}([8, 12], [1, 2], -1)
  GC.@preserve x y FFTL.execute!(p, pointer(y), pointer(x))
  0
end

# RPlan construction only
Base.@ccallable function fftl_trim_rplan_plan_f64(n::Cint)::Cint
  p = FFTL.RPlan{Float64}([Int(n)], [1], -1)
  Cint(length(p.sz) + p.nthreads)
end

# RPlan execution only: plan from build-time top-level code (unverified), the entry counts
# only the execution path (chain + Bluestein buffers)
const RP_EXEC = FFTL.RPlan{Float64}([1000], [1], -1)
const RPX_EXEC = Vector{ComplexF64}(undef, 1000)
const RPY_EXEC = similar(RPX_EXEC)

Base.@ccallable function fftl_trim_rplan_exec_f64()::Cint
  GC.@preserve RPX_EXEC RPY_EXEC FFTL.execute!(RP_EXEC, pointer(RPY_EXEC), pointer(RPX_EXEC))
  0
end

# RPlan threaded execution: plan with threads = 4 from build-time top-level code; the entry
# runs the spawn path of parallel_items / Threads.@spawn (fftl.h review point 4 for the
# run-time plan path).
const RP_THRP = FFTL.RPlan{Float64}([33, 1024], [1, 2], -1; threads = 4)
const RPTH_X = Array{ComplexF64}(undef, 33, 1024)
const RPTH_Y = similar(RPTH_X)

Base.@ccallable function fftl_trim_rplan_exec_threaded_f64()::Cint
  GC.@preserve RPTH_X RPTH_Y FFTL.execute!(RP_THRP, pointer(RPTH_Y), pointer(RPTH_X))
  0
end

# --- 3. codelet isolation (diagnosis of the stage_first_group! failure) ----------------

# The grouped first stage called directly from Main-verified code, like the phase-0 flat
# runners did: if this resolves, the forwarder itself is fine and the phase-3 failure comes
# from its call through FFTL.RChain's rstage_group!.
Base.@ccallable function fftl_trim_first_group_f64(m::Cint)::Cint
  x = Vector{ComplexF64}(undef, 4 * Int(m))
  y = similar(x)
  pt = Vector{ComplexF64}(undef, 3 * Int(m))
  GC.@preserve x y pt begin
    FFTL.stage_first_group!(Ptr{Float64}(pointer(y)), Ptr{Float64}(pointer(x)),
                            Ptr{Float64}(pointer(pt)), Int(m), Val(4), Val(-1),
                            SIMD.Vec{4, Float64})
  end
  0
end

# --- 4. Bluestein isolation (diagnosis: construction vs execution of the RChain-based
# Bluestein kernel; n must be a non-smooth length, 97 is prime) -----------------------

# Bluestein construction inside verified code (RChain inner chains, cached chirp and
# filter tables; the filter transform runs run! on the forward inner chain)
Base.@ccallable function fftl_trim_bluestein_ctor_f64()::Cint
  k = FFTL.Bluestein{Float64}(97, -1, RChain)
  Cint(k.n + length(k.chirp))
end

# A Bluestein kernel built at build time, executed from the entry (the execution path only:
# run! over the inner chains through run_lanes!)
const BLU_EXEC = FFTL.Bluestein{Float64}(97, -1, RChain)
const BLU_X = Vector{ComplexF64}(undef, 97)
const BLU_Y = similar(BLU_X)
const BLU_W1 = similar(BLU_X)
const BLU_W2 = similar(BLU_X)

Base.@ccallable function fftl_trim_bluestein_exec_f64()::Cint
  GC.@preserve BLU_X BLU_Y BLU_W1 BLU_W2 begin
    run!(Ptr{Float64}(pointer(BLU_Y)), Ptr{Float64}(pointer(BLU_X)), Ptr{Float64}(pointer(BLU_W1)),
         Ptr{Float64}(pointer(BLU_W2)), BLU_EXEC, 1)
  end
  0
end

# RPlan construction only, smooth length (no Bluestein involved)
Base.@ccallable function fftl_trim_rplan_smooth_plan_f64()::Cint
  p = FFTL.RPlan{Float64}([256], [1], -1)
  Cint(length(p.sz) + p.nthreads)
end

# RPlan construction only, prime length (Bluestein path)
Base.@ccallable function fftl_trim_rplan_bluestein_plan_f64()::Cint
  p = FFTL.RPlan{Float64}([97], [1], -1)
  Cint(length(p.sz) + p.nthreads)
end

# The grouped first stage with a vector of ONE complex value (SIMD.Vec{2, Float64}), the type
# rstage_group! carries as the vi == n branches below its 2/4-complex group vectors. Runtime
# never picks it; the trim verifier walks the branch statically and the generated body fails.
Base.@ccallable function fftl_trim_first_group1c_f64(m::Cint)::Cint
  x = Vector{ComplexF64}(undef, 4 * Int(m))
  y = similar(x)
  pt = Vector{ComplexF64}(undef, 3 * Int(m))
  GC.@preserve x y pt begin
    FFTL.stage_first_group!(Ptr{Float64}(pointer(y)), Ptr{Float64}(pointer(x)),
                            Ptr{Float64}(pointer(pt)), Int(m), Val(4), Val(-1),
                            SIMD.Vec{2, Float64})
  end
  0
end

# rstage_group! itself, all arguments run-time ints, from Main: the dispatch the run! body of
# RChain uses; the verifier walks every vi branch here.
Base.@ccallable function fftl_trim_rstage_group_f64(m::Cint)::Cint
  x = Vector{ComplexF64}(undef, 64 * Int(m))
  y = similar(x)
  pt = Vector{ComplexF64}(undef, 3 * Int(m))
  GC.@preserve x y pt begin
    FFTL.rstage_group!(Ptr{Float64}(pointer(y)), Ptr{Float64}(pointer(x)),
                       Ptr{Float64}(pointer(pt)), Int(m), 4, -1, 128)
  end
  0
end
