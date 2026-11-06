# Inference and allocations: after one warm-up call the execution of a plan is inferable and
# allocation-free. Execution is wrapped as `@test (@inferred ...) === result` so that it is
# counted (a bare @inferred is silent on pass). The JIT plans carry their radix sequence in the
# type, so their construction is not inferable (@test_broken: an unexpected pass is reported);
# the run-time plans (RPlan, RRFFTPlan, RR2RPlan) are inferable in construction and execution.
# Threaded execution must be inferable; allocation freedom is not required there (tasks).

@testset "inference and allocations, $T" for T in (Float64, Float32)
  # 1D chain, out of place and in place; construction of the same plan is broken
  let x = randn(Complex{T}, 128), y = similar(x)
    p = FFTL.plan_fft(x)
    mul!(y, p, x)                                  # warm-up: kernel compilation
    @test (@inferred mul!(y, p, x)) === y
    @test (@allocated mul!(y, p, x)) == 0
    mul!(x, p, x)                                  # warm-up of the in-place branch
    @test (@inferred mul!(x, p, x)) === x
    @test (@allocated mul!(x, p, x)) == 0
    @test_broken @inferred FFTL.plan_fft(x)
    @test_broken @inferred FFTL.plan_fft(x, (1, 2))
  end
  # Bluestein length, the plan holds the work buffers
  let x = randn(Complex{T}, 37), y = similar(x)
    p = FFTL.plan_fft(x)
    mul!(y, p, x)
    @test (@inferred mul!(y, p, x)) === y
    @test (@allocated mul!(y, p, x)) == 0
  end
  # 2D and 3D with all dimensions transformed, and a 2D plan over an interior dimension
  let x = randn(Complex{T}, 8, 12), y = similar(x)
    p = FFTL.plan_fft(x, (1, 2))
    mul!(y, p, x)
    @test (@inferred mul!(y, p, x)) === y
    @test (@allocated mul!(y, p, x)) == 0
  end
  let x = randn(Complex{T}, 4, 6, 10), y = similar(x)
    p = FFTL.plan_fft(x, (1, 2, 3))
    mul!(y, p, x)
    @test (@inferred mul!(y, p, x)) === y
    @test (@allocated mul!(y, p, x)) == 0
  end
  let x = randn(Complex{T}, 3, 8), y = similar(x)
    p = FFTL.plan_fft(x, (2,))
    mul!(y, p, x)
    @test (@inferred mul!(y, p, x)) === y
    @test (@allocated mul!(y, p, x)) == 0
  end
  # pass plans on views of a padded array: unpruned in place and out of place, pruned in place
  let G = randn(Complex{T}, 40, 36, 30), W = similar(G)
    v = view(G, 3:34, 2:33, 2:25)
    w = view(W, 3:34, 2:33, 2:25)
    # Julia 1.10: apply! on a view allocates 64-80 bytes per call (mul! with a separate output
    # does not); 1.11 and later are allocation-free.
    p = FFTL.plan_pass(v, 2; sign = -1, threads = 1)
    FFTL.apply!(p, v)
    @test (@inferred FFTL.apply!(p, v)) === v
    @test (@allocated FFTL.apply!(p, v)) == 0 broken = VERSION < v"1.11"
    mul!(w, p, v)
    @test (@inferred mul!(w, p, v)) === w
    @test (@allocated mul!(w, p, v)) == 0
    n = 64
    N = 10
    rng = (1:(N - N ÷ 2), (n - N ÷ 2 + 1):n)
    P = randn(Complex{T}, n + 4, 37)
    u = view(P, 2 .+ (1:n), 3:35)
    pnz = FFTL.plan_pass(u, 1; sign = -1, threads = 1, nonzero = rng)
    FFTL.apply!(pnz, u)
    @test (@inferred FFTL.apply!(pnz, u)) === u
    @test (@allocated FFTL.apply!(pnz, u)) == 0 broken = VERSION < v"1.11"
    pne = FFTL.plan_pass(u, 1; sign = 1, threads = 1, needed = rng)
    FFTL.apply!(pne, u)
    @test (@inferred FFTL.apply!(pne, u)) === u
    @test (@allocated FFTL.apply!(pne, u)) == 0 broken = VERSION < v"1.11"
    @test_broken @inferred FFTL.plan_pass(v, 2)
  end
  # real transforms
  let x = randn(T, 16), y = FFTL.plan_rfft(x) * x
    p = FFTL.plan_rfft(x)
    mul!(y, p, x)
    @test (@inferred mul!(y, p, x)) === y
    @test (@allocated mul!(y, p, x)) == 0
    @test_broken @inferred FFTL.plan_rfft(x)
  end
  let x = randn(T, 16), y = similar(x)
    p = FFTL.plan_r2r(x, FFTL.REDFT00)
    mul!(y, p, x)
    @test (@inferred mul!(y, p, x)) === y
    @test (@allocated mul!(y, p, x)) == 0
    @test_broken @inferred FFTL.plan_r2r(x, FFTL.REDFT00)
  end
  let x = randn(T, 9, 12, 5), y = similar(x)
    p = FFTL.plan_r2r(x, (FFTL.REDFT10, FFTL.RODFT11, FFTL.DHT), 1:3; threads = 1)
    mul!(y, p, x)
    @test (@inferred mul!(y, p, x)) === y
    @test (@allocated mul!(y, p, x)) == 0
  end
end

# Threaded execution: with more than one thread available, inference of the call is required
# (but not allocation freedom: the threads come from task spawning)
@testset "inference, threaded plans" begin
  if Threads.nthreads() > 1
    x = randn(ComplexF64, 33, 1024)                 # lines over the threading threshold
    y = similar(x)
    p = FFTL.plan_fft(x; threads = 2)
    mul!(y, p, x)
    @test (@inferred mul!(y, p, x)) === y
  end
end

# Run-time plans (the path compiled ahead of time for libfftl): construction and execution are
# inferable, execution is allocation-free after a warm-up call. The pointer calls go through
# functions: GC.@preserve inside @allocated at testset scope allocates by itself.
exe!(p, y, x) = GC.@preserve x y FFTL.execute!(p, pointer(y), pointer(x))
exe_r2c!(p, y, x) = GC.@preserve x y FFTL.execute_r2c!(p, pointer(y), pointer(x))
exe_c2r!(p, y, x) = GC.@preserve x y FFTL.execute_c2r!(p, pointer(y), pointer(x))
exe_r2r!(p, y, x) = GC.@preserve x y FFTL.execute_r2r!(p, pointer(y), pointer(x))

@testset "inference and allocations, run-time plans, $T" for T in (Float64, Float32)
  let sz = [16, 12], x = randn(Complex{T}, 16, 12), y = similar(x)
    p = @inferred FFTL.RPlan{T}(sz, [1, 2], -1; threads = 1)
    exe!(p, y, x)
    @test (@inferred exe!(p, y, x)) === nothing
    @test (@allocated exe!(p, y, x)) == 0
  end
  let p = (@inferred FFTL.RPlan{T}([37], [1], 1; threads = 1)), x = randn(Complex{T}, 37), y = similar(x)   # Bluestein
    exe!(p, y, x)
    @test (@allocated exe!(p, y, x)) == 0
  end
  let x = randn(T, 16, 6), X = Matrix{Complex{T}}(undef, 9, 6), z = similar(x)
    pf = @inferred FFTL.RRFFTPlan{T}([16, 6], [1, 2], -1; threads = 1)
    pb = @inferred FFTL.RRFFTPlan{T}([16, 6], [1, 2], 1; threads = 1)
    exe_r2c!(pf, X, x); exe_c2r!(pb, z, X)
    @test (@inferred exe_r2c!(pf, X, x)) === nothing
    @test (@allocated exe_r2c!(pf, X, x)) == 0
    @test (@allocated exe_c2r!(pb, z, X)) == 0
  end
  let x = randn(T, 9, 8), y = similar(x)
    p = @inferred FFTL.RR2RPlan{T}([9, 8], [FFTL.REDFT10, FFTL.HC2R], [1, 2]; threads = 1)
    exe_r2r!(p, y, x)
    @test (@inferred exe_r2r!(p, y, x)) === nothing
    @test (@allocated exe_r2r!(p, y, x)) == 0
  end
  # c2r and the inverse of a JIT plan
  let x = randn(T, 16), X = FFTL.plan_rfft(x) * x, z = similar(x)
    pb = FFTL.plan_brfft(X, 16)
    mul!(z, pb, X)
    @test (@inferred mul!(z, pb, X)) === z
    @test (@allocated mul!(z, pb, X)) == 0
  end
  let x = randn(Complex{T}, 64), y = similar(x)
    pi_ = inv(FFTL.plan_fft(x))
    mul!(y, pi_, x)
    @test (@inferred mul!(y, pi_, x)) === y
  end
end
