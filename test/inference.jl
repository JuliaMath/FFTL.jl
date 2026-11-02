# Inference and allocation requirements that phase 2 must guarantee (T1 in notes/coordination.md):
# with one warm-up call, the execution of a concrete plan is inferable and allocation-free, and
# plan construction becomes inferable through the runtime plan type. Execution is wrapped as
# `@test (@inferred ...) === result` so that it is counted (a bare @inferred is silent on pass
# in this Julia). Plan construction carries the radix sequence in its type today; these cases
# are @test_broken and flip when inference succeeds with the runtime plan — recorded as an
# error ("unexpected pass"), the signal to turn them into @test. Threaded execution must stay
# inferable; allocation freedom is deliberately not required there (task spawning). All cases
# go through public types and the small sizes of the suites above, so the file survives the
# rework unchanged.

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
    p = FFTL.plan_pass(v, 2; sign = -1, threads = 1)
    FFTL.apply!(p, v)
    @test (@inferred FFTL.apply!(p, v)) === v
    @test (@allocated FFTL.apply!(p, v)) == 0
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
    @test (@allocated FFTL.apply!(pnz, u)) == 0
    pne = FFTL.plan_pass(u, 1; sign = 1, threads = 1, needed = rng)
    FFTL.apply!(pne, u)
    @test (@inferred FFTL.apply!(pne, u)) === u
    @test (@allocated FFTL.apply!(pne, u)) == 0
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
# inferable, execution is allocation-free after a warm-up call.
@testset "inference and allocations, run-time plans, $T" for T in (Float64, Float32)
  let sz = [16, 12], x = randn(Complex{T}, 16, 12), y = similar(x)
    p = @inferred FFTL.RPlan{T}(sz, [1, 2], -1; threads = 1)
    GC.@preserve x y FFTL.execute!(p, pointer(y), pointer(x))
    @test (GC.@preserve x y @inferred FFTL.execute!(p, pointer(y), pointer(x))) === nothing
    @test (@allocated GC.@preserve x y FFTL.execute!(p, pointer(y), pointer(x))) == 0
  end
  let p = @inferred FFTL.RPlan{T}([37], [1], 1; threads = 1), x = randn(Complex{T}, 37), y = similar(x)   # Bluestein
    GC.@preserve x y FFTL.execute!(p, pointer(y), pointer(x))
    @test (@allocated GC.@preserve x y FFTL.execute!(p, pointer(y), pointer(x))) == 0
  end
  let x = randn(T, 16, 6), X = Matrix{Complex{T}}(undef, 9, 6), z = similar(x)
    pf = @inferred FFTL.RRFFTPlan{T}([16, 6], [1, 2], -1; threads = 1)
    pb = @inferred FFTL.RRFFTPlan{T}([16, 6], [1, 2], 1; threads = 1)
    GC.@preserve x X z (FFTL.execute_r2c!(pf, pointer(X), pointer(x)); FFTL.execute_c2r!(pb, pointer(z), pointer(X)))
    @test (GC.@preserve x X @inferred FFTL.execute_r2c!(pf, pointer(X), pointer(x))) === nothing
    @test (@allocated GC.@preserve x X FFTL.execute_r2c!(pf, pointer(X), pointer(x))) == 0
    @test (@allocated GC.@preserve X z FFTL.execute_c2r!(pb, pointer(z), pointer(X))) == 0
  end
  let x = randn(T, 9, 8), y = similar(x)
    p = @inferred FFTL.RR2RPlan{T}([9, 8], [FFTL.REDFT10, FFTL.HC2R], [1, 2]; threads = 1)
    GC.@preserve x y FFTL.execute_r2r!(p, pointer(y), pointer(x))
    @test (GC.@preserve x y @inferred FFTL.execute_r2r!(p, pointer(y), pointer(x))) === nothing
    @test (@allocated GC.@preserve x y FFTL.execute_r2r!(p, pointer(y), pointer(x))) == 0
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
