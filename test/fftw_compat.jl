# The interface under FFTW.jl's names, the scope rules and AbstractFFTs's generic functions.
# Needs FFTW.jl (test dependency); the test without FFTW runs first, in runtests.jl.
import AbstractFFTs

@eval using FFTW

@testset "Same results as FFTW.jl" begin
  for T in (Float32, Float64)
    x = randn(Complex{T}, 12, 10, 3)
    r = randn(T, 12, 10)
    for region in (1:3, 1, (1, 3), 2)
      @test FFTL.fft(x, region) ≈ FFTW.fft(x, region)
      @test FFTL.bfft(x, region) ≈ FFTW.bfft(x, region)
      @test FFTL.ifft(x, region) ≈ FFTW.ifft(x, region)
    end
    @test FFTL.fft(r) ≈ FFTW.fft(r)
    for region in (1:2, 1, 2)
      R = FFTW.rfft(r, region)
      @test FFTL.rfft(r, region) ≈ R
      @test FFTL.brfft(R, size(r, first(region)), region) ≈ FFTW.brfft(R, size(r, first(region)), region)
      @test FFTL.irfft(R, size(r, first(region)), region) ≈ r
    end
    # the halved dimension is the first entry of the region, not the smallest
    r3 = randn(T, 12, 10, 14)
    for region in ((2, 1), (2, 1, 3), (3, 1), (3, 2, 1))
      R = FFTW.rfft(r3, region)
      @test size(FFTL.rfft(r3, region)) == size(R)
      @test FFTL.rfft(r3, region) ≈ R
      @test FFTL.plan_rfft(r3, region) * r3 ≈ R
      @test FFTL.brfft(R, size(r3, first(region)), region) ≈ FFTW.brfft(R, size(r3, first(region)), region)
      @test FFTL.irfft(R, size(r3, first(region)), region) ≈ r3
      @test inv(FFTL.plan_rfft(r3, region)) * R ≈ r3
    end
    # plans: *, mul!, inv, \, keywords
    p = FFTL.plan_fft(x; flags = FFTL.MEASURE, timelimit = Inf, num_threads = 2)
    @test p * x ≈ FFTW.fft(x)
    y = similar(x); mul!(y, p, x)
    @test y ≈ FFTW.fft(x)
    @test inv(p) * y ≈ x
    @test p \ y ≈ x
    @test FFTL.plan_ifft(x) * x ≈ FFTW.ifft(x)
    @test FFTL.plan_irfft(FFTW.rfft(r), 12) * FFTW.rfft(r) ≈ r
    # in place: the result overwrites the argument, which is returned
    for (f!, g) in ((FFTL.fft!, FFTW.fft), (FFTL.bfft!, FFTW.bfft), (FFTL.ifft!, FFTW.ifft))
      z = copy(x)
      @test f!(z) === z
      @test z ≈ g(x)
    end
    for (pf!, g) in ((FFTL.plan_fft!, FFTW.fft), (FFTL.plan_bfft!, FFTW.bfft), (FFTL.plan_ifft!, FFTW.ifft))
      z = copy(x)
      q = pf!(z, (1, 2))
      @test q * z === z
      @test z ≈ g(x, (1, 2))
    end
    z = copy(x); q = FFTL.plan_fft!(z); q * z
    @test inv(q) * z ≈ x
    # DCT: functions, plans, in place
    for region in (1:2, 1, 2)
      @test FFTL.dct(r, region) ≈ FFTW.dct(r, region)
      @test FFTL.idct(r, region) ≈ FFTW.idct(r, region)
      @test FFTL.plan_dct(r, region) * r ≈ FFTW.dct(r, region)
      @test FFTL.plan_idct(r, region) * r ≈ FFTW.idct(r, region)
      @test inv(FFTL.plan_dct(r, region)) * FFTW.dct(r, region) ≈ r
      z = copy(r); @test FFTL.dct!(z, region) === z; @test z ≈ FFTW.dct(r, region)
      z = copy(r); @test FFTL.idct!(z, region) === z; @test z ≈ FFTW.idct(r, region)
      z = copy(r); q = FFTL.plan_dct!(z, region); q * z
      @test z ≈ FFTW.dct(r, region)
    end
    # real-to-real: every kind, FFTW's definitions
    for k in (:REDFT00, :REDFT10, :REDFT01, :REDFT11, :RODFT00, :RODFT10, :RODFT01, :RODFT11, :DHT, :R2HC, :HC2R)
      @test FFTL.r2r(r, getfield(FFTL, k), 1) ≈ FFTW.r2r(r, getfield(FFTW, k), 1)
      @test FFTL.plan_r2r(r, getfield(FFTL, k), 2; flags = FFTL.ESTIMATE) * r ≈ FFTW.r2r(r, getfield(FFTW, k), 2)
    end
  end
end

@testset "Real-to-real transforms of complex data, in place and on strided arrays (as FFTW.jl)" begin
  kinds = (:REDFT00, :REDFT10, :REDFT01, :REDFT11, :RODFT00, :RODFT10, :RODFT01, :RODFT11, :DHT, :R2HC, :HC2R)
  for T in (Float32, Float64), nt in (1, 3)
    x = randn(Complex{T}, 12, 10, 3)
    r = real(x)
    for k in kinds, region in (1:3, 2, (3, 1))
      kl, kw = getfield(FFTL, k), getfield(FFTW, k)
      ref = FFTW.r2r(x, kw, region)
      @test FFTL.r2r(x, kl, region) ≈ ref
      p = FFTL.plan_r2r(x, kl, region; threads = nt)
      @test eltype(p) == Complex{T}
      y = similar(x); @test mul!(y, p, x) === y; @test y ≈ ref
      @test p \ ref ≈ x
      z = copy(x); @test FFTL.r2r!(z, kl, region) === z; @test z ≈ ref
      z = copy(x); q = FFTL.plan_r2r!(z, kl, region; threads = nt); @test q * z === z; @test z ≈ ref
      z = copy(r); @test FFTL.r2r!(z, kl, region) === z; @test z ≈ FFTW.r2r(r, kw, region)
    end
    @test FFTL.r2r(x, (FFTL.REDFT11, FFTL.DHT), (3, 1)) ≈ FFTW.r2r(x, (FFTW.REDFT11, FFTW.DHT), (3, 1))
    for region in (1:3, 1, (2, 3))
      @test FFTL.dct(x, region) ≈ FFTW.dct(x, region)
      @test FFTL.idct(x, region) ≈ FFTW.idct(x, region)
      @test FFTL.plan_dct(x, region) * x ≈ FFTW.dct(x, region)
      @test FFTL.plan_idct(x, region) * x ≈ FFTW.idct(x, region)
      @test FFTL.plan_dct(x, region) \ FFTW.dct(x, region) ≈ x
      y = similar(x); mul!(y, FFTL.plan_dct(x, region), x); @test y ≈ FFTW.dct(x, region)
      z = copy(x); @test FFTL.dct!(z, region) === z; @test z ≈ FFTW.dct(x, region)
      z = copy(x); q = FFTL.plan_idct!(z, region); @test q * z === z; @test z ≈ FFTW.idct(x, region)
    end
  end
  # mul! with views and reinterpreted arrays: dense ones directly, the others through a copy
  xc = randn(ComplexF32, 8, 6)
  xi = reinterpret(reshape, Float32, xc)
  out = zeros(Float32, 2, 8, 6)
  @test mul!(out, FFTL.plan_r2r(out, FFTL.REDFT10, 2:3), xi) === out
  @test out ≈ FFTW.r2r(copy(xi), FFTW.REDFT10, 2:3)
  A = randn(20, 30)
  for k in kinds, (vx, vy) in ((view(A, 3:12, 5:9), zeros(10, 5)),
                               (view(A, :, 2:6), view(zeros(20, 30), :, 1:5)),
                               (view(A, 3:12, 5:9), view(zeros(20, 30), 2:2:20, 1:5)),
                               (view(A, 2:2:20, 1:5), view(zeros(20, 30), 1:10, 21:25)))
    ref = FFTW.r2r(copy(vx), getfield(FFTW, k))
    p = FFTL.plan_r2r(copy(vx), getfield(FFTL, k))
    @test mul!(vy, p, vx) === vy
    @test vy ≈ ref
    @test p * vx ≈ ref
  end
  v = view(copy(A), 2:2:20, 1:5); w = copy(v)
  @test FFTL.r2r!(v, FFTL.RODFT10) === v; @test v ≈ FFTW.r2r(w, FFTW.RODFT10)
  vc = view(randn(ComplexF64, 10, 12), 1:2:10, 2:7)    # its real view is not strided
  @test FFTL.dct(vc) ≈ FFTW.dct(copy(vc))
  @test FFTL.r2r(vc, FFTL.RODFT00, 2) ≈ FFTW.r2r(copy(vc), FFTW.RODFT00, 2)
  # plan and data of different precision, real data into a complex plan
  x64 = randn(ComplexF64, 8, 6)
  p = FFTL.plan_r2r(ComplexF32.(x64), FFTL.REDFT10)
  @test p * x64 ≈ FFTW.r2r(x64, FFTW.REDFT10) rtol = 1e-5
  @test p * real(x64) ≈ FFTW.r2r(real(x64), FFTW.REDFT10) rtol = 1e-5
  # generic number types transform the two parts one after the other
  b = Complex{BigFloat}.(randn(ComplexF64, 9, 4))
  @test ComplexF64.(FFTL.dct(b)) ≈ FFTW.dct(ComplexF64.(b))
  @test ComplexF64.(FFTL.r2r(b, FFTL.RODFT11, 2)) ≈ FFTW.r2r(ComplexF64.(b), FFTW.RODFT11, 2)
  @test FFTL.idct(FFTL.dct(b)) ≈ b rtol = 1e-60
  # the region is checked against the complex array, not its real view
  @test_throws ArgumentError FFTL.plan_r2r(xc, FFTL.REDFT10, 0)
  @test_throws ArgumentError FFTL.plan_r2r(xc, FFTL.REDFT10, 3)
end

module UsesFFTW
  using FFTW
  plan(x) = plan_fft(x)
end
module UsesFFTL
  using FFTL
  plan(x) = plan_fft(x)
  transform(x) = fft(x)
end
module UsesBoth
  using FFTW, FFTL
end

@testset "Scopes and AbstractFFTs with FFTW loaded" begin
  x = randn(ComplexF64, 16, 8)
  # FFTL's functions are its own: a scope with `using FFTL` gets FFTL, whatever else is loaded
  @test FFTL.fft !== AbstractFFTs.fft
  @test FFTL.plan_fft !== AbstractFFTs.plan_fft
  @test UsesFFTL.plan(x) isa FFTL.FFTLPlan
  @test UsesFFTL.transform(x) ≈ FFTW.fft(x)
  @test !(UsesFFTW.plan(x) isa FFTL.FFTLPlan)
  # both in one scope: the unqualified name is not resolved, the qualified ones are
  @test_throws UndefVarError Core.eval(UsesBoth, :(fft($x)))
  @test UsesBoth.FFTL.plan_fft(x) isa FFTL.FFTLPlan
  # the generic functions: FFTW's methods are more specific for its strided arrays, FFTL takes
  # the other arrays and number types
  @test !(AbstractFFTs.plan_fft(x) isa FFTL.FFTLPlan)
  v = view(x, [1, 3, 5, 7], :)
  @test AbstractFFTs.plan_fft(v) isa FFTL.FFTLPlan
  @test AbstractFFTs.fft(v) ≈ FFTW.fft(collect(v))
  b = big.(x[1:6, 1])
  @test AbstractFFTs.plan_fft(b) isa FFTL.GPlan
  @test AbstractFFTs.rfft(real(b)) ≈ FFTL.rfft(real(b))
  # no ambiguities between FFTL's methods and FFTW's or AbstractFFTs's
  amb = Test.detect_ambiguities(FFTL, FFTW, AbstractFFTs)
  @test isempty(filter(a -> a[1].module === FFTL || a[2].module === FFTL, amb))
end

@testset "set_num_threads" begin
  x = randn(ComplexF64, 64)
  try
    FFTL.set_num_threads(3)
    @test FFTL.plan_fft(x).nthreads == 3
    @test FFTL.plan_fft(x; threads = 1).nthreads == 1
    @test FFTL.plan_fft(x; num_threads = 2).nthreads == 2
  finally
    FFTL.DEFAULT_THREADS[] = 0
  end
  @test FFTL.plan_fft(x).nthreads == Threads.nthreads()
end
