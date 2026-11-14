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
