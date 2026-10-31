using FFTL, Test, LinearAlgebra

# Reference DFT in Float64 (BigFloat would be exact but slow; the tolerances below allow for it).
function dft(x, σ = -1)
  n = length(x)
  [sum(x[j+1] * cispi(2σ * mod(j * k, n) / n) for j in 0:n-1) for k in 0:n-1]
end

# Direct DFT along the dimensions in `region`, one after the other.
function dftn(x, region, σ = -1)
  y = ComplexF64.(x)
  for d in region
    y = mapslices(v -> dft(v, σ), y; dims = d)
  end
  y
end

relerr(a, b) = norm(a - b, Inf) / norm(b, Inf)
tol(::Type{Float64}) = 1e-13
tol(::Type{Float32}) = 2e-6

using SIMD: Vec

@testset "FFTL" begin
  @testset "complex vector operations" begin
    # Checks the active implementation (FCADD/FCMLA on aarch64) and the shuffle formulas used
    # where these instructions are missing.
    for (T, N) in ((Float64, 2), (Float32, 4), (Float32, 2))
      a, b, c = (Vec{N,T}(Tuple(randn(T, N))) for _ in 1:3)
      cx(v) = [complex(v[i], v[i+1]) for i in 1:2:N]
      @test cx(FFTL.fcadd90(a, b)) ≈ cx(a) .+ im .* cx(b)
      @test cx(FFTL.fcadd270(a, b)) ≈ cx(a) .- im .* cx(b)
      @test cx(FFTL.ccmul(a, b)) ≈ cx(a) .* cx(b)
      @test cx(a + FFTL.mul_i(b)) ≈ cx(a) .+ im .* cx(b)
      @test cx(muladd(FFTL.dup_re(a), b, c)) ≈ cx(c) .+ real.(cx(a)) .* cx(b)
      @test cx(muladd(FFTL.dup_im(a), FFTL.mul_i(b), muladd(FFTL.dup_re(a), b, zero(c)))) ≈ cx(a) .* cx(b)
    end
  end

  @testset "codelets and chains against the direct DFT, $T" for T in (Float64, Float32)
    for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 24, 32, 48, 49, 60, 64, 80, 100,
              105, 120, 128, 143, 169, 210, 243, 256, 360, 384, 500, 512, 1000, 1024,
              17, 19, 23, 29, 31, 68, 289, 928, 31 * 27]
      x = randn(Complex{T}, n)
      ref = dft(ComplexF64.(x))
      @test relerr(FFTL.fft(x), ref) < tol(T)
      @test relerr(FFTL.bfft(x), dft(ComplexF64.(x), 1)) < tol(T)
    end
  end

  @testset "Bluestein, $T" for T in (Float64, Float32)
    for n in [37, 41, 74, 101, 37^2, 1009]
      x = randn(Complex{T}, n)
      @test relerr(FFTL.fft(x), dft(ComplexF64.(x))) < 10tol(T)
      @test relerr(FFTL.bfft(x), dft(ComplexF64.(x), 1)) < 10tol(T)
    end
  end

  @testset "large lengths: inverse and Parseval, $T" for T in (Float64, Float32)
    for n in [4096, 3^8, 10^5, 2^18, 2 * 10007]
      x = randn(Complex{T}, n)
      y = FFTL.fft(x)
      @test relerr(FFTL.ifft(y), x) < 10tol(T)
      @test sum(abs2, y) / n ≈ sum(abs2, x) rtol = 10tol(T)
    end
  end

  @testset "multidimensional, $T" for T in (Float64, Float32)
    cases = [((8, 12), (1, 2)), ((16, 16), (1, 2)), ((17, 6), (1, 2)), ((5, 37), (1, 2)),
             ((3, 8), (2,)), ((3, 8), (1, 2)), ((12, 7), (1,)), ((1, 9), (1, 2)),
             ((4, 6, 10), (1, 2, 3)), ((3, 5, 7), (1, 2, 3)), ((6, 4, 5), (1, 3)), ((6, 4, 5), (2,)),
             ((16, 9, 2, 3), (1, 2, 3, 4))]
    for (sz, region) in cases
      x = randn(Complex{T}, sz)
      ref = dftn(x, region)
      @test relerr(FFTL.fft(x, region), ref) < 10tol(T)
      @test relerr(FFTL.bfft(x, region), dftn(x, region, 1)) < 10tol(T)
      p = FFTL.plan_fft(x, region)
      z = copy(x); mul!(z, p, z)
      @test relerr(z, ref) < 10tol(T)
      @test relerr(inv(p) * (p * x), x) < 10tol(T)
    end
    x = randn(Complex{T}, 64, 48, 20)
    @test relerr(FFTL.ifft(FFTL.fft(x)), x) < 10tol(T)
  end

  @testset "plans: in place, out of place, inv" begin
    for n in [1, 7, 64, 1000, 1009]
      x = randn(ComplexF64, n)
      p = FFTL.plan_fft(x)
      ref = p * x
      y = similar(x)
      mul!(y, p, x)
      @test y == ref
      z = copy(x)
      mul!(z, p, z)
      @test z ≈ ref
      @test inv(p) * ref ≈ x
      @test p \ ref ≈ x
      xv = view(copy(x), :)
      @test p * xv ≈ ref
    end
    @test_throws DimensionMismatch FFTL.plan_fft(zeros(ComplexF64, 8)) * zeros(ComplexF64, 9)
  end

  @testset "radix choice" begin
    for n in [2^k for k in 0:22]
      @test prod(FFTL.radices(n); init = 1) == n
    end
    for n in [1000, 3000, 10^6, 2^5 * 3^4 * 5^3 * 7 * 11 * 13]
      @test prod(FFTL.radices(n)) == n
    end
  end
end
