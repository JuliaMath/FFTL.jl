# KernelAbstractions extension on KA's CPU backend (POCL): the same generated kernels as on GPUs.
using KernelAbstractions

@testset "KernelAbstractions plans (CPU backend), $T" for T in (Float64, Float32)
  be = KernelAbstractions.CPU()
  # one pass, two passes (2^13), mixed radices, Bluestein (37, 101), batches, 2D/3D, partial regions
  for (dims, region) in pick((((64,), 1:1), ((1000,), 1:1), ((2^13,), 1:1), ((37,), 1:1), ((101,), 1:1),
                              ((16, 30), 1:1), ((12, 10), 1:2), ((16, 9, 5), 1:3), ((8, 37), 2:2), ((6, 7, 8), 2:3)),
                             thin(T, [((1000,), 1:1), ((2^13,), 1:1), ((37,), 1:1), ((16, 30), 1:1), ((16, 9, 5), 1:3), ((8, 37), 2:2)]))
    x = rand(Complex{T}, dims)
    p = FFTL.plan_ka(be, T, dims, -1; region)
    y = similar(x)
    mul!(y, p, x); KernelAbstractions.synchronize(be)
    @test relerr(y, dftn(x, region)) < tol(T)
    pb = FFTL.plan_ka(be, T, dims, 1; region)
    z = pb * y; KernelAbstractions.synchronize(be)
    @test relerr(z, prod(dims[d] for d in region) .* x) < tol(T)
    zi = inv(p) * y; KernelAbstractions.synchronize(be)
    @test relerr(zi, x) < tol(T)
  end
  p = FFTL.plan_ka(be, T, (8,), -1)
  @test_throws DimensionMismatch mul!(zeros(Complex{T}, 8), p, zeros(Complex{T}, 4))
  x = rand(Complex{T}, 8)
  @test_throws ArgumentError mul!(x, p, x)
end

@testset "KernelAbstractions pass rules under device limits" begin
  E = Base.get_extension(FFTL, :FFTLKernelAbstractionsExt)
  # Float64 on an RTX A6000: 48 KB static, 96 KB dynamic shared memory -> 3072 / 6144 values
  lim = E.Limits(3072, 6144, 1024, 48 << 20)
  for N in (2^12, 2^16, 2^20, 2^22, 1000, 3^9, 5 * 2^18)
    rs = E.default_passes(N, :generic, lim)
    @test prod(rs) == N && all(<=(lim.nmax), rs)
  end
  @test E.default_passes(4096, :generic, lim) == [64, 64]
  # small workgroups: 16 KB and 256 threads
  small = E.Limits(2048, 2048, 256, 8 << 20)
  for N in (2^12, 2^16, 2^20, 1000), prof in (:generic, :apple)
    rs = E.default_passes(N, prof, small)
    @test prod(rs) == N && all(<=(small.nstat), rs)
    s = 1
    for R in rs
      G = E.pass_G(R, s, N ÷ (R * s), 1, 1, 64, small)
      @test cld(G * R, maximum(E.local_radices(R))) <= small.wmax
      s *= R
    end
  end
  # the generated kernels run under these limits (CPU backend)
  be = KernelAbstractions.CPU()
  x = rand(ComplexF32, 2^14)
  p = FFTL.plan_ka(be, Float32, (2^14,), -1; lim = small)
  y = p * x; KernelAbstractions.synchronize(be)
  @test relerr(y, FFTL.plan_fft(ComplexF64.(x)) * ComplexF64.(x)) < tol(Float32)
end
