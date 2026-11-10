# KernelAbstractions extension on KA's CPU backend (POCL): the same generated kernels as on GPUs.
using KernelAbstractions

@testset "KernelAbstractions plans (CPU backend), $T" for T in pick((Float64, Float32), (Float64,))
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
  # Bluestein in one kernel (8 lines of M = 80 per workgroup) and in two launches (one line)
  @test length(FFTL.plan_ka(be, T, (8, 37), -1; region = 2:2).steps) == 1
  @test length(FFTL.plan_ka(be, T, (37,), -1).steps) == 2
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

@testset "KernelAbstractions real transforms (CPU backend), $T" for T in pick((Float64, Float32), (Float64,))
  be = KernelAbstractions.CPU()
  for (sz, reg) in pick((((16,), 1:1), ((15,), 1:1), ((2,), 1:1), ((37,), 1:1), ((12, 10), 1:2), ((12, 10), 2:2), ((9, 8, 6), (1, 3))),
                        (((16,), 1:1), ((15,), 1:1), ((12, 10), 1:2), ((9, 8, 6), (1, 3))))
    x = randn(T, sz)
    X = FFTL.plan_ka_rfft(be, T, sz; region = reg) * x; KernelAbstractions.synchronize(be)
    @test relerr(X, FFTL.rfft(x, reg)) < tol(T)
    Xc = copy(X)
    z = FFTL.plan_ka_brfft(be, T, sz; region = reg) * X; KernelAbstractions.synchronize(be)
    @test relerr(z, FFTL.brfft(Xc, sz[minimum(reg)], reg)) < tol(T)
    @test X == Xc
  end
  for k in instances(FFTL.R2RKind), n in pick((2, 5, 8, 33), (5, 8))
    x = randn(T, n)
    y = FFTL.plan_ka_r2r(be, T, (n,), k) * x; KernelAbstractions.synchronize(be)
    @test relerr(y, FFTL.r2r(x, k)) < 10tol(T)
  end
  x = randn(T, 9, 12, 5); ks = (FFTL.HC2R, FFTL.REDFT00, FFTL.RODFT11)
  p = FFTL.plan_ka_r2r(be, T, size(x), ks; region = 1:3)
  y = p * x; KernelAbstractions.synchronize(be)
  @test relerr(y, FFTL.r2r(x, ks, 1:3)) < 10tol(T)
  @test relerr(inv(p) * y, x) < 10tol(T)
  # DCT/DST-II, -III, -IV of even length and DCT-I inside the DFT's passes (contiguous and strided
  # dimensions, one and two passes, mirrored lanes), in place; DCT-I of 4096 (odd s in the last
  # pass) falls back, and so do DCT-II/DST-II along a strided dimension of odd stride (their post
  # hook works on pairs of columns)
  for k in (FFTL.REDFT10, FFTL.RODFT10, FFTL.REDFT01, FFTL.RODFT01, FFTL.REDFT11, FFTL.RODFT11, FFTL.REDFT00),
      (sz, reg) in pick((((64,), 1), ((32, 4096), 2), ((64, 256), 2), ((16, 64, 32), 1:3), ((6, 10), 1:2), ((17, 4096), 2)), (((64,), 1), ((32, 4096), 2), ((6, 10), 1:2)))
    x = randn(T, sz)
    p = FFTL.plan_ka_r2r(be, T, sz, k; region = reg)
    stride = prod(sz[1:minimum(reg)-1])
    oddpair = k in (FFTL.REDFT10, FFTL.RODFT10) && stride > 1 && isodd(stride)
    k == FFTL.REDFT00 || @test all(d -> d[1] === (oddpair ? :half : :fused), p.dims)
    y = p * x; KernelAbstractions.synchronize(be)
    @test relerr(y, FFTL.r2r(x, k, reg)) < 10tol(T)
    z = copy(x); mul!(z, p, z); KernelAbstractions.synchronize(be)
    @test relerr(z, y) < 10tol(T)
  end
  # DCT-I with n - 1 = 37 (Bluestein) and DST-I with n + 1 = 37 or 3: complex DFT of n ∓ 1
  # between kernels of their own
  for (k, sz, reg) in ((FFTL.REDFT00, (38,), 1), (FFTL.REDFT00, (5, 38), 2), (FFTL.REDFT00, (38, 6), 1:2),
                       (FFTL.RODFT00, (36,), 1), (FFTL.RODFT00, (5, 36), 2), (FFTL.RODFT00, (36, 2), 1:2))
    x = randn(T, sz)
    p = FFTL.plan_ka_r2r(be, T, sz, k; region = reg)
    @test p.dims[1][1] === (k == FFTL.REDFT00 ? :dct1 : :dst1)
    y = p * x; KernelAbstractions.synchronize(be)
    @test relerr(y, FFTL.r2r(x, k, reg)) < 10tol(T)
    z = copy(x); mul!(z, p, z); KernelAbstractions.synchronize(be)
    @test relerr(z, y) < 10tol(T)
  end
end

@testset "KernelAbstractions plans: regions, singleton and padded dimensions" begin
  be = KernelAbstractions.CPU()
  x = rand(ComplexF32, 16, 1, 6)
  @test relerr(FFTL.plan_ka(be, Float32, size(x), -1) * x, dftn(x, 1:3)) < tol(Float32)    # length-1 dimension
  @test relerr(FFTL.plan_ka(be, Float32, size(x), -1; region = 3) * x, dftn(x, 3:3)) < tol(Float32)
  @test FFTL.plan_ka(be, Float32, size(x), -1; region = Int[]) * x == x
  p = FFTL.plan_ka(be, Float32, size(x), 1; region = [3, 3])
  @test FFTL.AbstractFFTs.fftdims(p) == (3,)
  # padded layouts: two plans differing only in dims[1] keep their own kernels
  a = rand(ComplexF32, 8, 4, 16); b = rand(ComplexF32, 4, 8, 16)
  pa = FFTL.plan_ka(be, Float32, size(a), -1; pad = 4)
  ya = pa * a
  FFTL.plan_ka(be, Float32, size(b), -1; pad = 4) * b
  @test pa * a == ya
  @test relerr(ya, dftn(a, 1:3)) < tol(Float32)
end

