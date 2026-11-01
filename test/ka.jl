# KernelAbstractions extension on KA's CPU backend (POCL): the same generated kernels as on GPUs.
using KernelAbstractions

@testset "KernelAbstractions plans (CPU backend), $T" for T in (Float64, Float32)
  be = KernelAbstractions.CPU()
  # one pass, two passes (2^13), mixed radices, Bluestein (37, 101), batches, 2D/3D, partial regions
  for (dims, region) in (((64,), 1:1), ((1000,), 1:1), ((2^13,), 1:1), ((37,), 1:1), ((101,), 1:1),
                         ((16, 30), 1:1), ((12, 10), 1:2), ((16, 9, 5), 1:3), ((8, 37), 2:2), ((6, 7, 8), 2:3))
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
