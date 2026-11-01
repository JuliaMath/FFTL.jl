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

  @testset "threads, $T" for T in (Float64, Float32)
    # sizes above the threading threshold: slices (dimension 1 of a 2D array), lane blocks
    # (last dimension), odd strides (last block with fewer lanes than the vector width)
    for sz in [(256, 256), (64, 64, 64), (33, 1024), (100, 100, 10), (8, 4096), (1024, 37), (2 * 3^11,), (5^7,)]
      x = randn(Complex{T}, sz)
      ref = FFTL.plan_fft(x; threads = 1) * x
      for nt in (2, 3, 4)
        p = FFTL.plan_fft(x; threads = nt)
        @test relerr(p * x, ref) < tol(T)
        z = copy(x); mul!(z, p, z)
        @test relerr(z, ref) < tol(T)
      end
    end
  end

  @testset "passes on strided views, $T" for T in (Float64, Float32)
    # views of a padded array as an NFFT uses them: interior boxes and pruned boxes
    G0 = randn(Complex{T}, 40, 36, 30)
    boxes = [(3:34, 2:33, 2:25), (3:6, 2:33, 2:25), (3:34, 5:9, 2:25), (1:40, 1:36, 1:30),
             (2:38, 2:36, 3:3), (5:40, 1:36, 4:28)]
    for b in boxes, d in 1:3, sign in (-1, 1), nt in (1, 3)
      G = copy(G0)
      v = view(G, b...)
      ref = mapslices(u -> dft(ComplexF64.(u), sign), Array(v); dims = d)
      p = FFTL.plan_pass(v, d; sign, threads = nt)
      FFTL.apply!(p, v)
      @test relerr(v, ref) < 10tol(T)
      mask = trues(size(G)); mask[b...] .= false
      @test G[mask] == G0[mask]
    end
    # out of place, Bluestein length along d, 2D
    H0 = randn(Complex{T}, 45, 41); H1 = similar(H0)
    for d in 1:2
      v = view(H0, 2:44, 3:39); w = view(H1, 2:44, 3:39)
      p = FFTL.plan_pass(v, d)
      mul!(w, p, v)
      @test relerr(w, mapslices(u -> dft(ComplexF64.(u)), Array(v); dims = d)) < 10tol(T)
    end
    @test_throws DimensionMismatch FFTL.apply!(FFTL.plan_pass(view(G0, 1:8, 1:8, 1:2), 2), view(G0, 1:8, 1:9, 1:2))
    # one long line on several threads (four-step), inside a padded vector
    L0 = randn(Complex{T}, 2^17 + 8)
    for sign in (-1, 1)
      L = copy(L0); v = view(L, 5:2^17+4)
      ref = copy(v); FFTL.apply!(FFTL.plan_pass(ref, 1; sign), ref)
      p = FFTL.plan_pass(v, 1; sign, threads = 3)
      @test p.kernel isa FFTL.FourStep
      FFTL.apply!(p, v)
      @test relerr(v, ref) < 10tol(T)
      @test L[1:4] == L0[1:4] && L[end-3:end] == L0[end-3:end]
    end
  end

  @testset "pruned passes, $T" for T in (Float64, Float32)
    # NFFT pattern: N modes in a line of length n, the first N - N÷2 and the last N÷2 entries
    for (n, N) in [(128, 64), (100, 40), (96, 70), (240, 180), (1024, 600), (162, 120), (64, 10)], d in 1:2, nt in (1, 3)
      a = N - N ÷ 2; b = N ÷ 2
      rng = (1:a, n-b+1:n)
      dims = d == 1 ? (n + 4, 37) : (37, n + 4)
      idx(r) = d == 1 ? (2 .+ r, 3:35) : (3:35, 2 .+ r)
      G0 = randn(Complex{T}, dims)
      line_zero = trues(n); for r in rng; line_zero[r] .= false; end
      for sign in (-1, 1)
        # zero inputs outside the ranges
        G = copy(G0); v = view(G, idx(1:n)...)
        sel = d == 1 ? (findall(line_zero), :) : (:, findall(line_zero))
        v[sel...] .= 0
        ref = mapslices(u -> dft(ComplexF64.(u), sign), Array(v); dims = d)
        FFTL.apply!(FFTL.plan_pass(v, d; sign, threads = nt, nonzero = rng), v)
        @test relerr(v, ref) < 10tol(T)
        # only the outputs in the ranges are needed
        G = copy(G0); v = view(G, idx(1:n)...)
        ref = mapslices(u -> dft(ComplexF64.(u), sign), Array(v); dims = d)
        FFTL.apply!(FFTL.plan_pass(v, d; sign, threads = nt, needed = rng), v)
        keep = d == 1 ? (findall(.!line_zero), :) : (:, findall(.!line_zero))
        @test relerr(v[keep...], ref[keep...]) < 10tol(T)
      end
    end
  end

  @testset "real transforms, $T" for T in (Float64, Float32)
    for (sz, region) in [((8,), 1), ((9,), 1), ((10,), 1), ((15,), 1), ((16,), 1), ((37,), 1), ((100,), 1),
                         ((101,), 1), ((1024,), 1), ((1023,), 1), ((3 * 2^11,), 1),
                         ((8, 6), (1, 2)), ((9, 6), (1, 2)), ((7, 10), (1, 2)), ((8, 6), (2,)), ((12, 7), (2,)),
                         ((6, 8, 5), (1, 2, 3)), ((6, 8, 5), (2, 3)), ((6, 9, 5), (2, 3)), ((5, 4, 6), (3,)), ((16, 9), (1,))]
      x = randn(T, sz)
      reg = sort([region...]); d1 = reg[1]; n = sz[d1]; h = n ÷ 2 + 1
      full = dftn(x, reg)
      sel = ntuple(i -> i == d1 ? (1:h) : (:), length(sz))
      p = FFTL.plan_rfft(x, region)
      y = p * x
      @test size(y) == Base.setindex(sz, h, d1)
      @test relerr(y, full[sel...]) < 10tol(T)
      @test relerr(FFTL.irfft(y, n, region), x) < 10tol(T)
      # brfft of a half spectrum of a real array gives the array times the transform size
      @test relerr(FFTL.brfft(y, n, region), prod(sz[i] for i in reg) .* x) < 10tol(T)
      # adjoint: <y, p x> = <p' y, x>
      yr = randn(Complex{T}, size(y))
      @test real(dot(yr, p * x)) ≈ dot(adjoint(p) * yr, x) rtol = 100tol(T)
    end
    # threads, large lines and strided lines
    for (sz, region) in [((2^16, 4), (1, 2)), ((64, 513), (2,)), ((1000, 300), (1, 2))]
      x = randn(T, sz)
      @test relerr(FFTL.plan_rfft(x, region; threads = 4) * x, FFTL.plan_rfft(x, region; threads = 1) * x) < tol(T)
    end
  end

  # real-to-real kinds by their definitions
  function r2r_ref(x, kind)
    n = length(x); y = zeros(Float64, n)
    for k in 0:n-1
      y[k+1] = if kind == FFTL.REDFT00
        x[1] + (-1)^k * x[n] + 2sum((x[j+1] * cospi(j * k / (n - 1)) for j in 1:n-2); init = 0.0)
      elseif kind == FFTL.REDFT10
        2sum(x[j+1] * cospi((j + 0.5) * k / n) for j in 0:n-1)
      elseif kind == FFTL.REDFT01
        x[1] + 2sum((x[j+1] * cospi(j * (k + 0.5) / n) for j in 1:n-1); init = 0.0)
      elseif kind == FFTL.REDFT11
        2sum(x[j+1] * cospi((j + 0.5) * (k + 0.5) / n) for j in 0:n-1)
      elseif kind == FFTL.RODFT00
        2sum(x[j+1] * sinpi((j + 1) * (k + 1) / (n + 1)) for j in 0:n-1)
      elseif kind == FFTL.RODFT10
        2sum(x[j+1] * sinpi((j + 0.5) * (k + 1) / n) for j in 0:n-1)
      elseif kind == FFTL.RODFT01
        (-1)^k * x[n] + 2sum((x[j+1] * sinpi((j + 1) * (k + 0.5) / n) for j in 0:n-2); init = 0.0)
      elseif kind == FFTL.RODFT11
        2sum(x[j+1] * sinpi((j + 0.5) * (k + 0.5) / n) for j in 0:n-1)
      else
        sum(x[j+1] * (cospi(2j * k / n) + sinpi(2j * k / n)) for j in 0:n-1)
      end
    end
    y
  end
  r2rkinds = (FFTL.REDFT00, FFTL.REDFT10, FFTL.REDFT01, FFTL.REDFT11, FFTL.RODFT00, FFTL.RODFT10,
              FFTL.RODFT01, FFTL.RODFT11, FFTL.DHT)

  @testset "real-to-real, $T" for T in (Float64, Float32)
    for kind in r2rkinds, n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 31, 32, 33, 100, 101, 128]
      kind == FFTL.REDFT00 && n < 2 && continue
      x = randn(T, n)
      @test relerr(FFTL.r2r(x, kind), r2r_ref(Float64.(x), kind)) < 10tol(T)
      p = FFTL.plan_r2r(x, kind)
      @test relerr(inv(p) * (p * x), x) < 10tol(T)
    end
    # several dimensions with mixed kinds, strided dimensions, threads
    x = randn(T, 9, 12, 5)
    kinds = (FFTL.REDFT10, FFTL.RODFT11, FFTL.DHT)
    ref = Float64.(x)
    for (d, k) in zip(1:3, kinds)
      ref = mapslices(u -> r2r_ref(u, k), ref; dims = d)
    end
    for nt in (1, 3)
      p = FFTL.plan_r2r(x, kinds, 1:3; threads = nt)
      @test relerr(p * x, ref) < 10tol(T)
      @test relerr(inv(p) * (p * x), x) < 10tol(T)
    end
    @test relerr(FFTL.r2r(x, FFTL.REDFT00, (2,)), mapslices(u -> r2r_ref(u, FFTL.REDFT00), Float64.(x); dims = 2)) < 10tol(T)
    # pass plans on views of a padded array, all kinds, every dimension, in place and out of place
    G0 = randn(T, 26, 21, 13)
    for kind in r2rkinds, d in 1:3, nt in (1, 2)
      b = (3:22, 2:18, 2:11)
      G = copy(G0); v = view(G, b...)
      ref = mapslices(u -> r2r_ref(u, kind), Float64.(Array(v)); dims = d)
      p = FFTL.plan_r2r_pass(v, d, kind; threads = nt)
      FFTL.apply!(p, v)
      @test relerr(v, ref) < 10tol(T)
      mask = trues(size(G)); mask[b...] .= false
      @test G[mask] == G0[mask]
    end
    # NFCT/NFST lengths: DCT-I with Ñ + 1 points, DST-I with Ñ - 1, Ñ smooth, even and odd
    for Ñ in (96, 135, 250, 324), (kind, n) in ((FFTL.REDFT00, Ñ + 1), (FFTL.RODFT00, Ñ - 1))
      H0 = randn(T, n + 4, 7); H1 = similar(H0)
      v = view(H0, 3:n+2, 2:6); w = view(H1, 3:n+2, 2:6)
      mul!(w, FFTL.plan_r2r_pass(v, 1, kind), v)
      @test relerr(w, mapslices(u -> r2r_ref(u, kind), Float64.(Array(v)); dims = 1)) < 10tol(T)
    end
    # orthonormal DCT
    y = FFTL.dct(x)
    @test sum(abs2, y) ≈ sum(abs2, x) rtol = 10tol(T)
    @test relerr(FFTL.idct(y), x) < 10tol(T)
    @test relerr(FFTL.idct(FFTL.dct(x, 2), 2), x) < 10tol(T)
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
