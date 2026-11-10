using FFTL, Test, LinearAlgebra

# Reference DFT in Float64 (BigFloat would be exact but slow; the tolerances below allow for it).
function dft(x, σ = -1)
  n = length(x)
  [sum(x[j+1] * cispi(2σ * mod(j * k, n) / n) for j in 0:n-1) for k in 0:n-1]
end

# The DFT at a few frequencies only, each in O(n): the full reference is O(n²), too slow for lengths
# of 10⁵ and more. The frequencies cover both ends, the middle and a spread in between.
function dft_at(x, σ = -1)
  n = length(x)
  ks = unique([0, 1, n ÷ 3, n ÷ 2, n - 2, n - 1, (7n) ÷ 11])
  ref = [sum(ComplexF64(x[j+1]) * cispi(2σ * mod(j * k, n) / n) for j in 0:n-1) for k in ks]
  ks, ref
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

# Same shapes with dense, column-block, padded, stepped and reversed layouts.
function realpass_layouts(::Type{T}, sz::NTuple{N,Int}) where {T,N}
  block = ntuple(i -> i == N ? (3:sz[i]+2) : (1:sz[i]), N)
  padded = ntuple(i -> 2:sz[i]+1, N)
  stepped = ntuple(i -> 2:2:2sz[i], N)
  reversed = ntuple(i -> sz[i]+1:-1:2, N)
  (fill(T(7), sz),
   view(fill(T(7), Base.setindex(sz, sz[N] + 4, N)), block...),
   view(fill(T(7), sz .+ 4), padded...),
   view(fill(T(7), 2 .* sz .+ 4), stepped...),
   view(fill(T(7), sz .+ 4), reversed...))
end

using SIMD: Vec

# Default: a quick run that covers every feature with fewer sizes (CI on every push).
# FFTL_TEST_FULL=true: the full lists.
const FULL = get(ENV, "FFTL_TEST_FULL", "false") == "true"
pick(full, quick) = FULL ? full : quick
# quick run: Float32 takes every other case (its kernels compile separately from Float64's)
thin(T, v) = FULL || T == Float64 ? v : v[1:2:end]

@testset "FFTL" verbose = get(ENV, "FFTL_TEST_VERBOSE", "false") == "true" begin
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
    for n in pick([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 24, 32, 48, 49, 60, 64, 80, 100,
                   105, 120, 128, 143, 169, 210, 243, 256, 360, 384, 500, 512, 1000, 1024,
                   17, 19, 23, 29, 31, 68, 289, 928, 31 * 27],
                  thin(T, [1, 2, 3, 4, 5, 7, 8, 11, 13, 16, 31, 60, 128, 243, 1000, 31 * 27]))
      x = randn(Complex{T}, n)
      ref = dft(ComplexF64.(x))
      @test relerr(FFTL.fft(x), ref) < tol(T)
      (FULL || n <= 16) && @test relerr(FFTL.bfft(x), dft(ComplexF64.(x), 1)) < tol(T)
    end
  end

  @testset "Bluestein, $T" for T in (Float64, Float32)
    for n in pick([37, 41, 74, 101, 37^2, 1009], [37, 74, 1009])
      x = randn(Complex{T}, n)
      @test relerr(FFTL.fft(x), dft(ComplexF64.(x))) < 10tol(T)
      @test relerr(FFTL.bfft(x), dft(ComplexF64.(x), 1)) < 10tol(T)
    end
  end

  @testset "smooth part times leftover, $T" for T in (Float64, Float32)
    # n = s·q: a chain of length s, Bluestein on the leftover q (prime or composite)
    for n in pick([2 * 37, 3 * 41, 16 * 43, 29 * 37, 64 * 37, 2 * 37^2, 5 * 13 * 37, 2 * 37 * 41, 6 * 1009,
                   2 * 37 * 41 * 43, 6 * 37^2 * 41],
                  [2 * 37, 29 * 37, 2 * 37 * 41, 6 * 1009])
      x = randn(Complex{T}, n)
      @test FFTL.plan_fft(x; threads = 1).kernels[1] isa FFTL.Composite
      @test relerr(FFTL.fft(x), dft(ComplexF64.(x))) < 10tol(T)
      @test relerr(FFTL.bfft(x), dft(ComplexF64.(x), 1)) < 10tol(T)
    end
    # a leftover split into prime powers (the planner does this for large leftovers only): first
    # stages that are composite and Bluestein kernels themselves
    for (n, s, q) in pick([(2 * 37 * 41 * 43, 2 * 37 * 41, 43), (37 * 41, 37, 41), (6 * 37^2 * 41, 6 * 37^2, 41)],
                          [(2 * 37 * 41 * 43, 2 * 37 * 41, 43), (37 * 41, 37, 41)])
      x = randn(Complex{T}, n)
      for σ in (-1, 1)
        k = FFTL.Composite{T}(n, s, q, σ, 1)
        y = similar(x); w1 = similar(x); w2 = similar(x)
        GC.@preserve x y w1 w2 FFTL.run!(FFTL.cptr(y), FFTL.cptr(x), FFTL.cptr(w1), FFTL.cptr(w2), k, 1)
        ks, ref = dft_at(ComplexF64.(x), σ)
        @test relerr(y[ks .+ 1], ref) < 10tol(T)
      end
    end
    # lanes of a 2D array: along dimension 1 (contiguous lines) and dimension 2 (strided lanes)
    for (sz, region) in ((((2 * 37, 5)), (1,)), ((3, 2 * 37), (2,)), ((5, 3 * 41), (1, 2)))
      x = randn(Complex{T}, sz)
      @test relerr(FFTL.fft(x, region), dftn(x, region)) < 10tol(T)
      z = copy(x); p = FFTL.plan_fft(x, region); mul!(z, p, z)
      @test relerr(z, dftn(x, region)) < 10tol(T)
    end
  end

  @testset "large lengths: inverse and Parseval, $T" for T in (Float64, Float32)
    # 262147 (prime) and 2·262147: convolution length beyond 2^19, a four-step on x86
    for n in pick([4096, 3^8, 10^5, 2^18, 2 * 10007, 262147, 2 * 262147], [3^8, 2^18, 2 * 10007])
      x = randn(Complex{T}, n)
      y = FFTL.fft(x)
      @test relerr(FFTL.ifft(y), x) < 10tol(T)
      @test sum(abs2, y) / n ≈ sum(abs2, x) rtol = 10tol(T)
    end
  end

  @testset "multidimensional, $T" for T in (Float64, Float32)
    cases = pick([((8, 12), (1, 2)), ((16, 16), (1, 2)), ((17, 6), (1, 2)), ((5, 37), (1, 2)),
                  ((3, 8), (2,)), ((3, 8), (1, 2)), ((12, 7), (1,)), ((1, 9), (1, 2)),
                  ((4, 6, 10), (1, 2, 3)), ((3, 5, 7), (1, 2, 3)), ((6, 4, 5), (1, 3)), ((6, 4, 5), (2,)),
                  ((16, 9, 2, 3), (1, 2, 3, 4))],
                 thin(T, [((8, 12), (1, 2)), ((5, 37), (1, 2)), ((3, 8), (2,)), ((1, 9), (1, 2)),
                          ((4, 6, 10), (1, 2, 3)), ((6, 4, 5), (1, 3)), ((16, 9, 2, 3), (1, 2, 3, 4))]))
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

  @testset "complex strided plans, $T" for T in (Float64, Float32)
    for (sz, region) in (((8,), (1,)), ((37, 3), (1,)), ((3, 74, 2), (2,)),
                         ((8, 6), (1, 2)), ((3, 8, 2), (2,)), ((4, 3, 2), (1, 3)),
                         ((19, 8, 2), (2,)), ((16, 37, 2), (2,)), ((1, 9), (1, 2)),
                         ((3, 1, 2), (2,)), ((2, 3, 4, 2), (3,)))
      x = randn(Complex{T}, sz)
      inputs = realpass_layouts(Complex{T}, sz)
      outputs = realpass_layouts(Complex{T}, sz)
      for v in inputs
        v .= x
      end
      for (mk, sign, scale) in ((FFTL.plan_fft, -1, 1), (FFTL.plan_bfft, 1, 1),
                                (FFTL.plan_ifft, 1, prod(sz[d] for d in region; init = 1)))
        p = mk(inputs[2], region; threads = 1)
        ref = dftn(x, region, sign) ./ scale
        for (a, b) in ((1, 2), (2, 1), (2, 3), (3, 4), (4, 5), (5, 2), (1, 1))
          v, w = inputs[a], outputs[b]
          before = copy(parent(v)); fill!(parent(w), 7)
          mask = trues(size(parent(w))); view(mask, parentindices(w)...) .= false
          @test mul!(w, p, v) === w
          @test relerr(w, ref) < 10tol(T)
          @test parent(v) == before
          @test all(==(7), parent(w)[mask])
        end
        @test relerr(p * inputs[4], ref) < 10tol(T)
        outputs[4] .= ref
        @test relerr(inv(p) * outputs[4], x) < 10tol(T)
        v = inputs[5]
        @test mul!(v, p, v) === v
        @test relerr(v, ref) < 10tol(T)
        v .= x
      end
      v = inputs[4]
      q = FFTL.plan_fft!(v, region; threads = 1)
      @test q * v === v
      @test relerr(v, dftn(x, region)) < 10tol(T)
      @test inv(q) * v === v
      @test relerr(v, x) < 10tol(T)
    end
    # An empty forward/backward region copies values without transforming them.
    for mk in (FFTL.plan_fft, FFTL.plan_bfft)
      v = realpass_layouts(Complex{T}, (3, 4))[4]
      w = realpass_layouts(Complex{T}, (3, 4))[5]
      @test mul!(w, mk(v, (); threads = 1), v) == v
    end
    # Arbitrary indexing views retain the generic fallback.
    v = view(randn(Complex{T}, 7, 4), [5, 2, 4], :)
    w = view(fill(Complex{T}(7), 8, 4), [2, 6, 3], :)
    @test !(v isa StridedArray) && !(w isa StridedArray)
    before = copy(parent(v)); p = FFTL.plan_fft(v, 1)
    @test relerr(mul!(w, p, v), dftn(Array(v), (1,))) < 10tol(T)
    @test parent(v) == before
    @test all(==(7), parent(w)[[1, 4, 5, 7, 8], :])
    @test_throws DimensionMismatch mul!(zeros(Complex{T}, 2, 4), p, v)
    if Threads.nthreads() > 1
      for (sz, region) in (((128, 512), (1,)), ((33, 1024), (2,)),
                           ((8192,), (1,)), ((20011,), (1,)), ((2 * 37 * 41 * 43,), (1,)))
        x = randn(Complex{T}, sz)
        v = realpass_layouts(Complex{T}, sz)[4]; v .= x
        w = realpass_layouts(Complex{T}, sz)[5]
        p = FFTL.plan_fft(v, region; threads = 2)
        ref = FFTL.plan_fft(x, region; threads = 1) * x
        @test relerr(mul!(w, p, v), ref) < 10tol(T)
        @test v == x
        mask = trues(size(parent(w))); view(mask, parentindices(w)...) .= false
        @test all(==(7), parent(w)[mask])
        @test relerr(mul!(v, p, v), ref) < 10tol(T)
      end
      # LinesMT owns separate Bluestein/composite kernels per task. Exercise first-axis
      # slices, interior-axis lane blocks (including a partial block), and multiple passes.
      # The three-lane case has gather workspace with two planned tasks, but none with three.
      for (sz, region) in (((1009, 64), (1,)), ((65, 1009), (2,)), ((2018, 33), (1,)),
                           ((65, 2018), (2,)), ((19, 129, 32), (2, 3)), ((3, 6054, 2), (2,)))
        x = randn(Complex{T}, sz)
        inputs = realpass_layouts(Complex{T}, sz)
        third = view(fill(Complex{T}(7), 3 .* sz .+ 4), ntuple(i -> 2:3:3sz[i]-1, length(sz))...)
        outputs = (realpass_layouts(Complex{T}, sz)..., third)
        for mk in (FFTL.plan_fft, FFTL.plan_bfft, FFTL.plan_ifft)
          ref = mk(x, region; threads = 1) * x
          for nt in (2, 3)
            p = mk(inputs[2], region; threads = nt)
            raw = mk === FFTL.plan_ifft ? p.p : p
            @test any(k -> k isa FFTL.LinesMT, raw.kernels)
            for (a, b) in ((1, 2), (3, 2), (3, 4), (4, 5), (4, 6), (5, 3), (2, 2), (1, 1))
              v, w = inputs[a], outputs[b]; v .= x
              before = copy(parent(v)); fill!(parent(w), 7)
              mask = trues(size(parent(w))); view(mask, parentindices(w)...) .= false
              @test (@inferred mul!(w, p, v)) === w
              @test relerr(w, ref) < 10tol(T)
              @test parent(v) == before
              @test all(==(7), parent(w)[mask])
            end
            v = inputs[5]; v .= x
            mask = trues(size(parent(v))); view(mask, parentindices(v)...) .= false
            @test mul!(v, p, v) === v
            @test relerr(v, ref) < 10tol(T)
            @test all(==(7), parent(v)[mask])
          end
        end
      end
    end
  end

  @testset "threads, $T" for T in (Float64, Float32)
    # sizes above the threading threshold: slices (dimension 1 of a 2D array), lane blocks
    # (last dimension), odd strides (last block with fewer lanes than the vector width)
    for sz in pick([(256, 256), (64, 64, 64), (33, 1024), (100, 100, 10), (8, 4096), (1024, 37), (2 * 3^11,), (5^7,)],
                   [(64, 64, 64), (33, 1024), (1024, 37), (2 * 3^11,)])
      x = randn(Complex{T}, sz)
      ref = FFTL.plan_fft(x; threads = 1) * x
      for nt in pick((2, 3, 4), (3,))
        p = FFTL.plan_fft(x; threads = nt)
        @test relerr(p * x, ref) < tol(T)
        z = copy(x); mul!(z, p, z)
        @test relerr(z, ref) < tol(T)
      end
    end
  end

  @testset "Bluestein on several threads, $T" for T in (Float64, Float32)
    # one long line with a prime factor above MAX_RADIX_PRIME: threaded convolution transforms
    for n in pick((16411, 20011, 100003), (20011,))
      x = randn(Complex{T}, n)
      for (mk, σ) in ((FFTL.plan_fft, -1), (FFTL.plan_bfft, 1))
        ref = mk(x; threads = 1) * x
        for nt in pick((2, 3, 4), (3,))
          p = mk(x; threads = nt)
          @test p.kernels[1] isa FFTL.BluesteinMT
          @test relerr(p * x, ref) < tol(T)
          z = copy(x); mul!(z, p, z)
          @test relerr(z, ref) < tol(T)
        end
      end
    end
    # below the threshold the serial Bluestein kernel stays
    x = randn(Complex{T}, 4099)
    @test FFTL.plan_fft(x; threads = 3).kernels[1] isa FFTL.Bluestein
    # smooth part times leftover: the composite kernel on threads, bitwise equal to the serial one
    # 3·193·1021: the leftover is split, the first stage is a composite kernel on each task
    for n in pick((65 * 4033, 7 * 11777, 1500 * 1013, 3 * 193 * 1021), (3 * 193 * 1021,))
      x = randn(Complex{T}, n)
      for mk in (FFTL.plan_fft, FFTL.plan_bfft)
        ref = mk(x; threads = 1) * x
        for nt in pick((2, 3, 4), (3,))
          p = mk(x; threads = nt)
          @test p.kernels[1] isa FFTL.CompositeMT
          @test p * x == ref
          z = copy(x); mul!(z, p, z)
          @test z == ref
        end
      end
    end
    # many lines with a prime factor above MAX_RADIX_PRIME: slices and lane blocks over the tasks,
    # Bluestein kernels (127, 1009, 37) and composite kernels (129 = 3·43, 2018 = 2·1009)
    for (sz, region) in pick([((256, 127, 16), 1:3), ((16, 256, 127), (3,)), ((1009, 64), (1,)), ((64, 1009), (2,)),
                              ((37, 37, 37), 1:3), ((129, 300), 1:2), ((300, 129), 1:2), ((2018, 32), (1,)), ((32, 2018), (2,))],
                             [((256, 127, 16), 1:3), ((64, 1009), (2,)), ((1009, 64), (1,)), ((300, 129), 1:2)])
      x = randn(Complex{T}, sz...)
      for mk in (FFTL.plan_fft, FFTL.plan_bfft)
        p1 = mk(x, region; threads = 1)
        ref = p1 * x
        @test !any(k -> k isa FFTL.LinesMT, p1.kernels)
        for nt in pick((2, 3, 4, 8), (3,))
          p = mk(x, region; threads = nt)
          @test any(k -> k isa FFTL.LinesMT, p.kernels)
          @test relerr(p * x, ref) < tol(T)
          z = copy(x); mul!(z, p, z)
          @test relerr(z, ref) < tol(T)
        end
      end
    end
    # below the threshold of threaded work the serial kernels stay
    x = randn(Complex{T}, 127, 16)
    @test !any(k -> k isa FFTL.LinesMT, FFTL.plan_fft(x; threads = 3).kernels)
  end

  @testset "real strided plans, $T" for T in (Float64, Float32)
    for (sz, region) in (((1,), (1,)), ((8,), (1,)), ((9, 4), (1,)), ((3, 37, 2), (2,)),
                         ((4, 6), (1, 2)), ((4, 5), (2, 1)), ((3, 8, 2), (2, 3)),
                         ((8, 3, 4), (1, 3)), ((3, 1, 2), (2,)))
      # Preserve the existing sorted region and choice of the halved dimension.
      dims = sort(collect(region)); d = first(dims); n = sz[d]
      osz = Base.setindex(sz, n ÷ 2 + 1, d)
      x = randn(T, sz)
      full = dftn(x, dims)
      ref = Array(view(full, ntuple(j -> j == d ? (1:osz[j]) : Colon(), length(sz))...))
      pf = FFTL.plan_rfft(x, region; threads = 1)
      pb = FFTL.plan_brfft(Complex{T}.(ref), n, region; threads = 1)
      pi = FFTL.plan_irfft(Complex{T}.(ref), n, region; threads = 1)
      for (p, forward, expected) in ((pf, true, ref), (pb, false, prod(sz[j] for j in dims) .* x), (pi, false, x))
        inputs = realpass_layouts(forward ? T : Complex{T}, forward ? sz : osz)
        outputs = realpass_layouts(forward ? Complex{T} : T, forward ? osz : sz)
        for v in inputs
          v .= forward ? x : ref
        end
        for (a, b) in ((1, 2), (2, 1), (2, 3), (3, 4), (4, 5), (5, 2), (1, 1))
          v, w = inputs[a], outputs[b]
          before = copy(parent(v)); fill!(parent(w), 7)
          mask = trues(size(parent(w))); view(mask, parentindices(w)...) .= false
          @test mul!(w, p, v) === w
          @test relerr(w, expected) < 10tol(T)
          @test parent(v) == before
          @test all(==(7), parent(w)[mask])
        end
        v = inputs[4]
        @test relerr(p * v, expected) < 10tol(T)
        @test relerr(inv(p) * (p * v), v) < 10tol(T)
      end
    end
    x = realpass_layouts(T, (8, 3))[4]
    X = realpass_layouts(Complex{T}, (5, 3))[4]
    p = FFTL.plan_rfft(x, 1)
    q = FFTL.plan_brfft(X, 8, 1)
    @test_throws DimensionMismatch mul!(zeros(Complex{T}, 4, 3), p, x)
    @test_throws DimensionMismatch mul!(X, p, zeros(T, 7, 3))
    @test_throws DimensionMismatch mul!(zeros(T, 9, 3), q, X)
    @test_throws DimensionMismatch mul!(x, q, zeros(Complex{T}, 4, 3))
    @test_throws ArgumentError FFTL.plan_rfft(x, 0)
    @test_throws ArgumentError FFTL.plan_rfft(x, ())
    @test_throws ArgumentError FFTL.plan_brfft(X, 0, 1)
    @test_throws DimensionMismatch FFTL.plan_brfft(X, 10, 1)
    if Threads.nthreads() > 1
      for (sz, region) in (((128, 512), (1,)), ((32, 64, 32), (1, 3)), ((65536,), (1,)))
        d = first(region); n = sz[d]; osz = Base.setindex(sz, n ÷ 2 + 1, d)
        x = randn(T, sz)
        inputs = realpass_layouts(T, sz); spectra = realpass_layouts(Complex{T}, osz)
        outputs = realpass_layouts(T, sz)
        p = FFTL.plan_rfft(inputs[4], region; threads = 2)
        q = FFTL.plan_irfft(spectra[5], n, region; threads = 2)
        ref = FFTL.plan_rfft(x, region; threads = 1) * x
        for (a, b) in ((4, 5), (2, 2), (1, 1))
          v, w, y = inputs[a], spectra[b], outputs[a]; v .= x
          @test relerr(mul!(w, p, v), ref) < 10tol(T)
          @test v == x
          w .= ref
          @test relerr(mul!(y, q, w), x) < 10tol(T)
          @test w == ref
          for out in (w, y)
            mask = trues(size(parent(out))); view(mask, parentindices(out)...) .= false
            @test all(==(7), parent(out)[mask])
          end
        end
      end
      # The complex passes of real plans also select LinesMT, including inverse passes into
      # the plan's dense scratch. Use independent dense references for the inverse inputs.
      for (sz, region) in (((4, 1009, 32), (1, 2)), ((8, 65, 129), (1, 3)), ((65, 8, 129), (2, 3)))
        d = first(region); n = sz[d]; osz = Base.setindex(sz, n ÷ 2 + 1, d)
        x = randn(T, sz)
        ref = FFTL.plan_rfft(x, region; threads = 1) * x
        for nt in (2, 3)
          pf = FFTL.plan_rfft(x, region; threads = nt)
          pb = FFTL.plan_brfft(ref, n, region; threads = nt)
          pi = FFTL.plan_irfft(ref, n, region; threads = nt)
          for (p, forward, expected) in ((pf, true, ref), (pb, false, prod(sz[j] for j in region) .* x), (pi, false, x))
            raw = p === pi ? p.p : p
            @test any(k -> k isa FFTL.LinesMT, raw.cplan.kernels)
            inputs = realpass_layouts(forward ? T : Complex{T}, forward ? sz : osz)
            outputs = realpass_layouts(forward ? Complex{T} : T, forward ? osz : sz)
            for (a, b) in ((2, 2), (4, 2), (4, 5), (3, 4), (5, 3), (1, 1))
              v, w = inputs[a], outputs[b]; v .= forward ? x : ref
              before = copy(parent(v)); fill!(parent(w), 7)
              mask = trues(size(parent(w))); view(mask, parentindices(w)...) .= false
              @test (@inferred mul!(w, p, v)) === w
              @test relerr(w, expected) < 10tol(T)
              @test parent(v) == before
              @test all(==(7), parent(w)[mask])
            end
          end
        end
      end
    end
  end

  @testset "real passes, $T" for T in (Float64, Float32)
    # r2c along each dimension of a strided view, into a dense output and into a strided view of a
    # padded output (the RNFFT grid); the padding of the output is not written
    G0 = randn(T, 9, 12, 5)
    for d in 1:3, nt in pick((1, 3), (3,))
      v = view(G0, 2:8, 3:10, 2:4)
      p = FFTL.plan_r2c_pass(v, d; threads = nt)
      osz = ntuple(i -> i == d ? size(v, d) ÷ 2 + 1 : size(v, i), 3)
      ref = mapslices(u -> dft(ComplexF64.(u))[1:length(u) ÷ 2 + 1], Float64.(Array(v)); dims = d)
      w = Array{Complex{T}}(undef, osz)
      mul!(w, p, v)
      @test relerr(w, ref) < 10tol(T)
      P = zeros(Complex{T}, osz .+ 2)
      wv = view(P, ntuple(i -> 2:osz[i]+1, 3)...)
      mul!(wv, p, v)
      @test relerr(wv, ref) < 10tol(T)
      mask = trues(size(P)); mask[ntuple(i -> 2:osz[i]+1, 3)...] .= false
      @test all(iszero, P[mask])
    end
    # round trips, even and odd lengths, dense and into a strided view
    for (n, d) in pick(((8, 1), (9, 2), (16, 2), (7, 1), (100, 1), (45, 2)), ((8, 1), (9, 2))), nt in (1, 2)
      sz = d == 1 ? (n, 4) : (3, n)
      x = randn(T, sz...)
      pr = FFTL.plan_r2c_pass(x, d; threads = nt)
      X = Array{Complex{T}}(undef, ntuple(i -> i == d ? n ÷ 2 + 1 : sz[i], 2))
      mul!(X, pr, x)
      pc = FFTL.plan_c2r_pass(X, d; n, threads = nt)
      y = Array{T}(undef, sz)
      mul!(y, pc, X)
      @test relerr(y ./ n, x) < 10tol(T)
      Y = zeros(T, sz .+ 2)
      yv = view(Y, ntuple(i -> 2:sz[i]+1, 2)...)
      mul!(yv, pc, X)
      @test relerr(yv ./ n, x) < 10tol(T)
    end
    # Reuse each plan while changing input strides, output strides, both, then back to dense.
    for (sz, d) in (((8,), 1), ((9, 4), 1), ((3, 8, 2), 2), ((2, 37, 1), 2), ((3, 1, 2), 2))
      n = sz[d]
      x = randn(T, sz)
      osz = Base.setindex(sz, n ÷ 2 + 1, d)
      ref = mapslices(u -> dft(u)[1:n ÷ 2 + 1], x; dims = d)
      for forward in (true, false)
        inputs = realpass_layouts(forward ? T : Complex{T}, forward ? sz : osz)
        outputs = realpass_layouts(forward ? Complex{T} : T, forward ? osz : sz)
        for v in inputs
          v .= forward ? x : ref             # inverse input from the reference, not our r2c
        end
        p = forward ? FFTL.plan_r2c_pass(inputs[1], d) : FFTL.plan_c2r_pass(inputs[1], d; n)
        expected = forward ? ref : n .* x
        for (a, b) in ((1, 1), (2, 1), (2, 3), (3, 4), (4, 5), (5, 2), (1, 1))
          v, w = inputs[a], outputs[b]
          before = copy(parent(v))
          fill!(parent(w), 7)
          mask = trues(size(parent(w)))
          view(mask, parentindices(w)...) .= false
          @test mul!(w, p, v) === w
          @test relerr(w, expected) < 10tol(T)
          @test parent(v) == before
          @test all(==(7), parent(w)[mask])
        end
      end
    end
    # Enough work to execute on multiple tasks, with independently strided operands.
    if Threads.nthreads() > 1
      x = randn(T, 128, 512)
      v = view(zeros(T, 260, 514), 2:2:256, 2:513); v .= x
      w = view(fill(Complex{T}(7), 69, 514), 2:66, 2:513)
      y = view(fill(T(7), 260, 514), 2:2:256, 2:513)
      ref = FFTL.plan_rfft(x, 1; threads = 1) * x
      mul!(w, FFTL.plan_r2c_pass(v, 1; threads = 2), v)
      @test relerr(w, ref) < 10tol(T)
      w .= ref
      mul!(y, FFTL.plan_c2r_pass(w, 1; n = 128, threads = 2), w)
      @test relerr(y, 128 .* x) < 10tol(T)
      @test v == x && w == ref
      for out in (w, y)
        mask = trues(size(parent(out))); view(mask, parentindices(out)...) .= false
        @test all(==(7), parent(out)[mask])
      end
    end
    # sizes and lengths that do not match are errors
    x = randn(T, 10, 3)
    @test_throws DimensionMismatch FFTL.plan_c2r_pass(zeros(Complex{T}, 5), 1; n = 10)
    pr = FFTL.plan_r2c_pass(x, 1)
    @test_throws DimensionMismatch mul!(Array{Complex{T}}(undef, 5, 3), pr, x)
    @test_throws ArgumentError mul!(Array{Complex{T}}(undef, 6, 3), FFTL.plan_c2r_pass(zeros(Complex{T}, 6, 3), 1), zeros(T, 10, 3))
  end

  @testset "passes on strided views, $T" for T in (Float64, Float32)
    # views of a padded array as an NFFT uses them: interior boxes and boxes at the edge
    G0 = randn(Complex{T}, 40, 36, 30)
    boxes = pick([(3:34, 2:33, 2:25), (3:6, 2:33, 2:25), (3:34, 5:9, 2:25), (1:40, 1:36, 1:30),
                  (2:38, 2:36, 3:3), (5:40, 1:36, 4:28)],
                 thin(T, [(3:34, 2:33, 2:25), (3:6, 2:33, 2:25), (2:38, 2:36, 3:3)]))
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

  @testset "real transforms, $T" for T in (Float64, Float32)
    for (sz, region) in pick([((8,), 1), ((9,), 1), ((10,), 1), ((15,), 1), ((16,), 1), ((37,), 1), ((100,), 1),
                              ((101,), 1), ((1024,), 1), ((1023,), 1), ((3 * 2^11,), 1),
                              ((8, 6), (1, 2)), ((9, 6), (1, 2)), ((7, 10), (1, 2)), ((8, 6), (2,)), ((12, 7), (2,)),
                              ((6, 8, 5), (1, 2, 3)), ((6, 8, 5), (2, 3)), ((6, 9, 5), (2, 3)), ((5, 4, 6), (3,)), ((16, 9), (1,))],
                             thin(T, [((9,), 1), ((16,), 1), ((37,), 1), ((1024,), 1), ((3 * 2^11,), 1),
                                      ((9, 6), (1, 2)), ((12, 7), (2,)), ((6, 8, 5), (1, 2, 3)), ((6, 9, 5), (2, 3))]))
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
      # absolute tolerance on the scale |yr|·|p x|: a relative one fails when the product is near zero
      @test abs(real(dot(yr, p * x)) - dot(adjoint(p) * yr, x)) <= 100tol(T) * norm(yr) * norm(p * x)
    end
    # threads, large lines and strided lines
    for (sz, region) in pick([((2^16, 4), (1, 2)), ((64, 513), (2,)), ((1000, 300), (1, 2))], [((2^16, 4), (1, 2)), ((64, 513), (2,))])
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
      elseif kind == FFTL.R2HC   # Re F_k, then Im F_(n-k) of F = DFT(x)
        2k <= n ? sum(x[j+1] * cospi(2j * k / n) for j in 0:n-1) : -sum(x[j+1] * sinpi(2j * (n - k) / n) for j in 0:n-1)
      elseif kind == FFTL.HC2R   # Σ_m F_m e^(2πimk/n), F from halfcomplex x
        F(m) = m == 0 || 2m == n ? complex(x[m+1]) : 2m < n ? complex(x[m+1], x[n-m+1]) : conj(F(n - m))
        real(sum(F(m) * cispi(2m * k / n) for m in 0:n-1))
      else
        sum(x[j+1] * (cospi(2j * k / n) + sinpi(2j * k / n)) for j in 0:n-1)
      end
    end
    y
  end
  r2rkinds = (FFTL.REDFT00, FFTL.REDFT10, FFTL.REDFT01, FFTL.REDFT11, FFTL.RODFT00, FFTL.RODFT10,
              FFTL.RODFT01, FFTL.RODFT11, FFTL.DHT, FFTL.R2HC, FFTL.HC2R)

  @testset "real-to-real, $T" for T in pick((Float64, Float32), (Float64,))
    for kind in r2rkinds, n in pick([1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 31, 32, 33, 100, 101, 128], thin(T, [1, 2, 3, 5, 16, 33]))
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
    for nt in pick((1, 3), (3,))
      p = FFTL.plan_r2r(x, kinds, 1:3; threads = nt)
      @test relerr(p * x, ref) < 10tol(T)
      @test relerr(inv(p) * (p * x), x) < 10tol(T)
    end
    hc = mapslices(u -> r2r_ref(u, FFTL.R2HC), mapslices(u -> r2r_ref(u, FFTL.HC2R), Float64.(x); dims = 1); dims = 3)
    @test relerr(FFTL.r2r(x, (FFTL.HC2R, FFTL.R2HC), (1, 3)), hc) < 10tol(T)
    @test relerr(FFTL.r2r(x, FFTL.REDFT00, (2,)), mapslices(u -> r2r_ref(u, FFTL.REDFT00), Float64.(x); dims = 2)) < 10tol(T)
    # strided dimensions: blocked lines, and DCT-II/DST-II of even length by the half-length DFT
    x = randn(T, 6, 10, 7, 4)
    for kind in r2rkinds, reg in ((2,), (3,), (2, 4), (4, 1, 2)), nt in pick((1, 3), (3,))
      ref = Float64.(x)
      for d in reg
        ref = mapslices(u -> r2r_ref(u, kind), ref; dims = d)
      end
      p = FFTL.plan_r2r(x, kind, reg; threads = nt)
      @test relerr(p * x, ref) < 10tol(T)
      y = copy(x); mul!(y, p, y)
      @test relerr(y, ref) < 10tol(T)
    end
    @test FFTL.plan_r2r(x, FFTL.REDFT10, (2, 3)).halves[1] isa FFTL.HalfDFT
    @test FFTL.plan_r2r(x, FFTL.RODFT10, 1).halves[1] === nothing
    # pass plans on views of a padded array, all kinds, every dimension, in place and out of place
    G0 = randn(T, 26, 21, 13)
    for kind in thin(T, collect(r2rkinds)), d in pick(1:3, (1, 3)), nt in pick((1, 2), (2,))
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
    for Ñ in pick((96, 135, 250, 324), (96, 135)), (kind, n) in ((FFTL.REDFT00, Ñ + 1), (FFTL.RODFT00, Ñ - 1))
      H0 = randn(T, n + 4, 7); H1 = similar(H0)
      v = view(H0, 3:n+2, 2:6); w = view(H1, 3:n+2, 2:6)
      mul!(w, FFTL.plan_r2r_pass(v, 1, kind), v)
      @test relerr(w, mapslices(u -> r2r_ref(u, kind), Float64.(Array(v)); dims = 1)) < 10tol(T)
    end
    # DCT-I and DST-I as accurate as the other kinds (rounding errors must not grow with n);
    # Float32 against the Float64 reference, which is exact at this scale
    if T == Float32
      for (kind, n) in ((FFTL.REDFT00, 2049), (FFTL.RODFT00, 2047), (FFTL.REDFT10, 2048))
        u = randn(T, n)
        @test relerr(FFTL.r2r(u, kind), r2r_ref(Float64.(u), kind)) < 4eps(T)
      end
    end
    # orthonormal DCT
    y = FFTL.dct(x)
    @test sum(abs2, y) ≈ sum(abs2, x) rtol = 10tol(T)
    @test relerr(FFTL.idct(y), x) < 10tol(T)
    @test relerr(FFTL.idct(FFTL.dct(x, 2), 2), x) < 10tol(T)
  end

  # quick run: Float64 only (the first RChain call compiles every codelet of its element type)
  @testset "runtime-radix chains, $T" for T in pick((Float64, Float32), (Float64,))
    # RChain runs the same codelets in the same order as Chain: results must be identical
    p(a) = Ptr{T}(pointer(a))
    for n in pick([1, 2, 7, 8, 16, 60, 100, 128, 243, 1000, 31 * 27, 4096, 2^14, 3^8, 2 * 10007 - 14],
                  thin(T, [1, 8, 60, 128, 243, 1000, 31 * 27, 2^14]))
      rs = FFTL.radices(n, 1; esize = sizeof(Complex{T}))
      c = FFTL.build_chain(T, n, -1, rs); rc = FFTL.RChain{T}(n, -1, rs)
      x = randn(Complex{T}, n); w1 = similar(x); w2 = similar(x)
      y1 = similar(x); y2 = similar(x); z1 = copy(x); z2 = copy(x)
      GC.@preserve x y1 y2 z1 z2 w1 w2 begin
        FFTL.run!(p(y1), p(x), p(w1), p(w2), c, 1); FFTL.run!(p(y2), p(x), p(w1), p(w2), rc, 1)
        FFTL.run!(p(z1), p(z1), p(w1), p(w2), c, 1); FFTL.run!(p(z2), p(z2), p(w1), p(w2), rc, 1)
      end
      @test y1 == y2 && z1 == z2
      # three lanes, element j of lane i at i + 3j; strided batch S = 3 through run!
      B = 3; xs = randn(Complex{T}, B * n); wl1 = similar(xs); wl2 = similar(xs)
      l1 = similar(xs); l2 = similar(xs); s1 = similar(xs); s2 = similar(xs)
      cS = FFTL.build_chain(T, n, 1, FFTL.radices(n, B; esize = sizeof(Complex{T})))
      rS = FFTL.RChain{T}(n, 1, FFTL.radices(n, B; esize = sizeof(Complex{T})))
      GC.@preserve xs l1 l2 s1 s2 wl1 wl2 begin
        FFTL.run_lanes!(p(l1), p(xs), p(wl1), p(wl2), cS, B, B, B); FFTL.run_lanes!(p(l2), p(xs), p(wl1), p(wl2), rS, B, B, B)
        FFTL.run!(p(s1), p(xs), p(wl1), p(wl2), cS, B); FFTL.run!(p(s2), p(xs), p(wl1), p(wl2), rS, B)
      end
      @test l1 == l2 && s1 == s2
    end
  end

  # run-time plans (RChain kernels): same results as the reference, four-step and plane fusion
  @testset "RPlan, $T" for T in pick((Float64, Float32), (Float64,))
    for (sz, reg, nt) in pick([((1000,), [1], 1), ((2 * 3^11,), [1], 3), ((1009,), [1], 1), ((64, 48), [1, 2], 1),
                               ((37, 100), [1, 2], 3), ((16, 9, 5), [1, 2, 3], 1), ((8, 37, 6), [2], 3), ((40, 36, 10), [1, 3], 3)],
                              [((2 * 3^11,), [1], 3), ((1009,), [1], 1), ((37, 100), [1, 2], 3), ((16, 9, 5), [1, 2, 3], 1), ((8, 37, 6), [2], 3)])
      x = randn(Complex{T}, sz)
      p = FFTL.RPlan{T}(collect(sz), reg, -1; threads = nt)
      ref = FFTL.plan_fft(x, Tuple(reg); threads = 1) * x
      @test relerr(p * x, ref) < 10tol(T)
      z = copy(x); GC.@preserve z FFTL.execute!(p, pointer(z), pointer(z))
      @test relerr(z, ref) < 10tol(T)
      @test relerr(inv(p) * (p * x), x) < 10tol(T)
    end
    # one long line with a trailing dimension of length 1: four-step, as in FFTLPlan
    x = randn(Complex{T}, 2^14, 1)
    p = FFTL.RPlan{T}([2^14, 1], [1, 2], -1; threads = 2)
    @test p.four !== nothing
    @test relerr(p * x, FFTL.plan_fft(x; threads = 1) * x) < 10tol(T)
  end

  @testset "run-time real plans, $T" for T in pick((Float64, Float32), (Float64,))
    for (sz, reg, nt) in [((16,), [1], 1), ((101,), [1], 1), ((12, 10), [1, 2], 3), ((12, 10), [2], 1), ((9, 8, 7), [1, 3], 3)]
      x = randn(T, sz)
      p = FFTL.RRFFTPlan{T}(collect(sz), reg, -1; threads = nt)
      ref = FFTL.plan_rfft(x, Tuple(reg)) * x
      y = similar(ref)
      GC.@preserve x y FFTL.execute_r2c!(p, pointer(y), pointer(x))
      @test relerr(y, ref) < 10tol(T)
      q = FFTL.RRFFTPlan{T}(collect(sz), reg, 1; threads = nt)
      z = similar(x)
      GC.@preserve y z FFTL.execute_c2r!(q, pointer(z), pointer(y))
      @test relerr(z ./ prod(sz[d] for d in reg), x) < 10tol(T)
      # in place: real array padded to 2(n÷2+1) along the first transformed dimension
      d1 = reg[1]; psz = collect(sz); psz[d1] = 2 * (sz[d1] ÷ 2 + 1)
      buf = zeros(T, psz...); idx = ntuple(i -> i == d1 ? (1:sz[d1]) : (:), length(sz))
      buf[idx...] = x
      GC.@preserve buf FFTL.execute_r2c!(p, Ptr{Complex{T}}(pointer(buf)), pointer(buf))
      @test relerr(reshape(reinterpret(Complex{T}, vec(buf)), size(ref)), ref) < 10tol(T)
      GC.@preserve buf FFTL.execute_c2r!(q, pointer(buf), Ptr{Complex{T}}(pointer(buf)))
      @test relerr(buf[idx...] ./ prod(sz[d] for d in reg), x) < 10tol(T)
    end
    x = randn(T, 9, 12, 5)
    ks = [FFTL.HC2R, FFTL.REDFT00, FFTL.RODFT11]
    p = FFTL.RR2RPlan{T}([9, 12, 5], ks, [1, 2, 3]; threads = 3)
    y = copy(x)
    GC.@preserve y FFTL.execute_r2r!(p, pointer(y), pointer(y))
    @test relerr(y, FFTL.r2r(x, Tuple(ks), 1:3)) < 10tol(T)
  end

  @testset "generic float types (BigFloat)" begin
    setprecision(BigFloat, 128) do
      B = BigFloat; btol = 1e-35
      dftB(x, σ) = [sum(x[j+1] * cispi(B(2σ) * mod(j * k, length(x)) / length(x)) for j in 0:length(x)-1) for k in 0:length(x)-1]
      for n in pick((1, 2, 7, 12, 16, 37, 97, 100), (7, 16, 37))
        x = Complex{B}.(randn(ComplexF64, n))
        @test eltype(FFTL.fft(x)) == Complex{B}
        @test relerr(FFTL.fft(x), dftB(x, -1)) < btol
        @test relerr(FFTL.bfft(x), dftB(x, 1)) < btol
        @test relerr(FFTL.ifft(FFTL.fft(x)), x) < btol
      end
      x = Complex{B}.(randn(ComplexF64, 6, 5, 4))
      @test relerr(FFTL.fft(x, (1, 3)), FFTL.fft(FFTL.fft(x, 1), 3)) < btol
      @test relerr(FFTL.ifft(FFTL.fft(x)), x) < btol
      for (sz, reg) in (((16,), 1:1), ((37,), 1:1), ((6, 5), 1:2), ((6, 5), 2:2), ((4, 5, 3), (1, 3)))
        xr = B.(randn(sz...))
        X = FFTL.rfft(xr, reg)
        d1 = minimum(reg)
        @test relerr(X, FFTL.fft(Complex{B}.(xr), reg)[ntuple(i -> i == d1 ? (1:size(X, d1)) : (:), length(sz))...]) < btol
        @test relerr(FFTL.irfft(X, sz[d1], reg), xr) < btol
      end
      # the r2r formulas against the Float64 kernels, the precision through the inverse pairs
      for k in r2rkinds, n in (2, 5, 8, 33)
        xr = B.(randn(n))
        y = FFTL.r2r(xr, k)
        @test eltype(y) == B
        @test relerr(Float64.(y), FFTL.r2r(Float64.(xr), k)) < 1e-12
        @test relerr(inv(FFTL.plan_r2r(xr, k)) * y, xr) < btol
      end
      xr = B.(randn(5, 6, 4))
      @test relerr(FFTL.idct(FFTL.dct(xr)), xr) < btol
    end
  end

  @testset "real transforms of one long line on several threads" begin
    # from 256 KB per line on: Float64 2^15 and 50000, Float32 2^16 and 100000, and larger
    for (T, n) in ((Float64, 2^15), (Float64, 50000), (Float32, 2^16), (Float32, 100000),
                   (Float64, 2^17), (Float64, 3 * 2^16), (Float32, 2^17), (Float32, 3 * 2^16))
      x = randn(T, n)
      p = FFTL.plan_rfft(x; threads = 3)
      @test p.four !== nothing
      X = p * x
      @test relerr(X, FFTL.plan_rfft(x; threads = 1) * x) < 10tol(T)
      @test relerr(FFTL.plan_brfft(X, n; threads = 3) * X, n .* x) < 10tol(T)
    end
    # below 256 KB per line one task does the line
    @test FFTL.plan_rfft(randn(Float64, 2^14); threads = 3).four === nothing
    @test FFTL.plan_rfft(randn(Float32, 2^15); threads = 3).four === nothing
  end

  @testset "argument checks and fixed bugs" begin
    # Bluestein lines in threaded real and r2r plans (the kernel's own buffers, one task at a time)
    x = rand(1031, 40)
    @test FFTL.plan_rfft(x; threads = 4) * x == FFTL.plan_rfft(x; threads = 1) * x
    @test FFTL.plan_r2r(x, FFTL.REDFT10, 1; threads = 4) * x == FFTL.plan_r2r(x, FFTL.REDFT10, 1; threads = 1) * x
    # cache sizes changed after planning: lane blocks never exceed the plan's buffers
    z = randn(ComplexF64, 2048, 64)
    try
      FFTL.set_cpu_caches!(l2 = 512 << 10)
      p = FFTL.plan_fft(z; threads = 2); y1 = p * z
      FFTL.set_cpu_caches!(l2 = 8 << 20)
      @test p * z == y1
    finally
      FFTL.set_cpu_caches!()
    end
    # duplicate region entries, integer signs, invalid signs and sizes
    w = randn(ComplexF64, 16)
    @test FFTL.ifft(w, [1, 1]) ≈ FFTL.ifft(w)
    @test FFTL.plan_pass(randn(ComplexF64, 16, 4), 1; sign = Int32(-1)) isa FFTL.PassPlan
    @test_throws ArgumentError FFTL.FFTLPlan{Float64}((16,), 1:1, 2)
    @test_throws ArgumentError FFTL.rfft(Float64[])
    @test_throws ArgumentError FFTL.plan_pass(zeros(ComplexF64, 0, 4), 1)
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
    # every radix the rule produces has a runtime codelet (RChain)
    @test all(n -> all(in(FFTL.RADICES), FFTL.radices(n, 1)) && all(in(FFTL.RADICES), FFTL.radices(n, 4096)),
              filter(FFTL.smooth, 2:pick(20000, 4000)))
    for n in [2^k for k in 0:22]
      @test prod(FFTL.radices(n); init = 1) == n
    end
    for n in [1000, 3000, 10^6, 2^5 * 3^4 * 5^3 * 7 * 11 * 13]
      @test prod(FFTL.radices(n)) == n
    end
    for (n, S) in [(48, 2304), (144, 20736), (160, 25600), (120, 14400), (100, 10000), (128, 16384), (11, 4096)]
      @test prod(FFTL.radices(n, S)) == n
    end
    # (4, 12) at stride 2304 puts 12 streams into one L1 set; the packed choice has radices ≤ 8
    @test FFTL.set_conflict([4, 12], 48, 2304)
    @test !FFTL.set_conflict(FFTL.radices(48, 2304), 48, 2304)
    @test all(<=(8), FFTL.radices(144, 144^2))
  end

  @testset "cache-derived sizes" begin
    @test FFTL.cpu_caches().l1 > 0 && FFTL.cpu_caches().l2 > 0
    @test FFTL.parse_size("512K") == 512 << 10 && FFTL.parse_size("32768K") == 32 << 20
    @test FFTL.count_cpus("0-7,128-135\n") == 16 && FFTL.count_cpus("0,128") == 2
    try
      FFTL.set_cpu_caches!(l1 = 128 << 10, l2 = 3 << 20, llc = 12 << 20)   # M1 Max
      @test (FFTL.block_budget(2048, 8), FFTL.block_budget(2048, 1)) == (1 << 16, 1 << 16)
      FFTL.set_cpu_caches!(l1 = 32 << 10, l2 = 512 << 10, llc = 32 << 20)  # Zen 3
      @test (FFTL.block_budget(2048, 8), FFTL.block_budget(2048, 1), FFTL.block_budget(512, 8)) == (1 << 14, 1 << 16, 1 << 16)
      @test FFTL.cpu_caches().source === :user
      x = randn(ComplexF64, 2^16)
      @test FFTL.plan_fft(x; threads = 2) * x ≈ FFTL.plan_fft(x) * x
    finally
      FFTL.set_cpu_caches!()
    end
    @test FFTL.cpu_caches().source !== :user
  end
  include("inference.jl")
  include("ka.jl")
  include("fftw_compat.jl")
  include("aqua.jl")
end

@testset "radix sequences match the table of this platform" begin
  include("radix_table.jl")
  @test isfile(radix_table_path())
  isfile(radix_table_path()) && @test radix_table_lines() == readlines(radix_table_path())
end
