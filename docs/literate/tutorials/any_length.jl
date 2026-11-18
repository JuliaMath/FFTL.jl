# # Any length, no padding
#
# A measurement has the length it has: 1009 samples, 365 days, a detector with 487 pixels. Many FFT
# codes are fast only for powers of two, and the usual advice is to zero-pad to the next one. That
# changes the result, not just the speed. FFTL transforms every length as it is.

using FFTL, CairoMakie, LinearAlgebra

# ## Padding changes the spectrum
#
# A record of ``N = 1009`` samples holds a tone with exactly 50 periods. On the DFT grid of this
# length the tone falls on one bin.

N = 1009
t = 0:N-1
x = cos.(2π * 50 .* t ./ N);

X = fft(x)
count(>(1e-8 * N), abs.(X))   # bins that carry the tone: 50 and its mirror image

# Zero-padded to 1024, the grid no longer matches the tone. Its energy leaks into every bin:

Xp = fft([x; zeros(1024 - N)])
count(>(1e-8 * N), abs.(Xp))

# The two spectra, as amplitude per bin against frequency in cycles per record:

fig = Figure(size = (640, 330))
ax = Axis(fig[1, 1]; xlabel = "frequency (cycles per record)", ylabel = "amplitude",
          yscale = log10, limits = (0, 100, 1e-6, 1))
f = (0:N÷2)
lines!(ax, f, max.(2 .* abs.(X[1:N÷2+1]) ./ N, 1e-6); label = "N = 1009, as recorded")
fp = (0:512) .* (N / 1024)
lines!(ax, fp, max.(2 .* abs.(Xp[1:513]) ./ N, 1e-6); label = "padded to 1024")
axislegend(ax; position = :rt)
fig

# Padding also changes the frequencies of the bins: bin ``k`` of the padded transform is at
# ``k · 1009/1024`` cycles per record, not at ``k``. Padding is the right tool for linear
# convolution (see [Fast convolution](@ref "Fast convolution")), not for reading a spectrum.

# ## What other lengths cost
#
# Every length gets its own plan. Lengths whose prime factors are at most 73 run through
# generated codelets. A larger prime factor is split off and computed by resampling onto a smooth
# grid (see [Algorithms](@ref "Lengths with large prime factors: resampling")). That costs more,
# but it is not a cliff. The time per transform for the lengths around 1024, measured on the
# machine that built this page with one thread:

function transform_time(n)
  x = randn(ComplexF64, n); y = similar(x)
  p = plan_fft(x; threads = 1)
  mul!(y, p, x)
  minimum(@elapsed(for _ in 1:100; mul!(y, p, x); end) / 100 for _ in 1:5)
end

ns = [1000, 1001, 1008, 1009, 1012, 1013, 1020, 1021, 1024, 1025, 1031, 1040]
times = [transform_time(n) for n in ns];

# A length is marked as non-smooth when it has a prime factor above 73:

function largest_prime(n)
  p, last = 2, 1
  while n > 1
    while n % p == 0
      n ÷= p; last = p
    end
    p += 1
  end
  last
end
smooth = [largest_prime(n) <= 73 for n in ns]

fig = Figure(size = (640, 330))
ax = Axis(fig[1, 1]; xlabel = "length n", ylabel = "time (µs)", xticks = (1:length(ns), string.(ns)),
          xticklabelrotation = π / 4)
barplot!(ax, 1:length(ns), times .* 1e6; color = [s ? :steelblue : :darkorange for s in smooth])
fig

# The smooth lengths (blue) take about the time of the power of two. The ones with a large prime
# factor (orange: 1009, 1013, 1021, 1031 and the like) take several times as long, the price of one
# transform of a smooth length of at least twice the size plus the interpolation, still far from
# the ``n^2`` of a direct sum.

# ## How exact it is
#
# Resampling is exact up to rounding, with a larger error than the codelets in `Float64`: up to
# about ``25\,\varepsilon`` of the largest output instead of 1–2 ``\varepsilon`` (`Float64` on
# aarch64 takes Bluestein's algorithm for such a single line, at 2–4 ``\varepsilon``). The error
# on the machine that built this page, against a reference in 128-bit `BigFloat` (FFTL transforms
# that type as well), in units of ``\varepsilon``:

xr = randn(ComplexF64, N)
ref = setprecision(BigFloat, 128) do
  fft(Complex{BigFloat}.(xr))
end
err = Float64(norm(fft(xr) - ref, Inf) / norm(ref, Inf))
err / eps()

# Where that matters, a preference selects Bluestein's algorithm for these lengths instead, at
# 2–4 ``\varepsilon`` and 1.1–1.6 times the time; see
# [Platforms and limits](@ref "Accuracy").
