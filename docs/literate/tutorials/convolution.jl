# # Fast convolution
#
# A filter that slides over a signal costs ``O(nm)`` operations for signal length ``n`` and
# filter length ``m``. Through the convolution theorem, the convolution is a pointwise product
# of spectra: transform both, multiply, transform back. That costs ``O(N log N)`` with ``N ≥ n + m - 1``.
# The result is linear, not circular, so both inputs are zero-padded first.

using FFTL, Random
Random.seed!(3)

signal = randn(1000)
kernel = [exp(-(k - 50)^2 / 200) for k in 0:100]   # a smooth 101-tap filter
kernel ./= sum(kernel);

# ## Direct sum, for reference

function direct_convolution(x, h)
  y = zeros(length(x) + length(h) - 1)
  for i in eachindex(x), j in eachindex(h)
    y[i + j - 1] += x[i] * h[j]
  end
  y
end

# ## Through the FFT
#
# The padded length is a power of two here. FFTL plans any length, so any ``N ≥ n + m - 1``
# works; powers of two are simply a convenient choice.

n = length(signal) + length(kernel) - 1
N = nextpow(2, n)
xpad = zeros(N); xpad[1:length(signal)] .= signal;
hpad = zeros(N); hpad[1:length(kernel)] .= kernel;

# Real inputs: `rfft` and `irfft` transform the padded signals and their product with half the work

y_fft = FFTL.irfft(FFTL.rfft(xpad) .* FFTL.rfft(hpad), N)[1:n]
y_dir = direct_convolution(signal, kernel)
maximum(abs, y_fft - y_dir)

# The difference is at the level of rounding, so the two agree. Both give the same output, and
# the FFT version scales as ``N log N`` instead of ``nm``.

# ## Filtering a long signal
#
# For a signal much longer than the filter, overlap-add applies the same method block by block.
# The plan for the block length is made once and reused for every block: the
# [plans how-to](@ref "Plans and reuse") explains why that matters.
