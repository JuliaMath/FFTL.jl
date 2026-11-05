# # Your first transform
#
# FFTL.jl computes the discrete Fourier transform (DFT) of arrays with any number of dimensions
# and any length. The functions are called qualified, `FFTL.fft`, because FFTL does not export
# them: it can be loaded next to FFTW.jl without the two taking over each other's `fft`.

using FFTL, LinearAlgebra, Random
Random.seed!(1);

# ## A transform checked against the definition
#
# The forward transform of a vector ``x`` of length ``n`` is ``X_k = \sum_{j} x_j e^{-2\pi i jk/n}``.
# Here is that sum computed directly, next to `FFTL.fft`:

n = 8
x = randn(ComplexF64, n);
X_direct = [sum(x[j + 1] * cis(-2π * j * k / n) for j in 0:n-1) for k in 0:n-1]
X = FFTL.fft(x)
norm(X - X_direct)

# The forward transform is not normalised, the inverse is: `FFTL.ifft(FFTL.fft(x)) ≈ x`.
# `FFTL.bfft` is the unnormalised backward transform.

FFTL.ifft(X) ≈ x

# ## Plans
#
# `FFTL.fft(x)` makes a plan and applies it. A plan holds everything that depends only on the
# size and the dimensions: radix sequence, twiddle tables and work buffers. Make it once and
# apply it many times, in place or out of place:

p = FFTL.plan_fft(x)
y = p * x
z = similar(x)
mul!(z, p, x)
y ≈ X, z ≈ X

# Plans are `AbstractFFTs.Plan`s, so `inv(p)` and `p \ y` give the normalised inverse.

p \ X ≈ x

# ## Several dimensions
#
# An array is transformed along all its dimensions by default. A region selects some of them:

A = randn(ComplexF64, 16, 12)
size(FFTL.fft(A)), size(FFTL.fft(A, 1))

# ## Real input
#
# For real data, `FFTL.rfft` returns the half spectrum: the values for frequencies
# ``0 … n/2`` only, since the others are their complex conjugates.

r = randn(1000)
R = FFTL.rfft(r)
size(R), FFTL.irfft(R, 1000) ≈ r

# ## Orthonormal cosine transform
#
# `FFTL.dct` is the DCT-II scaled to an orthonormal map, so it preserves the Euclidean norm and
# `FFTL.idct` is its exact inverse:

c = randn(100)
D = FFTL.dct(c)
norm(D) ≈ norm(c), FFTL.idct(D) ≈ c

# Next: the [tutorials](@ref "Spectrum of a sampled signal") apply these to real problems, and
# the [how-to guides](@ref "Plans and reuse") explain the options.
