# # Your first transform
#
# FFTL.jl computes the discrete Fourier transform (DFT) of arrays with any number of dimensions
# and any length. `using FFTL` makes the transform functions available under the names of
# FFTW.jl (`fft`, `ifft`, `rfft`, `dct`, … and their `plan_` forms).

using FFTL, LinearAlgebra, Random
Random.seed!(1);

# ## A transform checked against the definition
#
# The forward transform of a vector ``x`` of length ``n`` is ``X_k = \sum_{j} x_j e^{-2\pi i jk/n}``.
# Here is that sum computed directly, next to `fft`:

n = 8
x = randn(ComplexF64, n);
X_direct = [sum(x[j + 1] * exp(-2π * im * j * k / n) for j in 0:n-1) for k in 0:n-1]
X = fft(x)
norm(X - X_direct)

# The forward transform is not normalised, the inverse is: `ifft(fft(x)) ≈ x`.
# `bfft` is the unnormalised backward transform.

ifft(X) ≈ x

# ## Plans
#
# `fft(x)` makes a plan and applies it. A plan holds everything that depends only on the
# size and the dimensions: radix sequence, twiddle tables and work buffers. Make it once and
# apply it many times, in place or out of place:

p = plan_fft(x)
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
size(fft(A)), size(fft(A, 1))

# ## Real input
#
# For real data, `rfft` returns the half spectrum: the values for frequencies
# ``0 … n/2`` only, since the others are their complex conjugates.

r = randn(1000)
R = rfft(r)
size(R), irfft(R, 1000) ≈ r

# ## Orthonormal cosine transform
#
# `dct` is the DCT-II scaled to an orthonormal map, so it preserves the Euclidean norm and
# `idct` is its exact inverse:

c = randn(100)
D = dct(c)
norm(D) ≈ norm(c), idct(D) ≈ c

# Next: the [tutorials](@ref "Spectrum of a sampled signal") apply these to real problems, and
# the [how-to guides](@ref "Plans and reuse") explain the options.
