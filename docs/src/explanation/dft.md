# The discrete Fourier transform

A sequence ``x_0, …, x_{n-1}`` of complex numbers has the discrete Fourier transform (DFT)

```math
X_k = \sum_{j=0}^{n-1} x_j\, e^{-2\pi i\, jk/n}, \qquad k = 0, \dots, n-1.
```

``X_k`` measures how strongly the sequence correlates with the complex exponential of frequency
``k`` cycles per record. The inverse maps back:

```math
x_j = \frac{1}{n} \sum_{k=0}^{n-1} X_k\, e^{+2\pi i\, jk/n}.
```

Both sums are matrix-vector products with the ``n × n`` matrix ``F_{kj} = e^{-2\pi i jk/n}``. The
matrix costs ``n^2`` multiplications, which is the naive algorithm. The point of the FFT is that
``F`` factors into sparse pieces, so the product costs ``O(n \log n)``.

## Conventions

FFTL uses the two signs and the two normalisations that are common in signal processing:

| function | sign | normalisation |
|:--|:--|:--|
| `fft` | ``-`` (forward) | none |
| `bfft` | ``+`` (backward) | none |
| `ifft` | ``+`` | ``1/n`` per transformed dimension |

With these, ``\text{ifft}(\text{fft}(x)) = x``, and ``\text{bfft}(\text{fft}(x)) = n\,x``. FFTW.jl uses the same
names and the same scalings. The `dct` of FFTL is the orthonormal one, the convention that keeps
the energy unchanged.

## Why a Fourier transform at all

For a signal sampled at rate ``f_s`` with ``n`` samples, bin ``k`` is the frequency ``k f_s / n``.
A tone that is not exactly on a bin spreads over several bins. This is the leakage that windows
trade off against resolution; the [spectrum tutorial](@ref "Spectrum of a sampled signal") uses
a record that contains whole cycles, so none occurs.

The bins above ``n/2`` are the negative frequencies. For real input ``X_{n-k} = \overline{X_k}``,
so the spectrum of a real signal carries only half its information, and `rfft` returns that half.

Multiplying spectra is convolution in time: the cyclic convolution of two sequences is the
inverse transform of the product of their transforms. A linear convolution is the cyclic one of
zero-padded inputs. This turns filtering of long signals into ``O(n \log n)`` work; the
[convolution tutorial](@ref "Fast convolution") shows it.

## The algorithm as divide and conquer

The idea that makes the FFT fast is the one behind merge sort. Split the sum by the parity of
the index, ``j = 2m`` and ``j = 2m+1``:

```math
X_k = \sum_{m} x_{2m}\, e^{-2\pi i\, mk/(n/2)} + e^{-2\pi i k/n} \sum_{m} x_{2m+1}\, e^{-2\pi i\, mk/(n/2)}.
```

Each sum is a DFT of length ``n/2``, and the two are combined with ``n`` multiplications by the
twiddle factors ``e^{-2\pi i k/n}``. Recursing on the halves gives
``T(n) = 2T(n/2) + O(n)``, so ``T(n) = O(n \log n)``. The same identity works for any factor
``n = r \cdot m``, not only 2. Radix ``r`` splits into a stage of ``r``-point transforms. The
[algorithms explanation](@ref "Algorithms") describes how FFTL turns this recursion into a
sequence of loops over memory without recursion.
