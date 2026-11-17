# News

## 0.3.0 (unreleased)

- Lengths with prime factors above 73: the large-prime part is resampled onto a smooth grid
  (one transform of about twice its length) instead of Bluestein's algorithm (two such
  transforms); batched over the lanes of a composite length, and on threads for a long line (#29).
  Faster on most such lengths. `Float64` on aarch64 keeps Bluestein's algorithm for single lines.
- Accuracy for such lengths: up to about 25 ε in `Float64` (largest error over the largest output,
  about 5·10⁻¹⁵; FFTL 0.2: 2–4 ε), up to 2.5 ε in `Float32` (more accurate than before). The
  preference `float64_large_primes = "bluestein"` restores Bluestein's algorithm for every `Float64`
  transform of such a length, at 1.1–1.6 times the time on most of them (#47, #49).
- Four-step transforms of long lines with cache buffers; the row copy and the non-temporal output
  only with 64-byte vectors (#33).
- Julia 1.10 is supported again (0.2.1, #45).
