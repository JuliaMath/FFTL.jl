# Single-case A/B timings of one package version: one thread, minimum over blocks of calls.
# Run one process per variant and thread count, pinned, and interleave the variants over rounds.
# Usage: taskset -c N julia --project=<env with FFTL developed from the checkout> -t 1 ab/bench_pr.jl

using FFTL, Printf
function tm(f; reps = 30)
  f(); minimum((t = time_ns(); for _ in 1:reps; f(); end; (time_ns() - t) / 1e9 / reps) for _ in 1:5)
end
cases = [("c2c", (1024,)), ("c2c", (4096,)), ("c2c", (1000,)), ("c2c", (101,)), ("c2c", (1009,)), ("c2c", (37888,)),
         ("c2c", (64576,)), ("c2c", (1009, 1009)), ("c2c", (101, 101, 101)), ("r2c", (37888,)), ("r2c", (92935,)), ("c2c", (12000,))]
for (k, sh) in cases
  x = k == "r2c" ? randn(Float64, sh...) : randn(ComplexF64, sh...)
  p = k == "r2c" ? FFTL.plan_rfft(x) : FFTL.plan_fft(x)
  t = tm(() -> p * x)
  @printf("%s %-14s %9.1f us\n", k, string(sh), 1e6 * t)
end
