# Multi-thread A/B timings of one package version: minimum over blocks of calls, threads = ARGS[1].
# Pin to a fixed core set with taskset and start Julia with the same -t, one process per variant.
# Usage: taskset -c 64-79 julia --project=<env> -t 16 ab/bench_mt.jl 16

using FFTL, Printf
nt = parse(Int, ARGS[1])
function tm(f; reps = 10)
  f(); minimum((t = time_ns(); for _ in 1:reps; f(); end; (time_ns() - t) / 1e9 / reps) for _ in 1:5)
end
cases = [("c2c", (37888,)), ("c2c", (64576,)), ("c2c", (92935,)), ("c2c", (1009, 16)), ("c2c", (101, 101)),
         ("c2c", (1009, 1009)), ("c2c", (101, 101, 101)), ("c2c", (1024, 1024)), ("c2c", (2^20,)), ("c2c", (1000003,)),
         ("r2c", (37888,)), ("r2c", (92935,)), ("r2c", (101, 101, 101)), ("r2c", (1024, 1024))]
for (k, sh) in cases
  x = k == "r2c" ? randn(Float64, sh...) : randn(ComplexF64, sh...)
  p = k == "r2c" ? FFTL.plan_rfft(x; threads = nt) : FFTL.plan_fft(x; threads = nt)
  t = tm(() -> p * x)
  @printf("%s %-14s %10.1f us\n", k, string(sh), 1e6 * t)
end
