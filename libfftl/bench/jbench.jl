# Interleaved timing, Julia side: FFTL in Julia, one measurement round per invocation.
# Same cases and protocol as bench.c: same sizes, signs and precisions, batches of at least
# 50 µs, times in ns per call. The create batch mirrors the plan of the library: kernel plus
# two work buffers; the execute batch is run! out of place with S = 1, as the library's flat
# runners do it.
#
# Fresh process per variant and round on purpose: the JIT specialisations have to be compiled
# anyway, and expanding run! for many chain variants in one Julia process on this shared
# machine proved to crash the GC (see notes/libfftl.md, first bench round). One variant per
# process isolates that; the process is short and small enough to stay clear of it. GC stays
# enabled around the timed batches (a collection inside a batch only inflates that round;
# the minimum and median over rounds in aggreg.jl are unaffected).

using FFTL
using FFTL: kernel, run!, cptr
using Printf

const MIN_BATCH_NS = 50_000

function bench_one(prec::Int, n::Int, sign::Int)
  T = prec == 1 ? Float64 : Float32
  CT = Complex{T}
  x = Vector{CT}(undef, n)
  y = Vector{CT}(undef, n)
  for j in 1:n
    x[j] = (mod1(j, 7) - 4) + (mod1(j, 5) - 3) * im
  end

  # warmup (JIT: kernel construction and the chain specialisations of run!)
  GC.gc()
  for _ in 1:2
    kernel(T, n, sign)
    Vector{CT}(undef, n)
    Vector{CT}(undef, n)
  end
  k = kernel(T, n, sign)
  w1 = Vector{CT}(undef, n)
  w2 = Vector{CT}(undef, n)
  for _ in 1:3
    run!(cptr(y), cptr(x), cptr(w1), cptr(w2), k, 1)
  end

  # create (kernel + two work buffers), batch of >= 50 µs
  GC.gc()
  t0 = time_ns()
  c = 0
  while true
    kernel(T, n, sign)
    Vector{CT}(undef, n)
    Vector{CT}(undef, n)
    c += 1
    time_ns() - t0 ≥ MIN_BATCH_NS && break
  end
  create_ns = (time_ns() - t0) / c

  # execute, batch of >= 50 µs on the warmed plan (run! does not allocate)
  GC.gc()
  t1 = time_ns()
  m = 0
  while true
    run!(cptr(y), cptr(x), cptr(w1), cptr(w2), k, 1)
    m += 1
    time_ns() - t1 ≥ MIN_BATCH_NS && break
  end
  exec_ns = (time_ns() - t1) / m

  @printf(stdout, "bench=j prec=%d n=%d sign=%d create=%.4f exec=%.4f\n", prec, n, sign,
          create_ns, exec_ns)
  flush(stdout)
  return 0
end

if isempty(ARGS)
  for prec in (1, 2), n in (256, 1000, 65536, 1048576, 101), sign in (-1, 1)
    bench_one(prec, n, sign)
  end
else
  bench_one(parse(Int, ARGS[1]), parse(Int, ARGS[2]), parse(Int, ARGS[3]))
end
