# The cases of bench.c through FFTL's RPlan in Julia (the code the library runs).
using FFTL
function tmin(f)
  f(); best = Inf
  for _ in 1:15
    r = 1; t = 0.0
    while (t = @elapsed(for _ in 1:r; f(); end)) < 2e-3; r *= 2; end
    best = min(best, t / r)
  end
  best
end
for (rank, n) in ((1, 1024), (1, 1000), (1, 65536), (1, 1048576), (1, 101), (2, 512), (3, 64)), nt in (1, 8)
  sz = fill(n, rank); x = rand(ComplexF64, sz...); y = similar(x)
  tp = @elapsed p = FFTL.RPlan{Float64}(sz, collect(1:rank), -1; threads = nt)
  t = GC.@preserve x y tmin(() -> FFTL.execute!(p, pointer(y), pointer(x)))
  println("J rank $rank n $n threads $nt plan_ms ", round(1e3tp, digits = 3), " exec_us ", round(1e6t, digits = 2))
end
