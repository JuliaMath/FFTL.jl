# Aggregate the interleaved benchmark rounds (bench.txt) into the comparison table:
# C = libfftl from C (bench.c), J = FFTL in Julia (jbench.jl). Each side is reduced to the
# minimum and the median over the rounds, and both ratios C/J are reported. Times in µs.

using Printf
using Statistics: median

path = isempty(ARGS) ? "../build/bench.txt" : ARGS[1]
samples = Dict{NTuple{4,Int},Vector{Tuple{Float64,Float64}}}()
for line in eachline(path)
  m = match(r"bench=([cj]) prec=(\d+) n=(\d+) sign=(-?\d+) create=([\d.]+) exec=([\d.]+)", line)
  m === nothing && continue
  b = m[1] == "c" ? 1 : 2
  key = (b, parse(Int, m[2]), parse(Int, m[3]), parse(Int, m[4]))
  push!(get!(samples, key, Tuple{Float64,Float64}[]),
        (parse(Float64, m[5]), parse(Float64, m[6])))
end

fmt_us(t) = t < 10 ? @sprintf("%.3f", t) : t < 100 ? @sprintf("%.2f", t) : @sprintf("%.1f", t)

println("| prec | n | σ | create C | create J | C/J create | exec C | exec J | C/J exec (min) | C/J exec (med) |")
println("|---|---|---|---|---|---|---|---|---|---|")
for prec in (1, 2), n in (256, 1000, 65536, 1048576, 101), sign in (-1, 1)
  c = get(samples, (1, prec, n, sign), nothing)
  j = get(samples, (2, prec, n, sign), nothing)
  (c === nothing || j === nothing) && continue
  cc = minimum(first.(c)); ce = minimum(last.(c))
  jc = minimum(first.(j)); je = minimum(last.(j))
  cm = median(first.(c)); cem = median(last.(c))
  jm = median(first.(j)); jem = median(last.(j))
  @printf("| %s | %d | %+d | %s | %s | %.2f | %s | %s | %.2f | %.2f |\n",
          prec == 1 ? "f64" : "f32", n, sign,
          fmt_us(cc / 1e3), fmt_us(jc / 1e3), cc / jc,
          fmt_us(ce / 1e3), fmt_us(je / 1e3), ce / je, cem / jem)
end
println()
println("create/execute times in µs (min over rounds, median in the ratios: exec column med);")
println("C = libfftl from C, J = FFTL in Julia.")
