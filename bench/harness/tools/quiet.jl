# Quiet windows and CPU blocks from the load history of harness/tools/loadlog.sh.
#   julia bench/harness/tools/quiet.jl <loadlog.csv> [hours = 4]
# Prints the mean busy fraction per hour of day (all CPUs and the least busy 16-CPU block), the
# blocks ordered by mean busy fraction, and the quietest window of <hours> consecutive hours.
using Statistics, Printf

rows = [split(l, ',') for l in readlines(ARGS[1])[2:end]]
w = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
hdr = split(readline(ARGS[1]), ',')[3:end]
hour = [parse(Int, r[1][12:13]) for r in rows]
busy = permutedims(reduce(hcat, [parse.(Float64, r[3:end]) for r in rows]))
println("samples: ", length(rows), " from ", rows[1][1], " to ", rows[end][1])
println("hour  mean busy  least busy block")
hm = Dict{Int, Float64}()
for h in 0:23
  sel = hour .== h
  any(sel) || continue
  b = vec(mean(busy[sel, :]; dims = 1)); j = argmin(b)
  hm[h] = mean(b)
  @printf("%02d    %.2f       %s (%.2f)\n", h, mean(b), hdr[j], b[j])
end
b = vec(mean(busy; dims = 1))
println("blocks by mean busy fraction: ", join((@sprintf("%s %.2f", hdr[j], b[j]) for j in sortperm(b)), ", "))
if length(hm) == 24
  s = [(h, mean(hm[mod(h + k, 24)] for k in 0:w-1)) for h in 0:23]
  h, v = s[argmin(last.(s))]
  @printf("quietest %d-hour window: %02d:00–%02d:00 (mean busy %.2f)\n", w, h, mod(h + w, 24), v)
end
