# Aggregates the raw rows of measure.jl into one statistics table. Figures read this table; nothing
# here plots.
#
#   julia --project=. harness/aggregate.jl OUT.csv RAW.csv [RAW.csv ...]
#   KEEP=fftw_measure,ducc julia --project=. harness/aggregate.jl OUT.csv RAW.csv ...
#
# All raw files must come from one machine and one tier; they may differ in thread count.
# KEEP lists the competitors that take part (FFTL is always in); default: every library present.
#
# Per case and library the thread count with the lowest median over rounds is used: each library
# at its best measured thread count, the same rule for all. For library L the ratio of round r is
#   min over the other kept libraries of their round-r time / L's round-r time,
# so > 1 means L is faster than the best of the others in that round. For FFTL this is the speed-up
# of the speed-up figure; for every library it is the fold change of the standing figure.
# The point estimate is the median over rounds, the interval a 95 % bootstrap over rounds (needs at
# least three rounds). The A/A pair (fftl_aa) gives the ratio of two identical FFTL runs; its
# interval is the band of "no difference" for the same case.

using Random, Statistics, Printf

const RELTOL = Dict("Float64" => 1e-10, "Float32" => 1e-4)  # max-norm error against FFTW in F64
const CLASS_LIMITS = (2^12, 2^18)                             # short ≤ 2^12 < medium ≤ 2^18 < long
const BOOT = 2000
const AA = "fftl_aa"

size_class(n) = n <= CLASS_LIMITS[1] ? "short" : n <= CLASS_LIMITS[2] ? "medium" : "long"

# One run: one library, one case, one thread count, times by round.
struct Run
  threads::String
  seconds::Dict{Int,Float64}
  relerr::Float64
  plan::Float64
end

median_seconds(r::Run) = median(values(r.seconds))

# Median of x and a 95 % bootstrap interval of that median.
function summary(x)
  n = length(x)
  n == 0 && return (NaN, NaN, NaN)
  n < 3 && return (median(x), NaN, NaN)
  rng = MersenneTwister(1)
  meds = [median(x[rand(rng, 1:n, n)]) for _ in 1:BOOT]
  (median(x), quantile(meds, 0.025), quantile(meds, 0.975))
end

# Returns the runs keyed by (kind, T, shape, lib, threads), the n and the family of each case,
# the machine label and the tier. Raw files from before the family column count as "tuned".
function readraw(paths)
  runs = Dict{NTuple{5,String},Run}()
  ns = Dict{NTuple{3,String},Int}()
  fam = Dict{NTuple{3,String},String}()
  machines, tiers = Set{String}(), Set{String}()
  for p in paths
    lines = filter(!isempty, readlines(p))
    hdr = Dict(String(h) => i for (i, h) in enumerate(split(lines[1], ",")))
    for l in lines[2:end]
      c = split(l, ",")
      push!(machines, c[hdr["machine"]]); push!(tiers, c[hdr["tier"]])
      case = (c[hdr["kind"]], c[hdr["T"]], c[hdr["shape"]])
      ns[case] = parse(Int, c[hdr["n"]])
      fam[case] = haskey(hdr, "family") ? c[hdr["family"]] : "tuned"
      key = (case..., c[hdr["lib"]], c[hdr["threads"]])
      r = get!(runs, key) do
        Run(c[hdr["threads"]], Dict{Int,Float64}(), 0.0, parse(Float64, c[hdr["plan_seconds"]]))
      end
      round = parse(Int, c[hdr["round"]])
      haskey(r.seconds, round) && error("duplicate round $round for $key in $p")
      r.seconds[round] = parse(Float64, c[hdr["seconds"]])
      runs[key] = Run(r.threads, r.seconds, max(r.relerr, parse(Float64, c[hdr["relerr"]])), r.plan)
    end
  end
  length(machines) == 1 && length(tiers) == 1 ||
    error("raw files mix machines or tiers: $(collect(machines)) $(collect(tiers))")
  (runs = runs, ns = ns, fam = fam, machine = only(machines), tier = only(tiers))
end

function aggregate(d; keep = String[])
  allowed(l) = l == AA || isempty(keep) || l in keep
  best = Dict{NTuple{4,String},Run}()   # (kind, T, shape, lib) => run at the best thread count
  for (k, r) in d.runs
    allowed(k[4]) || continue
    bk = k[1:4]
    if !haskey(best, bk) || median_seconds(r) < median_seconds(best[bk])
      best[bk] = r
    end
  end
  cases = sort(unique([k[1:3] for k in keys(best)]))
  rows = NamedTuple[]
  for case in cases
    kind, T, shape = case
    libs = sort([k[4] for k in keys(best) if k[1:3] == case])
    comps = filter(l -> l != AA, libs)
    for L in comps
      r = best[(kind, T, shape, L)]
      others = [best[(kind, T, shape, o)] for o in comps if o != L]
      ratios = Float64[]
      for k in sort(collect(keys(r.seconds)))
        ts = [o.seconds[k] for o in others if haskey(o.seconds, k)]
        isempty(ts) || push!(ratios, minimum(ts) / r.seconds[k])
      end
      med, lo, hi = summary(ratios)
      aa = (L == "fftl" && AA in libs) ? best[(kind, T, shape, AA)] : nothing
      aa_r = aa === nothing ? Float64[] :
             [aa.seconds[k] / r.seconds[k] for k in sort(collect(keys(r.seconds))) if haskey(aa.seconds, k)]
      amed, alo, ahi = summary(aa_r)
      n = d.ns[case]
      push!(rows, (machine = d.machine, tier = d.tier, kind = kind, T = T, shape = shape, n = n,
                   class = size_class(n), family = d.fam[case], lib = L, threads = r.threads,
                   rounds = length(r.seconds), median_seconds = median_seconds(r),
                   ratio = med, ratio_lo = lo, ratio_hi = hi,
                   valid = r.relerr <= RELTOL[T] && all(o.relerr <= RELTOL[T] for o in others),
                   relerr = r.relerr, aa_ratio = amed, aa_lo = alo, aa_hi = ahi))
    end
  end
  rows
end

function write_table(path, rows)
  open(path, "w") do io
    println(io, "machine,tier,kind,T,shape,n,class,family,lib,threads,rounds,median_seconds,ratio,ratio_lo,ratio_hi,valid,relerr,aa_ratio,aa_lo,aa_hi")
    for r in rows
      println(io, join((r.machine, r.tier, r.kind, r.T, r.shape, r.n, r.class, r.family, r.lib, r.threads, r.rounds,
                        @sprintf("%.4e", r.median_seconds), @sprintf("%.4f", r.ratio), @sprintf("%.4f", r.ratio_lo),
                        @sprintf("%.4f", r.ratio_hi), r.valid, @sprintf("%.2e", r.relerr),
                        @sprintf("%.4f", r.aa_ratio), @sprintf("%.4f", r.aa_lo), @sprintf("%.4f", r.aa_hi)), ","))
    end
  end
end

# Per class and library: how many valid cases are faster, slower or tied (interval excludes 1 or not).
function print_summary(rows)
  println("rows: ", length(rows), ", invalid (relerr above tolerance): ", count(r -> !r.valid, rows))
  rows = filter(r -> r.valid, rows)
  for cls in ("short", "medium", "long"), lib in sort(collect(Set(r.lib for r in rows)))
    rs = filter(r -> r.class == cls && r.lib == lib, rows)
    isempty(rs) && continue
    haveci = filter(r -> !isnan(r.ratio_lo), rs)
    faster = count(r -> r.ratio_lo > 1, haveci)
    slower = count(r -> r.ratio_hi < 1, haveci)
    @printf("%-7s %-12s cases %3d  median ratio %.3f  faster %3d  slower %3d  tie %3d  (no interval %d)\n",
            cls, lib, length(rs), median([r.ratio for r in rs]), faster, slower,
            length(haveci) - faster - slower, length(rs) - length(haveci))
  end
  aa = [r.aa_ratio for r in rows if r.lib == "fftl" && !isnan(r.aa_ratio)]
  isempty(aa) || @printf("A/A (FFTL vs FFTL) median ratio per case: 2.5 %% %.3f, 97.5 %% %.3f over %d cases\n",
                         quantile(aa, 0.025), quantile(aa, 0.975), length(aa))
end

function main(args)
  length(args) >= 2 || error("usage: aggregate.jl OUT.csv RAW.csv [RAW.csv ...]")
  keep = filter(!isempty, split(get(ENV, "KEEP", ""), ","))
  d = readraw(args[2:end])
  rows = aggregate(d; keep = isempty(keep) ? String[] : vcat(["fftl"], keep))
  write_table(args[1], rows)
  println("wrote ", args[1], " (", length(rows), " rows)")
  print_summary(rows)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
