# Runs one tier on one machine profile: one measure.jl process per thread count, pinned where the
# profile says so, all output in one directory for this run.
#
#   julia --project=. harness/run.jl MACHINE TIER [--threads=1,16] [--force]
#
# Examples (from bench/):
#   julia --project=. harness/run.jl m1max smoke
#   julia --project=. harness/run.jl mymachine dev --threads=16
#
# Output: harness/out/<machine>/<tier>/<timestamp>/t<threads>.csv and .meta
# (out/ is not versioned; see harness/.gitignore).

using TOML, Dates
include(joinpath(@__DIR__, "profiles.jl"))

const HERE = @__DIR__
const BENCH = normpath(joinpath(HERE, ".."))

function parse_args(args)
  pos, opts = String[], Dict{String,String}()
  for a in args
    if startswith(a, "--")
      key, val = occursin("=", a) ? split(a[3:end], "="; limit = 2) : (a[3:end], "")
      opts[key] = val
    else
      push!(pos, a)
    end
  end
  length(pos) == 2 || error("usage: run.jl MACHINE TIER [--threads=1,16] [--force]")
  pos[1], pos[2], opts
end

# 1-minute load average, decimal comma or point (see measure.jl).
function load1()
  m = match(r"load averages?:\s*([0-9]+[.,][0-9]+)", readchomp(`uptime`))
  m === nothing ? NaN : parse(Float64, replace(m.captures[1], "," => "."))
end

function project_path(p)
  startswith(p, "~") && return expanduser(p)
  isabspath(p) ? p : normpath(joinpath(BENCH, p))
end

function main(args)
  machine, tier, opts = parse_args(args)
  profile = TOML.parsefile(profile_path(machine))
  isfile(joinpath(HERE, "tiers", tier * ".toml")) || error("no tier $tier in harness/tiers")

  threads = haskey(opts, "threads") ? parse.(Int, split(opts["threads"], ",")) : Int.(profile["threads"])
  load = load1()
  if load > profile["max_load"] && !haskey(opts, "force")
    error("load $load is above max_load $(profile["max_load"]) of $machine; wait or pass --force")
  end

  stamp = Dates.format(now(), dateformat"yyyy-mm-ddTHHMMSS")
  dir = joinpath(HERE, "out", machine, tier, stamp)
  mkpath(dir)
  julia = joinpath(Sys.BINDIR, "julia")
  project = project_path(profile["project"])
  for nt in threads
    pin = get(profile["pin"], string(nt), nothing)
    out = joinpath(dir, "t$nt.csv")
    cmd = `$julia --startup-file=no --project=$project -t $nt $(joinpath(HERE, "measure.jl")) $tier $machine $nt $out`
    if pin !== nothing && Sys.islinux()
      cmd = `taskset -c $pin $cmd`
    end
    cmd = addenv(cmd, "HARNESS_PIN" => pin === nothing ? "none" : (Sys.islinux() ? pin : "not available on this OS"))
    println("run $machine $tier threads=$nt pin=$(pin === nothing ? "none" : pin) -> $out")
    run(cmd)
  end
  println("done: ", dir)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
