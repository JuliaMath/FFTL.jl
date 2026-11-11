# Progress monitor for measurement runs. It starts the runs of several machines and draws one progress
# bar per job (one job = one machine and one thread count).
#
#   julia --project=. harness/monitor.jl m1max mymachine [--tier=dev] [--force]
#   julia --project=. harness/monitor.jl m1max:1,8 mymachine:16 --tier=smoke
#
# An argument is a machine profile (see profiles.jl), optionally with thread counts.
# Machines run in parallel; the thread counts of one machine run one after another. A profile with
# `host` runs over ssh in `remote_root`; its results are copied back into the local output directory
# when the job has finished. Output: harness/out/<machine>/<tier>/<stamp>/t<threads>.csv and .meta.
# Stderr of each job goes to the .log next to it.
#
# Ctrl-C stops all jobs.

using TOML, Dates, Printf
include(joinpath(@__DIR__, "profiles.jl"))

const HERE = @__DIR__
const BENCH = normpath(joinpath(HERE, ".."))

mutable struct Job
  machine::String
  nt::Int
  done::Int
  total::Int
  started::Float64
  state::Symbol             # :waiting, :running, :done, :failed
  msg::String
  proc::Union{Nothing,Base.Process}
  outdir::String            # local directory of the run
end

parse_spec(s) = begin
  id, th = occursin(":", s) ? split(s, ":"; limit = 2) : (s, "")
  id, (isempty(th) ? nothing : parse.(Int, split(th, ",")))
end

function profile(id)
  p = profile_path(id)
  isfile(p) || error("no machine profile $id")
  TOML.parsefile(p)
end

# 1-minute load average from uptime, locally or over ssh (decimal comma or point).
function load1(host)
  out = host === nothing ? readchomp(`uptime`) : readchomp(`ssh -x $host uptime`)
  m = match(r"load averages?:\s*([0-9]+[.,][0-9]+)", out)
  m === nothing ? NaN : parse(Float64, replace(m.captures[1], "," => "."))
end

# Remote profiles are pinned on their Linux host whatever the local OS is.
pin_of(p, nt) = (haskey(p, "host") || Sys.islinux()) ? get(get(p, "pin", Dict()), string(nt), nothing) : nothing

# Highest utilisation (%) of the GPUs of a remote host.
gpu_util(host) = maximum(parse.(Float64, split(readchomp(`ssh -x $host nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`), "\n")))

# Command of one job. Remote jobs run the same measure.jl under remote_root on the host.
function job_command(p, tier, nt, out)
  host = get(p, "host", nothing)
  pin = pin_of(p, nt)
  if host === nothing
    julia = joinpath(Sys.BINDIR, "julia")
    project = startswith(p["project"], "~") ? expanduser(p["project"]) :
              isabspath(p["project"]) ? p["project"] : normpath(joinpath(BENCH, p["project"]))
    measure = joinpath(HERE, "measure.jl")
    cmd = `$julia --startup-file=no --project=$project -t $(max(nt, 1)) $measure $tier $(p["id"]) $nt $out`
    pin === nothing ? cmd : addenv(`taskset -c $pin $cmd`, "HARNESS_PIN" => pin)
  else
    root = p["remote_root"]
    remote_out = "$root/harness/out/$(p["id"])/$tier/$(basename(dirname(out)))/t$nt.csv"
    body = "mkdir -p $(dirname(remote_out)) && cd $root/harness && " *
           (pin === nothing ? "" : "HARNESS_PIN=$pin taskset -c $pin ") *
           "$(p["julia"]) --startup-file=no --project=$(p["project"]) -t $(max(nt, 1)) measure.jl $tier $(p["id"]) $nt $remote_out"
    `ssh -x $host $body`
  end
end

# Waits until the machine is quiet enough for a job: load below max_load and, for GPU profiles on a
# host, GPU utilisation below gpu_max_util. Shows the job as waiting and checks again every 30 s.
# --force skips the checks.
function wait_for_quiet(j, p, host, force)
  force && return
  while true
    l = load1(host)
    busy = l > p["max_load"] ? "load $(round(l; digits = 2)) above max_load $(p["max_load"])" : nothing
    if busy === nothing && haskey(p, "gpu_max_util") && host !== nothing
      u = gpu_util(host)
      u > p["gpu_max_util"] && (busy = "GPU $(u) % busy, above gpu_max_util $(p["gpu_max_util"])")
    end
    busy === nothing && return
    j.state = :waiting; j.msg = busy
    sleep(30)
  end
end

# Runs the jobs of one machine one after another.
function run_machine(jobs, p, tier, force)
  host = get(p, "host", nothing)
  for j in jobs
    try
      wait_for_quiet(j, p, host, force)
      j.started = time()
      j.state = :running
      out = joinpath(j.outdir, "t$(j.nt).csv")
      mkpath(j.outdir)
      cmd = job_command(p, tier, j.nt, out)
      log = joinpath(j.outdir, "t$(j.nt).log")
      j.proc = open(pipeline(cmd; stderr = log), "r")
      for line in eachline(j.proc)
        if startswith(line, "PROGRESS ")
          f = split(line)
          j.done = parse(Int, f[2]); j.total = parse(Int, f[3])
        end
      end
      wait(j.proc)
      if success(j.proc)
        host === nothing || fetch_remote(j, p, tier)
        j.state = :done; j.done = j.total
      else
        j.state = :failed; j.msg = "exit code $(j.proc.exitcode), see $(basename(log))"
      end
    catch e
      j.state = :failed
      j.msg = e isa InterruptException ? "interrupted" : sprint(showerror, e)
    end
  end
end

# Copies the CSV and .meta of a finished remote job into its local output directory.
function fetch_remote(j, p, tier)
  rdir = "$(p["remote_root"])/harness/out/$(p["id"])/$tier/$(basename(j.outdir))"
  for ext in ("csv", "meta")
    run(`scp -q -o ForwardX11=no $(p["host"]):$rdir/t$(j.nt).$ext $(joinpath(j.outdir, "t$(j.nt).$ext"))`)
  end
end

function fmt(t)
  t = round(Int, t)
  h, r = divrem(t, 3600); m, s = divrem(r, 60)
  h > 0 ? @sprintf("%d:%02d:%02d", h, m, s) : @sprintf("%02d:%02d", m, s)
end

function line(j)
  name = rpad("$(j.machine) t$(j.nt)", 16)
  if j.state == :waiting
    return "$name  waiting: $(j.msg)"
  elseif j.state == :failed
    return "$name  FAILED: $(j.msg)"
  end
  frac = j.total > 0 ? j.done / j.total : 0.0
  w = 30
  filled = round(Int, frac * w)
  bar = "[" * "#"^filled * "-"^(w - filled) * "]"
  el = time() - j.started
  eta = j.done > 0 && j.state == :running ? el / j.done * (j.total - j.done) : NaN
  etas = isnan(eta) ? "  ETA --:--" : "  ETA " * fmt(eta)
  status = j.state == :done ? "done" : "$(j.done)/$(j.total)"
  "$name $bar $(lpad(string(round(Int, 100frac)), 3))%  $status  $(fmt(el))$(j.state == :running ? etas : "")"
end

function render(jobs, first)
  first || print("\e[$(length(jobs))A")
  for j in jobs
    print("\r\e[2K", line(j), "\n")
  end
  flush(stdout)
end

function main(args)
  opts = Dict{String,String}()
  specs = String[]
  for a in args
    if startswith(a, "--")
      k, v = occursin("=", a) ? split(a[3:end], "="; limit = 2) : (a[3:end], "")
      opts[k] = v
    else
      push!(specs, a)
    end
  end
  isempty(specs) && error("usage: monitor.jl MACHINE[:THREADS,...] ... [--tier=dev] [--force]")
  tier = get(opts, "tier", "dev")
  isfile(joinpath(HERE, "tiers", tier * ".toml")) || error("no tier $tier")
  force = haskey(opts, "force")
  stamp = Dates.format(now(), dateformat"yyyy-mm-ddTHHMMSS")

  groups = Vector{Pair{Dict,Vector{Job}}}()
  for s in specs
    id, th = parse_spec(s)
    p = profile(id)
    threads = th === nothing ? Int.(p["threads"]) : th
    dir = joinpath(HERE, "out", id, tier, stamp)
    jobs = [Job(id, nt, 0, 0, time(), :waiting, "", nothing, dir) for nt in threads]
    push!(groups, p => jobs)
  end
  all = reduce(vcat, [g[2] for g in groups])

  print("\e[?25l")   # hide the cursor while drawing
  tasks = [@async(run_machine(jobs, p, tier, force)) for (p, jobs) in groups]
  try
    render(all, true)
    while any(j -> j.state in (:waiting, :running), all)
      sleep(0.5)
      render(all, false)
    end
    render(all, false)
  catch e
    if e isa InterruptException
      for j in all
        j.proc !== nothing && process_running(j.proc) && kill(j.proc)
      end
      println("\ninterrupted")
    else
      rethrow()
    end
  finally
    print("\e[?25h")
  end
  println("\nresults in ", joinpath(HERE, "out"), " (",
          join(unique([j.outdir for j in all]), "; "), ")")
  for j in all
    j.state == :failed && println("FAILED ", j.machine, " t", j.nt, ": ", j.msg)
  end
end

main(ARGS)
