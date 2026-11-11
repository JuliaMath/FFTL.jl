# One measurement process: one tier, one thread count, one machine profile. Writes one raw row per
# library, case and round, plus a .meta with the provenance of the run. Statistics are computed
# later by aggregate.jl; nothing is aggregated here.
#
#   julia --project=. -t 8 harness/measure.jl TIER MACHINE NTHREADS OUT.csv
#
# Run from bench/. run.jl starts this process once per thread count. Julia must have at least
# NTHREADS threads (-t). The library setups, references and shape lists come from overview/run_kinds.jl,
# which is only included here (it runs its main part only as a script).

include(joinpath(@__DIR__, "..", "overview", "run_kinds.jl"))
include(joinpath(@__DIR__, "catalog.jl"))
include(joinpath(@__DIR__, "fftw_threads.jl"))   # FFTW with its own threads (see the file)

using TOML, Dates
include(joinpath(@__DIR__, "profiles.jl"))

const TIER_NAME, MACHINE, NT, OUT = ARGS[1], ARGS[2], parse(Int, ARGS[3]), ARGS[4]
const TIER = TOML.parsefile(joinpath(@__DIR__, "tiers", TIER_NAME * ".toml"))
const PROFILE = TOML.parsefile(profile_path(MACHINE))
const DEVICE = get(PROFILE, "device", "cpu")   # "cpu", "metal" or "cuda" (gpu.jl)
DEVICE == "cpu" || include(joinpath(@__DIR__, "gpu.jl"))

# Median of the rounds of one case is taken in aggregate.jl; a round takes the minimum over its
# samples, and each sample times r calls so that it lasts at least 50 µs (as in the overview).
function round_time(run, budget)
  run()
  r = 1
  while (t = @elapsed repeat_run(run, r)) < 50e-6
    r *= 2
  end
  best = t / r; n = 1; t0 = time()
  while n < 3 || time() - t0 < budget
    best = min(best, (@elapsed repeat_run(run, r)) / r)
    n += 1
  end
  best
end

# Commit of the FFTL checkout that is loaded, with a marker when the tree has local changes.
function fftl_version()
  d = dirname(dirname(pathof(FFTL)))
  rev = readchomp(`git -C $d rev-parse --short HEAD`)
  dirty = isempty(readchomp(`git -C $d status --porcelain --untracked-files=no`)) ? "" : " (dirty)"
  rev * dirty
end

# Load average of the 1-minute column, as printed by uptime (decimal comma in German locales).
function load_average()
  m = match(r"load averages?:\s*([0-9]+[.,][0-9]+)", readchomp(`uptime`))
  m === nothing ? "unknown" : replace(m.captures[1], "," => ".")
end

timestamp() = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")

# GPU run: the header, then gpu.jl measures the catalog (the CPU setups above are not used).
function gpu_main()
  cases = catalog(TIER)
  meta = replace(OUT, r"\.csv$" => ".meta")
  start, load_start = timestamp(), load_average()
  libs = get(PROFILE, "libs", String[])
  open(meta, "w") do io
    println(io, "label: ", PROFILE["label"])
    println(io, "tier: ", TIER_NAME, "; rounds ", TIER["rounds"], "; device ", DEVICE)
    println(io, "threads: 0 (GPU)")
    println(io, "FFTL ", fftl_version(), "; julia ", VERSION)
    println(io, "libraries: ", join(libs, ","))
    println(io, "cases: ", length(cases), " (kind, shape, family)")
    println(io, "start: ", start, "; load: ", load_start)
  end
  isfile(OUT) || open(io -> println(io, "machine,tier,threads,lib,kind,T,dims,shape,n,family,round,seconds,plan_seconds,relerr"), OUT, "w")
  gpu_measure(OUT, cases, libs, TIER["rounds"], PROFILE["label"], TIER_NAME)
  open(io -> println(io, "end: ", timestamp(), "; load: ", load_average()), meta, "a")
end

function main()
  DEVICE == "cpu" || return gpu_main()
  libnames = Symbol.(split(get(ENV, "OVERVIEW_LIBS", "fftl,fftw_measure,ducc,vdsp"), ","))
  DUCC_AVAILABLE || (:ducc in libnames) && @warn "ducc0 binding not built (ducc-ffi/build.sh): ducc0 is left out"
  libs = Tuple(l for l in (:fftl, :fftw_measure, :ducc, :vdsp) if l in libnames && (l != :vdsp || Sys.isapple()) &&
               (l != :ducc || DUCC_AVAILABLE))
  (:fftl in libnames || :ducc in libnames) && Threads.nthreads() < NT &&
    error("start Julia with at least $NT threads (-t $NT)")
  :fftw_measure in libnames && fftw_pthreads!()   # before the first FFTW plan
  cases = catalog(TIER)
  meta = replace(OUT, r"\.csv$" => ".meta")
  start, load_start = timestamp(), load_average()
  open(meta, "w") do io
    println(io, "label: ", PROFILE["label"])
    println(io, "tier: ", TIER_NAME, "; rounds ", TIER["rounds"], "; budget ", TIER["budget"], " s; stride ", TIER["stride"], "; A/A ", TIER["aa"])
    println(io, "threads: ", NT, "; pin: ", get(ENV, "HARNESS_PIN", "none"))
    println(io, "FFTL ", fftl_version(), "; julia ", VERSION)
    println(io, "libraries: ", join(libs, ","), TIER["aa"] ? ",fftl_aa (A/A)" : "")
    println(io, "cases: ", length(cases), " (kind, shape, family), each in both precisions")
    println(io, "start: ", start, "; load: ", load_start)
  end
  isfile(OUT) || open(io -> println(io, "machine,tier,threads,lib,kind,T,dims,shape,n,family,round,seconds,plan_seconds,relerr"), OUT, "w")

  total = 2 * length(cases)
  done = 0
  for (kind, shape, family) in cases, T in (Float64, Float32)
    # Progress for monitor.jl: completed (case, precision) pairs out of the total.
    println("PROGRESS $done $total"); flush(stdout)
    done += 1
    n = prod(shape)
    xr = randn(MersenneTwister(1), T, shape...)
    x = kind == "c2c" ? randn(MersenneTwister(1), Complex{T}, shape...) :
        kind == "c2r" ? Complex{T}.(FFTW.rfft(Float64.(xr))) : xr
    FFTW.set_num_threads(NT)
    ref = reference(kind, x, shape[1])
    ysz = kind == "r2c" ? (shape[1] ÷ 2 + 1, shape[2:end]...) : shape
    yT = kind in ("c2c", "r2c") ? Complex{T} : T
    entries = [(string(lib), lib) for lib in libs]
    TIER["aa"] && :fftl in libs && push!(entries, ("fftl_aa", :fftl))

    prepared = []
    for (name, lib) in entries
      y = Array{yT}(undef, ysz)
      s = try setup(lib, kind, x, y, NT) catch e; @warn "$lib $kind $shape $T" e; nothing end
      s === nothing && continue
      run, tplan = s
      run()
      scale = lib == :vdsp && kind == "r2r_REDFT10" ? 2 : 1
      err = maximum(abs.(scale .* ComplexF64.(y) .- ref)) / maximum(abs.(ref))
      push!(prepared, (name = name, run = run, tplan = tplan, err = err))
    end
    isempty(prepared) && continue

    rows = IOBuffer()
    for round in 1:TIER["rounds"]
      # Alternate the order of the libraries from round to round, so no library always runs first.
      order = isodd(round) ? prepared : reverse(prepared)
      for p in order
        t = round_time(p.run, TIER["budget"])
        println(rows, join((PROFILE["label"], TIER_NAME, NT, p.name, kind, T, length(shape), join(shape, "x"), n,
                            family, round, @sprintf("%.4e", t), @sprintf("%.3e", p.tplan), @sprintf("%.2e", p.err)), ","))
      end
    end
    open(io -> write(io, take!(rows)), OUT, "a")
    any(p -> p.name == "fftw_measure", prepared) && forget_wisdom()
    GC.gc(false)
  end
  open(meta, "a") do io
    println(io, "end: ", timestamp(), "; load: ", load_average())
  end
  println("PROGRESS $total $total"); flush(stdout)
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
