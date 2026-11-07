# Overview benchmark data shared by the two figures of the benchmarks page: panels, competitors,
# file choice and loading.
#
# The data lives in bench/results/ (versioned with the package): one CSV and one .meta per
# overview run. The .meta carries the provenance (FFTL commit, Julia version, machine, load
# averages, start and end); its `end:` line marks a finished run. Fresh data is produced by the
# workspace benchmark harness (FastJuliaFFT/bench/overview) and copied in here.

const BENCH = @__DIR__
const RES = joinpath(BENCH, "results")

const PANELS = [
    (key = "m1t1",   title = "Apple M1 Max CPU, 1 thread",    file = "overview-m1-t1-kinds-8e31d3a.csv", comps = ["fftw_measure", "ducc", "vdsp"]),
    (key = "epyct1", title = "AMD EPYC 7702 CPU, 1 thread",   file = "overview-reco2-t1-kinds-8e31d3a.csv", comps = ["fftw_measure", "ducc"]),
    (key = "m1gpu",  title = "Apple M1 Max GPU",              file = "overview-m1-gpu-kinds-0ab2ed0.csv", comps = ["metal", "vkfft"]),
    (key = "m1t8",   title = "Apple M1 Max CPU, 8 threads",   file = "overview-m1-t8-kinds-8e31d3a.csv", comps = ["fftw_measure", "ducc"], single = "overview-m1-t1-kinds-8e31d3a.csv"),
    (key = "epyct16", title = "AMD EPYC 7702 CPU, 16 threads", file = "overview-reco2-t16-kinds-8e31d3a.csv", comps = ["fftw_measure", "ducc"], single = "overview-reco2-t1-kinds-8e31d3a.csv"),
    (key = "a6000",  title = "NVIDIA RTX A6000 GPU",          file = "overview-a6000-gpu.csv",          comps = ["cufft", "vkfft"]),
]
const COMPNAME = Dict("fftw_measure" => "FFTW", "ducc" => "ducc0", "vdsp" => "vDSP",
                      "metal" => "MPSGraph", "cufft" => "cuFFT",
                      "vkfft" => "VkFFT")

# The all-kinds overview (`<name>-kinds.csv`, column `kind`) replaces the c2c-only file once its
# run has finished, i.e. its `.meta` has an `end:` line; an aborted run keeps the c2c-only file.
# VkFFT enters per call; its batched timing (lib vkfft_batch) amortises launches over many
# transforms in one submission and is not compared with FFTL's per-call time.
function datafile(f)
    k = occursin("-kinds", f) ? f : replace(f, ".csv" => "-kinds.csv")
    meta = joinpath(RES, replace(k, ".csv" => ".meta"))
    finished = isfile(meta) && any(startswith("end:"), eachline(meta))
    isfile(joinpath(RES, k)) && finished ? k : f
end

# (kind, T, dims, shape) => lib => seconds
function load(file)
    rows = Dict{NTuple{4,String},Dict{String,Float64}}()
    lines = eachline(joinpath(RES, datafile(file)))
    hdr = Dict(String(h) => i for (i, h) in enumerate(split(first(lines), ",")))
    for l in lines
        c = split(l, ",")
        length(c) >= length(hdr) || continue
        kind = haskey(hdr, "kind") ? String(c[hdr["kind"]]) : "c2c"
        key = (kind, String(c[hdr["T"]]), String(c[hdr["dims"]]), String(c[hdr["shape"]]))
        # cuFFT is timed through CUDA.jl ("cufft") and from a native harness ("cufft_native");
        # it enters at the faster of the two.
        lib = String(c[hdr["lib"]]) == "cufft_native" ? "cufft" : String(c[hdr["lib"]])
        t = parse(Float64, c[hdr["seconds"]])
        r = get!(rows, key, Dict{String,Float64}())
        r[lib] = min(get(r, lib, Inf), t)
    end
    return rows
end
