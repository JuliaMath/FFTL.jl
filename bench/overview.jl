# Panels of the two benchmark figures. The panels are read from panels.toml; each one is an
# aggregated table, results/panel-<key>.csv, with one row per case and library. The tables come
# from the benchmark harness (aggregate.jl), or for the legacy panels from its import script.
# The figures read nothing else.
#
# A row carries the speed-up `ratio` of its library against the best other library of the panel
# (> 1: faster), the 95 % bootstrap interval `ratio_lo`/`ratio_hi` (NaN with fewer than three
# rounds), and `valid` (false above the accuracy tolerance; the figures drop those rows).

using TOML

const BENCH = @__DIR__
const RES = joinpath(BENCH, "results")
const COMPNAME = Dict("fftl" => "FFTL", "fftw_measure" => "FFTW", "ducc" => "ducc0", "vdsp" => "vDSP",
                      "mpsgraph" => "MPSGraph", "metal" => "MPSGraph", "cufft" => "cuFFT", "vkfft" => "VkFFT")

const PANELS = [(key = p["key"], title = p["title"], machine = p["machine"], comps = Vector{String}(p["competitors"]))
                for p in TOML.parsefile(joinpath(BENCH, "panels.toml"))["panel"]]

# Smooth lengths: every dimension has prime factors of at most 31 (FFTL's codelet limit, MAX_RADIX_PRIME).
const CODELET_LIMIT = 31
function largest_prime_factor(n)
  m, p, last = n, 2, 1
  while p * p <= m
    while m % p == 0
      last = p; m ÷= p
    end
    p += 1
  end
  m > 1 ? max(last, m) : last
end
is_smooth(shape::AbstractString) = all(d -> largest_prime_factor(parse(Int, d)) <= CODELET_LIMIT, split(shape, "x"))

# Rows of one panel as NamedTuples. Invalid rows stay in: the callers filter on `valid`.
function load_panel(p)
  lines = filter(!isempty, readlines(joinpath(RES, "panel-$(p.key).csv")))
  hdr = Dict(String(h) => i for (i, h) in enumerate(split(lines[1], ",")))
  num(c, k) = parse(Float64, c[hdr[k]])
  rows = NamedTuple[]
  for l in lines[2:end]
    c = split(l, ",")
    push!(rows, (kind = String(c[hdr["kind"]]), T = String(c[hdr["T"]]), n = parse(Int, c[hdr["n"]]),
                 class = String(c[hdr["class"]]), family = String(c[hdr["family"]]),
                 lib = String(c[hdr["lib"]]), ratio = num(c, "ratio"), ratio_lo = num(c, "ratio_lo"),
                 ratio_hi = num(c, "ratio_hi"), valid = c[hdr["valid"]] == "true",
                 aa_ratio = num(c, "aa_ratio"), shape = String(c[hdr["shape"]]),
                 smooth = is_smooth(String(c[hdr["shape"]])), held = c[hdr["family"]] == "heldout"))
  end
  rows
end

# Libraries of a panel, FFTL first, competitors in panels.toml order.
present_libs(rows, p) = [l for l in ["fftl"; p.comps] if any(r -> r.lib == l, rows)]
