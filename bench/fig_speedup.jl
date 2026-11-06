# Speed-up of FFTL over the fastest other library on each platform, one panel per platform.
#
# Same data pipeline, competitors and fair thread rule as the FFTL paper's Fig. 1 (its
# scripts/fig1.jl; docs sizing and PNG output instead of the Nature figure). Speed-up = time of
# the fastest competitor for that case / time of FFTL; > 1 means FFTL is faster. Competitors per
# platform: CPU: FFTW (MEASURE), ducc0 and, where measured, Apple vDSP; GPU: the vendor FFT
# (Apple's MPSGraph through Metal.jl's plans, cuFFT) and VkFFT. One marker per case; in the
# multi-thread panels each library is timed at its better setting per case (the minimum over one
# and all threads), so FFTW is not charged for spreading tiny transforms over many threads.
# Transform kinds: column `kind` in the overview CSVs (files without it are c2c only). Marker
# shape is the kind group, colour the precision. Cases no competitor computes are not plotted;
# their count is printed with the summary.
#
# Writes docs/src/generated/benchmarks/speedup.png and speedup.toml (the per-panel summary that
# the benchmarks page quotes).
# Usage: julia --project=docs bench/fig_speedup.jl   (or included from docs/make.jl)

using CairoMakie, Statistics, Printf, TOML

include(joinpath(@__DIR__, "overview.jl"))

# Plot palette: the Uni Lübeck corporate colours of the knopp-toolkit figure style
# (tools/knopp-toolkit/tikz/svtikz.sty, uhl-*; copied here so the scripts stay standalone).
# Float64 in Blau Ton 3, Float32 in Orange Ton 3; ink is figfg, muted is figmuted (60 % ink on
# white), the shaded "faster" half is Blau Ton 1.
const COLOR = Dict("Float64" => colorant"#0083AD", "Float32" => colorant"#CB5119")
const KINDS = [("c2c", :circle, "c2c"), ("r2c", :rect, "r2c, c2r"), ("r2r", :diamond, "r2r (DCT, DST, DHT)")]
const MARKER = Dict(k => m for (k, m, _) in KINDS)
kindgroup(k) = k == "c2c" ? "c2c" : k in ("r2c", "c2r") ? "r2c" : startswith(k, "r2r") ? "r2r" : error("unknown kind $k")
const INK, MUTED = colorant"#222222", colorant"#7A7A7A"
const FASTER = (colorant"#C2D9E6", 0.4)
const YMIN, YMAX = 0.25, 8.0
const OUT = normpath(joinpath(BENCH, "..", "docs", "src", "generated", "benchmarks"))

# One point per case that FFTL and at least one competitor measured.
function speedups(p)
    pts = NamedTuple[]
    nocomp = 0
    one = hasproperty(p, :single) ? load(p.single) : Dict{NTuple{4,String},Dict{String,Float64}}()
    for (case, libs) in load(p.file)
        (kind, T, dims, shape) = case
        haskey(libs, "fftl") || continue
        alt = get(one, case, Dict{String,Float64}())
        ts = [min(libs[c], get(alt, c, Inf)) for c in p.comps if haskey(libs, c) && isfinite(libs[c])]
        isempty(ts) && (nocomp += 1; continue)
        n = prod(parse.(Int, split(shape, "x")))
        tf = min(libs["fftl"], get(alt, "fftl", Inf))
        push!(pts, (kind = kindgroup(kind), T = T, dims = dims, n = n, s = minimum(ts) / tf))
    end
    return pts, nocomp
end

function main()
    set_theme!(fontsize = 10)
    fig = Figure(size = (920, 640), figure_padding = (4, 10, 4, 4))
    summary = Dict{String,Any}()
    for (i, p) in enumerate(PANELS)
        pts, nocomp = speedups(p)
        s = [q.s for q in pts]
        med, share = median(s), mean(s .> 1)
        summary[p.key] = Dict("median" => round(med; digits = 2), "faster_percent" => round(Int, 100share),
                              "cases" => length(s))
        println(@sprintf("%-8s median %.2f×, faster in %d%% of %d cases, %d without competitor",
                         p.key, med, round(Int, 100share), length(s), nocomp))
        r, c = fldmod1(i, 3)
        present = Set(l for (_, libs) in load(p.file) for l in keys(libs))
        comps = join(unique(COMPNAME[c] for c in p.comps if c in present), ", ")
        sub = (occursin(",", comps) ? "best of " : "vs ") * comps *
              @sprintf("  ·  median %.2f×, faster in %d %%", med, round(Int, 100share))
        ax = Axis(fig[r, c]; title = p.title, titlesize = 10, titlefont = :bold,
                  subtitle = sub, subtitlesize = 8, subtitlecolor = MUTED,
                  xscale = log10, yscale = log2, xlabel = r == 2 ? "points per transform" : "",
                  ylabel = c == 1 ? "speed-up over fastest" : "",
                  xgridvisible = false, ygridcolor = (:black, 0.06),
                  yticks = ([0.25, 0.5, 1, 2, 4, 8], ["¼", "½", "1", "2", "4", "8"]),
                  spinewidth = 0.7, xtickwidth = 0.7, ytickwidth = 0.7)
        xlims!(ax, 8, 4e7)
        ylims!(ax, YMIN, YMAX)
        hspan!(ax, 1, YMAX; color = FASTER)
        hlines!(ax, 1; color = INK, linewidth = 0.7)
        for q in pts
            if q.s > YMAX
                scatter!(ax, q.n, YMAX * 0.93; marker = :utriangle, color = :white,
                         strokecolor = COLOR[q.T], strokewidth = 0.8, markersize = 6)
            elseif q.s < YMIN
                scatter!(ax, q.n, YMIN * 1.07; marker = :dtriangle, color = :white,
                         strokecolor = COLOR[q.T], strokewidth = 0.8, markersize = 6)
            else
                scatter!(ax, q.n, q.s; color = (COLOR[q.T], 0.8), marker = MARKER[q.kind],
                         markersize = 6, strokewidth = 0.4, strokecolor = :white)
            end
        end
    end
    Legend(fig[3, 1:3],
           [[MarkerElement(color = COLOR[t], marker = :circle, markersize = 7) for t in ("Float64", "Float32")];
            [MarkerElement(color = MUTED, marker = m, markersize = 7) for (_, m, _) in KINDS]],
           ["Float64", "Float32", last.(KINDS)...];
           orientation = :horizontal, framevisible = false, labelsize = 8.5,
           patchsize = (8, 8), colgap = 12)
    rowgap!(fig.layout, 6); colgap!(fig.layout, 10)
    mkpath(OUT)
    save(joinpath(OUT, "speedup.png"), fig)
    open(io -> TOML.print(io, summary), joinpath(OUT, "speedup.toml"), "w")
    set_theme!()   # reset: the figures of the other pages must not inherit this theme
    println("wrote ", joinpath(OUT, "speedup.png"))
end

main()
