# Speed-up of FFTL over the fastest other library on each platform, one panel per platform.
#
# Same competitors and fair thread rule as the FFTL paper's Fig. 1. Speed-up = time of the fastest
# competitor for that case / time of FFTL; > 1 means FFTL is faster. Competitors per platform: CPU:
# FFTW (MEASURE), ducc0 and, where measured, Apple vDSP; GPU: MPSGraph (Metal) or cuFFT, and VkFFT.
# Each library is timed at its better thread count per case (the minimum over the measured thread
# counts), so FFTW is not charged for spreading tiny transforms over many threads.
#
# What the figure shows, from the panel tables (results/panel-<key>.csv, see overview.jl):
# - Smooth or not: the length is smooth when every dimension has prime factors of at most 31 (the
#   codelet limit). Smooth points are drawn in full colour, non-smooth points faded.
# - Held-out: a point with a black outline is a size drawn at random with a fixed seed and never used
#   in development. The others are overview sizes, on which FFTL's thresholds were measured.
# - Significance: a point with an interval is filled when the interval lies entirely above or below
#   1 (faster or slower), hollow when it contains 1 (a tie). Points without an interval (single round)
#   are filled.
# - The grey band is the A/A band: the 2.5 % to 97.5 % quantiles of FFTL against an identical FFTL
#   run over the cases of the panel. A ratio inside it is noise.
# - Background: the size classes short, medium, long (the same limits as aggregate.jl).
#
# Writes docs/src/generated/benchmarks/speedup.png and speedup.toml (the per-panel summary that
# the benchmarks page quotes; its keys are unchanged, the headline set is the held-out sizes when
# the panel has them).
# Usage: julia --project=docs bench/fig_speedup.jl   (or included from docs/make.jl)

using CairoMakie, Statistics, Printf, TOML

include(joinpath(@__DIR__, "overview.jl"))

# Plot palette: the Uni Lübeck corporate colours of the knopp-toolkit figure style
# (tools/knopp-toolkit/tikz/svtikz.sty, uhl-*; copied here so the scripts stay standalone).
# Float64 in Blau Ton 3, Float32 in Orange Ton 3; ink is figfg, muted is figmuted (60 % ink on
# white), the shaded "faster" half is Blau Ton 1.
const COLOR = Dict("Float64" => colorant"#0083AD", "Float32" => colorant"#CB5119")
const KINDS = [("c2c", :circle, "c2c"), ("r2c", :rect, "r2c, c2r"), ("r2r", :diamond, "r2r (DCT, DST)")]
const MARKER = Dict(k => m for (k, m, _) in KINDS)
kindgroup(k) = k == "c2c" ? "c2c" : k in ("r2c", "c2r") ? "r2c" : startswith(k, "r2r") ? "r2r" : error("unknown kind $k")
const INK, MUTED = colorant"#222222", colorant"#7A7A7A"
const FASTER = (colorant"#C2D9E6", 0.4)
const AABAND = (colorant"#7A7A7A", 0.22)
const MEDIUM_BAND = (colorant"#000000", 0.035)
const YMIN, YMAX = 0.25, 8.0
const CLASS_X = [("short", 8, 2^12), ("medium", 2^12, 2^18), ("long", 2^18, 4e7)]
const OUT = normpath(joinpath(BENCH, "..", "docs", "src", "generated", "benchmarks"))

# Points of a panel: one per valid FFTL row with a competitor. `nocomp` counts FFTL cases that no
# competitor computes.
function speedups(p)
    rows = load_panel(p)
    pts = NamedTuple[]
    nocomp = 0
    for r in rows
        (r.lib == "fftl" && r.valid) || continue
        isfinite(r.ratio) || (nocomp += 1; continue)
        push!(pts, (kind = kindgroup(r.kind), T = r.T, n = r.n, s = r.ratio, lo = r.ratio_lo, hi = r.ratio_hi,
                    smooth = r.smooth, held = r.held))
    end
    pts, nocomp
end

# A/A band of a panel from the FFTL rows; nothing for legacy panels without the A/A column.
function aa_band(p)
    aa = [r.aa_ratio for r in load_panel(p) if r.lib == "fftl" && r.valid && isfinite(r.aa_ratio)]
    isempty(aa) ? nothing : (quantile(aa, 0.025), quantile(aa, 0.975))
end

verdict(q) = !isfinite(q.lo) || !isfinite(q.hi) ? :none : q.lo > 1 ? :faster : q.hi < 1 ? :slower : :tie

function main()
    set_theme!(fontsize = 10)
    fig = Figure(size = (920, 660), figure_padding = (4, 10, 4, 4))
    summary = Dict{String,Any}()
    for (i, p) in enumerate(PANELS)
        pts, nocomp = speedups(p)
        s = [q.s for q in pts]
        med, share = median(s), mean(s .> 1)
        entry = Dict("median" => round(med; digits = 2), "faster_percent" => round(Int, 100share),
                     "cases" => length(s))
        for (name, sel) in (("smooth", q -> q.smooth), ("nonsmooth", q -> !q.smooth), ("heldout", q -> q.held))
            v = [q.s for q in pts if sel(q)]
            isempty(v) || (entry["$(name)_median"] = round(median(v); digits = 2); entry["$(name)_cases"] = length(v))
        end
        summary[p.key] = entry
        println(@sprintf("%-8s median %.2f×, faster in %d%% of %d cases, %d without competitor",
                         p.key, med, round(Int, 100share), length(s), nocomp))

        r, c = fldmod1(i, 3)
        present = Set(r.lib for r in load_panel(p))
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
        for (cls, x0, x1) in CLASS_X
            cls == "medium" && vspan!(ax, x0, x1; color = MEDIUM_BAND)
        end
        hspan!(ax, 1, YMAX; color = FASTER)
        band = aa_band(p)
        band === nothing || hspan!(ax, band[1], band[2]; color = AABAND)
        hlines!(ax, 1; color = INK, linewidth = 0.7)
        for (cls, x0, x1) in CLASS_X
            text!(ax, sqrt(max(x0, 8) * x1), YMIN * 1.25; text = cls, fontsize = 7, color = MUTED, align = (:center, :bottom))
        end
        for q in pts
            alpha = q.smooth ? 1.0 : 0.35
            v = verdict(q)
            filled = v != :tie
            if q.s > YMAX
                scatter!(ax, q.n, YMAX * 0.93; marker = :utriangle, color = :white,
                         strokecolor = (COLOR[q.T], alpha), strokewidth = 0.8, markersize = 6)
            elseif q.s < YMIN
                scatter!(ax, q.n, YMIN * 1.07; marker = :dtriangle, color = :white,
                         strokecolor = (COLOR[q.T], alpha), strokewidth = 0.8, markersize = 6)
            else
                if isfinite(q.lo) && isfinite(q.hi)
                    rangebars!(ax, [q.n], [q.lo], [q.hi]; color = (COLOR[q.T], 0.5 * alpha), linewidth = 0.8)
                end
                fill = filled ? (COLOR[q.T], 0.8 * alpha) : :white
                edge = q.held ? :black : (filled ? :white : (COLOR[q.T], alpha))
                scatter!(ax, q.n, q.s; color = fill, marker = MARKER[q.kind], markersize = 6,
                         strokewidth = q.held ? 1.0 : (filled ? 0.4 : 0.9), strokecolor = edge)
            end
        end
    end
    Legend(fig[3, 1:3],
           [[MarkerElement(color = COLOR[t], marker = :circle, markersize = 7) for t in ("Float64", "Float32")];
            [MarkerElement(color = MUTED, marker = m, markersize = 7) for (_, m, _) in KINDS];
            [MarkerElement(color = MUTED, marker = :circle, markersize = 7)];
            [MarkerElement(color = (MUTED, 0.35), marker = :circle, markersize = 7)];
            [MarkerElement(color = :white, marker = :circle, markersize = 7, strokecolor = MUTED, strokewidth = 0.9)];
            [MarkerElement(color = :white, marker = :circle, markersize = 7, strokecolor = :black, strokewidth = 1.0)];
            [PolyElement(color = AABAND)]],
           ["Float64", "Float32", last.(KINDS)..., "smooth length", "non-smooth length (faded)",
            "tie (interval contains 1)", "held-out size (outline)", "A/A band"];
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
