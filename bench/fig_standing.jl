# Standing: every library against its best counterpart, per platform.
#
# Same data pipeline, competitors and fair thread rule as the FFTL paper's Fig. 5 (its
# scripts/fig5-dominance.jl; docs sizing and PNG output instead of the Nature figure). The
# measured quantity is the fold change: per case, the time of the fastest OTHER library divided
# by this library's time. > 1 means the library is faster than its best counterpart (a win,
# visible as the right tail), < 1 means slower (the left tail), 1 = parity. Unlike a "distance
# to the fastest" plot this does not clip the wins at the parity line — the speed-up factor
# itself is the plotted quantity. The fold factor (the geometric mean of the fold changes) is
# the library's standing against its best counterpart.
#
# One panel per platform, one column per library, three box-and-dot groups per column: all cases, smooth
# lengths, non-smooth lengths (every box: quartile rectangle, median line, Tukey whiskers), and the
# parity line at 1. Tick labels carry the fold factor over all cases, win rate and coverage (cases
# computed / panel cases): the factor is computed over the cases the library itself computes, and
# libraries with partial coverage (ducc0: no DCT kinds, vDSP: c2c and DCT-II only) would otherwise
# read as comparable over the full panel. All panels share one y range in whole octaves, exact
# powers of two as ticks.
#
# Writes docs/src/generated/benchmarks/standing.png and standing.toml (the per-panel summary that
# the benchmarks page quotes).
# Usage: julia --project=docs bench/fig_standing.jl   (or included from docs/make.jl)

using CairoMakie, Statistics, Printf, TOML

include(joinpath(@__DIR__, "overview.jl"))

# fold change per library over the cases it runs: time(fastest other) / time(this).
# Also returns the coverage per library (cases with a measured time among the panel's FFTL
# cases) and the number of those cases: libraries with partial coverage compute their factor
# over a different case subset, so the label states the coverage next to the factor.
# Fold changes of a panel: `out[(lib, group)]` lists the valid cases of a library for the groups
# "all", "smooth" and "nonsmooth" lengths; `cov` the cases each library has in the panel; `total`
# the FFTL cases.
function folds(p)
    rows = filter(r -> r.valid, load_panel(p))
    libs = ["fftl"; p.comps]
    group(r) = r.smooth ? "smooth" : "nonsmooth"
    out = Dict{Tuple{String,String},Vector{Float64}}()
    for l in libs
        for g in ("smooth", "nonsmooth")
            out[(l, g)] = Float64[r.ratio for r in rows if r.lib == l && group(r) == g && isfinite(r.ratio)]
        end
        out[(l, "all")] = vcat(out[(l, "smooth")], out[(l, "nonsmooth")])
    end
    cov = Dict(l => count(r -> r.lib == l, rows) for l in libs)
    total = count(r -> r.lib == "fftl", rows)
    return out, cov, total
end

# Box plot drawn by hand: CairoMakie 0.15's Boxplot fails to resolve its arguments with the
# new ComputePipeline (only the median line renders). Rectangle, median line, Tukey whiskers.
function box!(ax, x, vals; width = 0.16, col, stroke = colorant"#222222")
    q1, med, q3 = quantile(vals, 0.25), quantile(vals, 0.5), quantile(vals, 0.75)
    iqr = q3 - q1
    lo = minimum(v for v in vals if v >= q1 - 1.5 * iqr)
    hi = maximum(v for v in vals if v <= q3 + 1.5 * iqr)
    x0 = x - width / 2; x1 = x + width / 2
    linesegments!(ax, [Point2f(x, lo) => Point2f(x, q1), Point2f(x, q3) => Point2f(x, hi)];
                  color = stroke, linewidth = 0.7)
    poly!(ax, [Point2f(x0, q1), Point2f(x1, q1), Point2f(x1, q3), Point2f(x0, q3)];
                  color = col, strokewidth = 0.5, strokecolor = stroke)
    linesegments!(ax, [Point2f(x0, med) => Point2f(x1, med)]; color = stroke, linewidth = 1.0)
end

# Plot palette: the Uni Lübeck corporate colours of the knopp-toolkit figure style
# (tools/knopp-toolkit/tikz/svtikz.sty, uhl-*; copied here so the scripts stay standalone).
# FFTL in the house colour Ozeangrün, one Uni Lübeck family per library (Ton 3), VkFFT in grey.
const LIBCOLOR = Dict("fftl" => colorant"#004B5A", "fftw_measure" => colorant"#CB5119",
                      "ducc" => colorant"#BF8614", "vdsp" => colorant"#B51621",
                      "mpsgraph" => colorant"#248F85", "metal" => colorant"#248F85", "cufft" => colorant"#0083AD",
                      "vkfft" => colorant"#595959")
const LIBNAME = merge(COMPNAME, Dict("fftl" => "FFTL"))
const OUT = normpath(joinpath(BENCH, "..", "docs", "src", "generated", "benchmarks"))

# Three boxes per library: all cases (left, the summary), smooth lengths (middle, full colour) and
# non-smooth lengths (right, faded). The strip of single cases is drawn for the two length groups.
const GROUPS = ["all", "smooth", "nonsmooth"]
const GROUP_OFFSET = Dict("all" => -0.32, "smooth" => 0.0, "nonsmooth" => 0.32)
const GROUP_ALPHA = Dict("all" => 0.55, "smooth" => 0.9, "nonsmooth" => 0.35)

# Box and rain of one group at x: the box on the left of x, the cases as jittered dots on the right.
function boxrain!(ax, x, vals; col)
    box!(ax, x - 0.05, vals; width = 0.1, col = (col, 0.9))
    jit = [0.04 * sin(3.7 * k) for k in 1:length(vals)]
    scatter!(ax, x + 0.06 .+ jit, vals; color = (col, 0.5), markersize = 3, strokewidth = 0)
end

function main()
    set_theme!(fontsize = 10)
    fig = Figure(size = (920, 680), figure_padding = (4, 10, 4, 4))
    allpanels = [folds(p) for p in PANELS]
    summary = Dict{String,Any}()
    vals = [v for (f, _, _) in allpanels for v in values(f) if !isempty(v)]
    lo, hi = minimum(minimum.(vals)), maximum(maximum.(vals))
    k0, k1 = floor(Int, log2(lo)), ceil(Int, log2(hi))
    yvals = [2.0^k for k in k0:k1]
    ylabels = [k == 0 ? "1" : k > 0 ? string(Int(2^k)) : "1/" * string(Int(2^-k)) for k in k0:k1]
    fold(v) = isempty(v) ? NaN : exp(mean(log.(v)))
    win(v) = isempty(v) ? 0 : round(Int, 100count(>(1), v) / length(v))
    for (i, p) in enumerate(PANELS)
        f, cov, total = allpanels[i]
        libs = ["fftl"; p.comps]
        present = [l for l in libs if !isempty(f[(l, "all")])]
        summary[p.key] = Dict{String,Any}()
        for l in present
            d = Dict("fold" => round(fold(f[(l, "all")]); digits = 2), "win_percent" => win(f[(l, "all")]),
                     "coverage" => cov[l], "total" => total,
                     "fold_smooth" => round(fold(f[(l, "smooth")]); digits = 2),
                     "fold_nonsmooth" => round(fold(f[(l, "nonsmooth")]); digits = 2))
            summary[p.key][LIBNAME[l]] = d
        end
        println(@sprintf("%-8s %s", p.key,
                         join([@sprintf("%s ×%.2f (smooth ×%.2f, non-smooth ×%.2f, %d%% of %d/%d)", LIBNAME[l],
                                        fold(f[(l, "all")]), fold(f[(l, "smooth")]), fold(f[(l, "nonsmooth")]),
                                        win(f[(l, "all")]), cov[l], total) for l in present], ", ")))
        r, c = fldmod1(i, 3)
        ax = Axis(fig[r, c]; title = p.title, titlesize = 10, titlefont = :bold,
                  xticks = (1:length(present),
                            [@sprintf("%s\n×%.2f·%d\n%d/%d", LIBNAME[l], fold(f[(l, "all")]), win(f[(l, "all")]),
                                      cov[l], total) for l in present]),
                  xticklabelsize = 7.5,
                  ylabel = r == 2 ? "" : "fold change vs the fastest\nother library",
                  yscale = log2, yticks = (yvals, ylabels),
                  xgridvisible = false, ygridcolor = (:black, 0.06),
                  spinewidth = 0.7, xtickwidth = 0.7, ytickwidth = 0.7)
        hlines!(ax, [1]; color = colorant"#222222", linewidth = 0.7)   # 1 = parity
        for (j, l) in enumerate(present)
            col = LIBCOLOR[l]
            for g in GROUPS
                v = f[(l, g)]
                isempty(v) && continue
                boxrain!(ax, j + GROUP_OFFSET[g], v; col = (col, GROUP_ALPHA[g]))
            end
        end
        xlims!(ax, 0.5, length(present) + 0.5)
        ylims!(ax, 2.0^k0, 2.0^k1)
    end
    Legend(fig[3, 1:3],
           [[PolyElement(color = (colorant"#595959", GROUP_ALPHA[g])) for g in GROUPS];
            [LineElement(color = colorant"#222222", linewidth = 0.7)]],
           ["all cases (left group)", "smooth lengths (middle group)", "non-smooth lengths (right group, faded)",
            "1 = parity · box: quartiles and median · dots: cases · labels: ×fold · win % · coverage"];
           orientation = :horizontal, framevisible = false, labelsize = 8.5)
    rowgap!(fig.layout, 6); colgap!(fig.layout, 10)
    mkpath(OUT)
    save(joinpath(OUT, "standing.png"), fig)
    open(io -> TOML.print(io, summary), joinpath(OUT, "standing.toml"), "w")
    set_theme!()   # reset: the figures of the other pages must not inherit this theme
    println("wrote ", joinpath(OUT, "standing.png"))
end

main()
