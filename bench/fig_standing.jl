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
# One panel per platform, one column per library: raw strip (one point per case), box summary
# (quartile rectangle, median line, Tukey whiskers), and the parity line at 1. Tick labels carry
# the fold factor, win rate and coverage (cases computed / panel cases): the factor is
# computed over the cases the library itself computes, and libraries with partial coverage
# (ducc0: no DCT kinds, vDSP: c2c and DCT-II only) would otherwise read as comparable over the
# full panel. All panels share one y range in whole octaves, exact powers of two as ticks.
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
function folds(p)
    libs = ["fftl"; p.comps]
    one = hasproperty(p, :single) ? load(p.single) : Dict{NTuple{4,String},Dict{String,Float64}}()
    out = Dict(l => Float64[] for l in libs)
    cov = Dict(l => 0 for l in libs)
    total = 0
    for (case, t) in load(p.file)
        haskey(t, "fftl") || continue
        total += 1
        alt = get(one, case, Dict{String,Float64}())
        best = Dict(l => min(get(t, l, Inf), get(alt, l, Inf)) for l in libs)
        for l in libs
            isfinite(best[l]) && (cov[l] += 1)
            fastest_other = minimum(best[o] for o in libs if o != l)
            (isfinite(fastest_other) && isfinite(best[l])) || continue
            push!(out[l], fastest_other / best[l])
        end
    end
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
                      "metal" => colorant"#248F85", "cufft" => colorant"#0083AD",
                      "vkfft" => colorant"#595959")
const LIBNAME = merge(COMPNAME, Dict("fftl" => "FFTL"))
const OUT = normpath(joinpath(BENCH, "..", "docs", "src", "generated", "benchmarks"))

function main()
    set_theme!(fontsize = 10)
    fig = Figure(size = (920, 640), figure_padding = (4, 10, 4, 4))
    # One y range for every panel so the panels are directly comparable: the enclosing whole
    # octaves of the data, with exact powers of two as ticks.
    allpanels = [folds(p) for p in PANELS]
    summary = Dict{String,Any}()
    lo = minimum(minimum(f[l]) for (f, _, _) in allpanels for l in keys(f) if !isempty(f[l]))
    hi = maximum(maximum(f[l]) for (f, _, _) in allpanels for l in keys(f) if !isempty(f[l]))
    k0, k1 = floor(Int, log2(lo)), ceil(Int, log2(hi))
    yvals = [2.0^k for k in k0:k1]
    ylabels = [k == 0 ? "1" : k > 0 ? string(Int(2^k)) : "1/" * string(Int(2^-k)) for k in k0:k1]
    for (i, p) in enumerate(PANELS)
        f, cov, total = allpanels[i]
        libs = ["fftl"; p.comps]
        present = [l for l in libs if !isempty(f[l])]
        summary[p.key] = Dict(LIBNAME[l] => Dict("fold" => round(exp(mean(log.(f[l]))); digits = 2),
                                                  "win_percent" => round(Int, 100count(>(1), f[l]) / length(f[l])),
                                                  "coverage" => cov[l], "total" => total)
                              for l in present)
        println(@sprintf("%-8s %s", p.key,
                         join([@sprintf("%s ×%.2f (%d%% of %d/%d)", LIBNAME[l],
                                        exp(mean(log.(f[l]))),
                                        round(Int, 100count(>(1), f[l]) / length(f[l])),
                                        cov[l], total) for l in present], ", ")))
        r, c = fldmod1(i, 3)
        ax = Axis(fig[r, c]; title = p.title, titlesize = 10, titlefont = :bold,
                  xticks = (1:length(present),
                            [@sprintf("%s\n×%.2f·%d\n%d/%d", LIBNAME[l], exp(mean(log.(f[l]))),
                                      round(Int, 100count(>(1), f[l]) / length(f[l])), cov[l], total)
                             for l in present]),
                  xticklabelsize = 7.5,
                  ylabel = r == 2 ? "" : "fold change vs the fastest\nother library",
                  yscale = log2, yticks = (yvals, ylabels),
                  xgridvisible = false, ygridcolor = (:black, 0.06),
                  spinewidth = 0.7, xtickwidth = 0.7, ytickwidth = 0.7)
        hlines!(ax, [1]; color = colorant"#222222", linewidth = 0.7)   # 1 = parity
        for (j, l) in enumerate(present)
            col = LIBCOLOR[l]
            jit = [0.13 * sin(3.7 * k) for k in 1:length(f[l])]
            scatter!(ax, j .+ 0.12 .+ jit, f[l]; color = (col, 0.6), markersize = 4,
                     strokewidth = 0)
            box!(ax, j - 0.25, f[l]; width = 0.15, col = (col, 0.75))
        end
        xlims!(ax, 0.6, length(present) + 0.35)
        ylims!(ax, 2.0^k0, 2.0^k1)
    end
    Legend(fig[3, 1:3],
           [LineElement(color = colorant"#222222", linewidth = 0.7)],
           ["1 = parity with the fastest other library · label lines: ×factor · win rate in % · coverage"];
           orientation = :horizontal, framevisible = false, labelsize = 8.5)
    rowgap!(fig.layout, 6); colgap!(fig.layout, 10)
    mkpath(OUT)
    save(joinpath(OUT, "standing.png"), fig)
    open(io -> TOML.print(io, summary), joinpath(OUT, "standing.toml"), "w")
    set_theme!()   # reset: the figures of the other pages must not inherit this theme
    println("wrote ", joinpath(OUT, "standing.png"))
end

main()
