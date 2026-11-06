# Benchmarks

!!! warning "Interim numbers"
    The figures and numbers on this page are the development state: one round per case, measured
    on the machines listed at the bottom under their normal load, at the commits given below.
    The measurement campaign on quiet machines will replace them; the protocol below is already
    the one the campaign follows.

The numbers in the text are read from the summaries that the figure scripts write
(`bench/fig_speedup.jl`, `bench/fig_standing.jl`), so text and figures come from the same data.

## Which library to depend on

![The standing of every library against its best counterpart on each platform](../generated/benchmarks/standing.png)

This figure asks the question a package author faces: which single library to depend on. Per
case, the fold change is the time of the fastest *other* library divided by this library's
time; right of the parity line is a win, left of it a loss, and wins are not clipped at 1. The
box is the quartile rectangle with the median line and Tukey whiskers, the points are the raw
cases. Colour marks the library. The label under each column carries the fold factor (the
geometric mean of the fold changes over the cases the library itself computes), the win rate,
and the coverage: libraries that lack transform kinds (ducc0 without DCT/DST, vDSP with c2c and
DCT-II only, MPSGraph and cuFFT without real-to-real at all) compute their factor over a smaller
subset, so the coverage is part of the label.

```@eval
using TOML, Markdown, Printf, FFTL
const BDIR = joinpath(pkgdir(FFTL), "docs", "src", "generated", "benchmarks")
st = TOML.parsefile(joinpath(BDIR, "standing.toml"))
f = [st[k]["FFTL"]["fold"] for k in keys(st)]
gpu = st["a6000"]
best = argmax(l -> gpu[l]["fold"], collect(keys(gpu)))
Markdown.parse(@sprintf("""
FFTL is the only library that runs in all six panels and computes every kind. Its fold factor ranges
from %.2f× to %.2f× over the panels. It is below 1 on the NVIDIA A6000 (%.2f×), where %s
has the highest factor (%.2f×); see [Platforms and limits](platforms.md).
""", minimum(f), maximum(f), gpu["FFTL"]["fold"], best, gpu[best]["fold"]))
```

## How FFTL compares

![Speed-up of FFTL over the fastest other library on each platform](../generated/benchmarks/speedup.png)

The speed-up is the time of the fastest other library for that case divided by FFTL's time;
above 1 (shaded), FFTL is faster. One marker per transform; colour is the precision, shape the
kind; open triangles at the panel edge mark values beyond the axis. In the two multi-thread
panels every library is timed at its better setting per case, the minimum over one and all
threads, so FFTW is not charged for spreading tiny transforms over many threads. Transforms that
no competitor computes (DCT and DST in cuFFT and MPSGraph, most real-to-real kinds) are not
plotted. The table gives the share of cases in which FFTL is faster than the fastest other
library, and the median speed-up; on the A6000, small transforms are bound by kernel launches.

```@eval
using TOML, Markdown, Printf, FFTL
const BDIR = joinpath(pkgdir(FFTL), "docs", "src", "generated", "benchmarks")
s = TOML.parsefile(joinpath(BDIR, "speedup.toml"))
q(k) = @sprintf("| %d %% | %.2f× | %d |", s[k]["faster_percent"], s[k]["median"], s[k]["cases"])
Markdown.parse("""
| platform | competitors | FFTL faster in | median speed-up | cases |
|:--|:--|--:|--:|--:|
| Apple M1 Max, 1 thread | FFTW, ducc0, vDSP $(q("m1t1"))
| AMD EPYC 7702, 1 thread | FFTW, ducc0 $(q("epyct1"))
| Apple M1 Max, 8 threads | FFTW, ducc0 $(q("m1t8"))
| AMD EPYC 7702, 16 threads | FFTW, ducc0 $(q("epyct16"))
| Apple M1 Max GPU | MPSGraph, VkFFT $(q("m1gpu"))
| NVIDIA RTX A6000 | cuFFT, VkFFT $(q("a6000"))
""")
```

The data is the overview set in `bench/results`. Each data file has a `.meta` file with its
provenance; the table lists the FFTL commit and the start of each run.

```@eval
using Markdown, FFTL
dir = joinpath(pkgdir(FFTL), "bench", "results")
rows = String[]
for f in sort(filter(endswith(".meta"), readdir(dir)))
  m = read(joinpath(dir, f), String)
  commit = match(r"FFTL (\w+)", m)
  start = match(r"start: (\S+?)T", m)
  push!(rows, "| `$(replace(f, ".meta" => ".csv"))` | `$(commit === nothing ? "?" : commit[1])` | $(start === nothing ? "?" : start[1]) |")
end
Markdown.parse(join(["| data | FFTL commit | measured |", "|:--|:--|:--|", rows...], "\n"))
```

## What the campaign covers

- **Complex, real and real-to-real transforms** on the CPU, 1D to 3D, Float32 and Float64, against
  the competitors that support the same kind. Each kind has its own column; kinds without a
  competitor are shown separately.
- **Threads.** The same sizes with one thread and with the number of threads of the machine,
  with the noise band of each machine.
- **GPU.** Complex, real and real-to-real transforms against the vendor library (cuFFT on CUDA,
  MPSGraph and Metal.jl's FFT on Metal) and against VkFFT.
- **libfftl.** The C library against the Julia package.
- **Planning.** The cost of the first plan in a fresh process, separate from the execution time.
- **Accuracy.** Error against a high-precision reference over the same sizes.

## How it is measured

The protocol is described on the [Performance](@ref "Performance") page: A/B runs on the same
machine, medians of interleaved runs, A/A runs for the noise band, and sizes reported separately
from the sizes used for tuning.

## Machines

| machine | role |
|:--|:--|
| Apple M1 Max | development machine, CPU and Metal GPU |
| AMD EPYC 7702 (Zen 2, 2 sockets, AVX2) | x86-64 CPU of the figures above |
| AMD EPYC 7763 (Zen 3, 2 sockets, AVX2) | x86-64 development machine; hosts the A6000 |
| NVIDIA RTX A6000 | CUDA GPU |

Each result names the library, its version, the build flags and the thread count. Competitors
are built from their current releases with native CPU flags; FFTW is planned with `MEASURE`.

## Status

| part | status |
|:--|:--|
| harness, protocol and figures | prepared |
| campaign (quiet machines, several rounds, held-out sizes) | planned |
