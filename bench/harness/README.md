# Benchmark harness

Measurement and aggregation of FFTL against FFTW, ducc0 and the vendor FFTs (vDSP, MPSGraph, cuFFT)
and VkFFT. The harness runs in three steps:

1. **Measure** (`measure.jl`): one process per thread count, writes raw rows, one per library,
   case and round, plus a `.meta` with the provenance of the run. Nothing is aggregated here.
2. **Aggregate** (`aggregate.jl`): reads the raw rows of one machine and one tier, computes per case
   and library the median over rounds, the 95 % bootstrap interval of the speed-up, the A/A band and
   the validity (relerr). Writes one table. The figures in `bench/` read only this table.
3. **Run** (`run.jl`, `monitor.jl`): one tier on one machine profile, locally or over ssh. Checks the
   load, pins the process where the profile says so, starts `measure.jl` per thread count and puts
   the output into one directory.

The library setups, the reference transforms and the shape lists come from `../overview/run_kinds.jl`,
which is included, not copied. The wrappers of the vendor libraries are in `../vkfft`, `../cufft` and
`../ducc-ffi`.

## Setup

From `bench/`, the environment is instantiated once:

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

It develops the package from the checkout (`FFTL = {path = ".."}`), so the benchmarks measure the
code you have checked out. Record the commit: the `.meta` of every run contains it.

The wrappers of the vendor libraries are built separately (only when the profile uses them):

- `ducc-ffi/build.sh <ducc0-source-dir>`: ducc0 through its public FFT header only. ducc0 is not
  part of this repository.
- `vkfft/build.sh <vkfft-dir>`: VkFFT through its public header (metal-cpp on macOS).
- `cufft/build.sh`: cuFFT (CUDA toolkit).

## Usage

From `bench/`, one tier on one machine profile:

```
julia --project=. harness/run.jl m1max smoke
julia --project=. harness/run.jl m1max dev --threads=1,8
julia --project=. harness/aggregate.jl OUT.csv harness/out/m1max/dev/<stamp>/t1.csv harness/out/m1max/dev/<stamp>/t8.csv
```

Options of `run.jl`: `--threads=1,16` overrides the profile's thread counts, `--force` starts despite
a load above the profile's `max_load`.

## Progress monitor (`monitor.jl`)

One command starts the runs of several machines and shows one progress bar per job:

```
julia --project=. harness/monitor.jl m1max --tier=dev
julia --project=. harness/monitor.jl m1max:1,8 mymachine:16 --tier=dev
```

An argument is a machine profile, optionally with thread counts. Machines run in parallel, the
thread counts of one machine one after another. A profile with `host` runs over ssh in
`remote_root`; the CSV and .meta are copied back when the job has finished. The measurement prints
`PROGRESS done total` lines, which the monitor reads. Ctrl-C stops all jobs. Each job's stderr goes
to a .log next to its CSV.

For a remote profile, copy the bench directory to `remote_root` once (and again after changes),
without the outputs:

```
rsync -a --exclude out/ --exclude Manifest.toml bench/ host:remote_root/
```

and instantiate the environment there.

## Machine profiles

A profile names the machine, its public label (the only machine name that goes into any output),
the Julia project, the default thread counts, the load limit and the pinning per thread count
(`taskset`, Linux only).

- `harness/machines/*.toml` are the public profiles (no host names, the machine label only).
- `machines.local/<id>.toml` (in `bench/`, not versioned) overrides a public profile of the same id
  and adds private ones: host, `remote_root`, Julia binary, project path, load limit. Copy a public
  profile there and add the host fields. A profile with `host` runs over ssh.

Adding a machine means adding a profile; no script changes.

| field | meaning |
|---|---|
| `id`, `label` | profile name; public machine label, written into the output |
| `project` | Julia environment (relative to `bench/`, or absolute on the machine) |
| `threads` | default thread counts |
| `max_load` | start refused when the 1-minute load is higher (`--force` overrides) |
| `pin` | `[pin]` table: thread count → CPU list for `taskset` (Linux) |
| `device` | `"metal"` or `"cuda"` for GPU profiles, `libs` lists the libraries |
| `host`, `remote_root`, `julia` | only for remote profiles |

## GPU (`gpu.jl`)

A profile with `device = "metal"` or `"cuda"` runs the catalog on the GPU, with the same raw format
(threads = 0). Julia-side libraries: `fftl` (KernelAbstractions extension) and, on Metal, `mpsgraph`
(Apple's FFT through Metal.jl, Float32 only). Separate-process libraries: `vkfft` and `cufft`
(wrappers in `../vkfft` and `../cufft`). Julia-side timing: median over seven samples of `reps`
back-to-back transforms, per transform; separate-process libraries report their own times. Not
included: vkfft_batch. DHT is not run on the GPU.

GPU profiles need the packages of the device (Metal.jl or CUDA.jl) in their environment; the
public `bench/Project.toml` does not list them.

## Raw format

`machine,tier,threads,lib,kind,T,dims,shape,n,family,round,seconds,plan_seconds,relerr`

`machine` is the public processor label ("Apple M1 Max"), not the profile id. `family` is `tuned`,
`nonsmooth` or `heldout` (see below).

`seconds` is the minimum over the samples of one round (each sample times enough calls to last at
least 50 µs). Rounds alternate the order of the libraries. `fftl_aa` is a second FFTL run, the same
code on other buffers, for the A/A band.

## Tiers (`tiers/`)

| tier | stride | rounds | budget per round and library | A/A | purpose |
|---|---|---|---|---|---|
| `smoke` | 15 | 1 | 0.05 s | no | does the pipeline run |
| `check` | 15 | 3 | 0.02 s | yes | tests the statistics, numbers not for figures |
| `dev` | 1 | 3 | 0.1 s | yes | development state of the figures |
| `campaign` | 1 | 7 | 0.2 s | yes | the measurement campaign, quiet machines only |

`stride` keeps every n-th case of the catalog, `kinds` selects the transform kinds. A tier is a
file: changing a budget or a round count does not change the code.

## Aggregated table

Columns: `machine,tier,kind,T,shape,n,class,family,lib,threads,rounds,median_seconds,ratio,ratio_lo,ratio_hi,valid,relerr,aa_ratio,aa_lo,aa_hi`

`KEEP=fftw_measure,ducc` restricts the competitors (FFTL is always in); the figure panels use this per
panel (`bench/panels.toml`).

- `ratio` for a library L: median over rounds of (fastest other library in the round) / (L in the
  round). Above 1, L is faster than the best of the others. For FFTL this is the speed-up of the
  speed-up figure, for every library it is the fold change of the standing figure.
- `threads` is the thread count with the lowest median for that library and case.
- `class` is the size class from `n` (`short` ≤ 2^12, `medium` ≤ 2^18, `long` above). The limits are
  provisional; `aggregate.jl` recomputes them from `n`, so they can change without measuring again.
- Intervals are NaN with fewer than three rounds.
- `valid` is false when relerr exceeds the tolerance (1e-10 for Float64, 1e-4 for Float32).

## Output location

`harness/out/<profile id>/<tier>/<timestamp>/` (not versioned). The profile id is the directory name,
but the host name is never written.

## Catalog (`catalog.jl`)

Three families of cases, one per kind group (c2c, r2c, c2r, DCT-II; the other r2r kinds only have
the tuned sizes):

- `tuned`: the sizes of the overview. The thresholds of FFTL were measured on them, so they are not a
  held-out test.
- `nonsmooth`: lengths with a prime factor above 31 (1-D, 2-D, 3-D), listed in `catalog.jl`.
- `heldout`: sizes drawn once with seed 20261110 (`make_heldout.jl`, committed in `heldout.txt`).
  Not used for tuning. The runs read the file and never draw again.

## Known limits

- `vDSP` takes only lengths f·2^k with f in {1, 3, 5, 15}; `overview/run_kinds.jl` skips other
  lengths (they crashed the library in Execute).
- The same commit should run on every machine of a comparison; `.meta` records which one ran.
