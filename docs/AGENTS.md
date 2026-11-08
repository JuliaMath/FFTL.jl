# FFTL.jl documentation

Conventions for the manual in `docs/`. Read them together with the package's `AGENTS.md`: the licence
rules there apply to everything written here as well. Everything in this folder is written in English.

## Goal and tone

- The manual is complete, precise and well exemplified: a reference for the transforms, plans, number
  types, threads, GPU path and the C library that people can work from. Clear and concrete, not
  marketing: every claim is checked against the code or a measurement.
- DocumenterVitepress (VitePress theme on top of Documenter), no fancy landing page. Good examples over
  decoration. Light mode.
- Tone: no claim to be the best. The message is that Julia and coding agents make a library of this scope
  possible, and that FFTL is meant to be developed further by the Julia community as the one FFT the
  ecosystem shares, said as an invitation. Other packages are not named or judged. Speed is reported as
  measurement, on the benchmarks page. The MIT licence is stated, not advertised. NFFT.jl appears as one
  user of the pass interface, not as a goal. FFTW is credited prominently (Explanation → "FFTW and FFTL").
- Two separate selling points, never merged into "one library for everything": (1) all transforms, number
  types and platforms in Julia; (2) libfftl, the C library compiled from the Julia code by
  `juliac --trim`, with only the Julia runtime as dependency.
- Examples are real: getting started and tutorials are Literate scripts that run during the build,
  how-to pages use `@example` blocks. A broken example fails the build.

## Structure (Diátaxis, same split as NFFT.jl)

- **Home** (`src/index.md`): logo, what you get, quick start, status.
- **Getting Started**: "Your first transform".
- **Tutorials** (Literate, `literate/tutorials/`): spectrum of a sampled signal, compressing an image
  with the DCT, Poisson with the DST (second-order convergence shown), fast convolution. Each has
  at least one figure (CairoMakie) next to its printed results.
- **How-to Guides** (`src/howto/`): coming from FFTW.jl, plans and reuse, transforms along dimensions
  (including `plan_pass`, which NFFT.jl uses), real data, cosine/sine/Hartley transforms,
  threads, number types, precompilation, GPU, C library (libfftl), compiling ahead of time (the run-time
  plans `RPlan`, `RRFFTPlan`, `RR2RPlan`), contribute.
- **Explanation** (`src/explanation/`): the DFT and its conventions, FFTW and FFTL (history, papers, what
  FFTL takes from FFTW; written from papers and documentation only), algorithms (Stockham chains,
  codelets, resampling, multidimensional without transposition, four-step, real and real-to-real
  reductions), implementation (type specialisation, SIMD, kernel cache, cache sizes, the stage loop of the
  chain, why libfftl needs `RChain`), design decisions (what was decided and why, with the evidence), package
  size and load time, performance (measurement protocol).
- **Reference** (`src/reference/`): Julia API (`@docs`), AbstractFFTs (what is implemented), libfftl C API
  (prose on FFTW3's interface as libfftl implements it; the header pages are generated from
  `libfftl/include/fftw3.h` and `fftl.h`), platforms and limits, benchmarks (the speed-up and dominance
  figures, generated from the committed overview data in `bench/results`; the campaign on quiet machines
  replaces the data).

## Decisions for the manual

The design decisions of the package (interface, the chain, rules for lengths, package size) are on the
Design decisions page. These concern the manual itself:

- **Interface in the text:** the manual writes the exported names without prefix; next to `using FFTL` it
  uses `import AbstractFFTs`, because `using AbstractFFTs` would make `fft` ambiguous.
- **Feature/platform table:** built from the README, the code and the libfftl header. The grid-length
  rule of resampled lengths is taken from the code (aarch64: smallest 5-smooth m ≥ 2n; x86: cost model,
  7-smooth m in [2n, 2.8·n]); the FFTW-padded real layout of in-place r2c/c2r in libfftl comes
  from the header.
- **Figures:** CairoMakie in the tutorials. Literate runs with `execute = false` (its default for
  markdown), so the scripts execute as Documenter `@example` blocks during `makedocs`; a figure cell ends
  with `fig` (no semicolon), because Documenter captures the block result — explicit `display(fig)` calls
  would be discarded. Documenter renders SVG first (PNG is the LaTeX fallback); Makie's default theme
  background is white.
- **Benchmarks:** the page carries the two figures of the paper (the standing of every library against
  its best counterpart, and the speed-up over the fastest other library — standing first), regenerated
  from the committed overview data in `bench/results` (about 0.4 MB) on every build: `make.jl` runs
  `bench/fig_standing.jl` and `bench/fig_speedup.jl` as their own processes, because the scripts are
  standalone with their own constants. The scripts also write `speedup.toml` and `standing.toml` next to
  the figures; the numbers on the page are read from them in `@eval` blocks, and the provenance table from
  the `.meta` files, so no number on the page is typed by hand. "Dominance" is called "standing" here, the
  summary statistic is the "fold factor". The numbers are labelled as the development state (loaded
  machines, one round per case); the campaign on quiet machines replaces the data, and the figures follow
  it. A `.meta` file names the device class, not the host.
- **libfftl:** the interface is FFTW3's (C, legacy Fortran, Fortran 2003), credited to the FFTW manual; no
  comparison with FFTW's licence, no "drop-in replacement" slogan. The pages cover size (32 MB on macOS
  arm64: library 23 MB, Julia runtime 10 MB), build requirements (Julia 1.12+ with juliac), build variables
  (`FFTL_THREADS`, `FFTL_TRIM`, `JULIA_CPU_TARGET`), the FFTW library names as links,
  `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` for existing binaries, what differs from FFTW (the list at the top
  of `fftw3.h`) and what is missing, error handling via `fftl_last_error`, concurrency rules. The C and
  Fortran examples in the how-to were compiled and run against the build.
- **Logo:** three hand-built SVGs in `docs/assets/`, each by size of the place it sits in: `fftl-main.svg`
  (wordmark with spectrum underline, everyday use), `fftl-icon.svg` (bars only, 16-48 px), `fftl-logo.svg`
  (wide: butterfly, wordmark and spectrum, for headers with room; the README links it).
  `fftl-main-dark.svg` and `fftl-logo-dark.svg` are the dark-ground variants. Copies in `docs/src/assets/`:
  `logo.svg` and `logo-dark.svg` (main; DocumenterVitepress moves every `*logo*` file to `public/` and uses
  `logo.svg` in the navbar; `logo-dark.svg` is unused until a custom `.vitepress/config.mts` sets a
  light/dark pair), `fftl-wide.svg` (home page header), `favicon.ico` (from `fftl-icon.svg`: `rsvg-convert`
  to 16, 32 and 48 px PNGs, then `magick` to combine them; `magick` alone drops the stems). No other raster
  images. If the SVGs change, copy them over.

## Verification rules

- Every numerical or behavioural claim is checked by running Julia (scratch scripts outside the repo), or
  it is marked as not verified in the text. Examples run in the build, but look at their printed output,
  not only at the exit code: several output lines were wrong before being checked (for example `brfft`
  normalisation is the product of the transformed lengths, not `d`).
- Claims about GPUs cannot be run in the build; they come from the package README and are labelled as such.
- Nothing private goes into the documentation, the data files or the pull requests: no host names or
  accounts of machines, no local paths, no private contact data, no names of people beyond the authors
  listed in the package. A machine is named by its processor ("AMD EPYC 7702", "Apple M1 Max").

## Build and git

- Build: `julia --project=docs docs/make.jl` (after `Pkg.instantiate` in `docs/`; the docs env carries
  FFTL as a path dep plus Documenter, Literate and CairoMakie). Output in `docs/build` (ignored).
  `docs/.gitignore` covers build output, generated pages, Manifest and logs.
- Work on a branch off `main` and open a pull request; do not push to `main`.

## Open items

- Accuracy: the manual claims accuracy checks but shows no numbers; a page with the error against
  BigFloat is missing.
- Dark mode: the VitePress theme offers a dark-mode switch; the manual is meant for light mode only. The
  tutorial figures have a white background, so they read in both themes.
- Benchmarks: the campaign results (quiet machines, several rounds, held-out sizes) replace the interim
  data in `bench/results`; the figures and the numbers on the page follow it.
