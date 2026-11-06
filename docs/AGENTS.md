# FFTL.jl documentation

Documentation of FFTL.jl (`FFTL.jl/docs` on branch `main`; the former worktree
`FastJuliaFFT/FFTL-docs` was merged into `main` as ffeac17, feature work continues on branches
off `main`). Read this together with
the package's `AGENTS.md`: the licence rules there apply to everything written here as well.

Conversation with Tobias is in German; everything in this folder is written in English.

## Goal and tone

- The manual must be complete, precise and well exemplified. Tobias asked for an "end boss" manual
  that makes up for the weak docs of AbstractFFTs.jl and FFTW.jl. Strong, clear and concrete, not
  marketing: every claim is checked against the code or a measurement.
- DocumenterVitepress (VitePress theme on top of Documenter), no fancy landing page. Good examples over decoration. Light mode.
- Tone (Tobias, 2026-11-06): no claim to be the best. The message is that Julia and coding agents
  make a library of this scope possible, and that FFTL is meant to be developed further by the
  Julia community as the one FFT the ecosystem shares (said as an invitation; other packages are
  not named or judged). Speed is reported as measurement, on the benchmarks page. The MIT licence
  is stated, not advertised. NFFT.jl appears as one user of the pass interface, not as a goal.
  FFTW is credited prominently (Explanation → "FFTW and FFTL").
- Two separate selling points, never merged into "one library for everything": (1) all
  transforms, number types and platforms in Julia; (2) libfftl, the C library compiled from the
  Julia code by `juliac --trim`, with only the Julia runtime as dependency.
- Examples are real: getting started and tutorials are Literate scripts that run during the build,
  how-to pages use `@example` blocks. A broken example fails the build.

## Structure (Diátaxis, same split as NFFT.jl)

- **Home** (`src/index.md`): logo, what you get, quick start, status.
- **Getting Started**: "Your first transform".
- **Tutorials** (Literate, `literate/tutorials/`): spectrum of a sampled signal, compressing an image
  with the DCT, Poisson with the DST (second-order convergence shown), fast convolution. Each has
  at least one figure (CairoMakie) next to its printed results.
- **How-to Guides** (`src/howto/`): coming from FFTW.jl, plans and reuse, transforms along dimensions (including
  `plan_pass` with pruning, which NFFT.jl uses), real data, cosine/sine/Hartley transforms, threads,
  number types, precompilation, GPU, C library (libfftl), compiling ahead of time (the run-time
  plans `RPlan`, `RRFFTPlan`, `RR2RPlan`), contribute.
- **Explanation** (`src/explanation/`): the DFT and its conventions, FFTW and FFTL (history,
  papers, what FFTL takes from FFTW; written from papers and documentation only), algorithms (Stockham chains,
  codelets, pruning, Bluestein, multidimensional without transposition, four-step, real and
  real-to-real reductions), implementation (type specialisation, SIMD, kernel cache, cache sizes,
  the two chain forms, why libfftl needs `RChain`), performance (measurement protocol).
- **Reference** (`src/reference/`): Julia API (`@docs`), AbstractFFTs (what is implemented),
  libfftl C API (prose; the header page is generated from `libfftl/include/fftl.h`), platforms and
  limits, benchmarks (the speed-up and dominance figures, generated from the committed overview
  data in `bench/results`; the campaign on quiet machines replaces the data).

## Decisions

- **Interface:** FFTL exports FFTW.jl's exports (DFT and DCT functions with `plan_` and in-place
  forms, AbstractFFTs's helpers), as its own functions; `r2r`, kinds, flags, `set_num_threads`
  are prefixed. Generic AbstractFFTs code reaches FFTL through less specific methods
  (`src/abstractffts.jl`); with FFTW.jl loaded, FFTW takes its strided Float32/Float64 arrays
  there. Decision 2026-11-06 in the workspace `notes/decisions.md`. The manual writes the
  exported names without prefix; next to `using FFTL` it uses `import AbstractFFTs`, because
  `using AbstractFFTs` would make `fft` ambiguous.
- **Feature/platform table:** built from README, code and the libfftl header. Checked against the
  FFTL paper (`~/Arbeit/Eigene Publikationen/Draft/2026 FFTL.jl/manuscript/`, 2026-11-05): the
  table itself matched; fixed from the code was the Bluestein convolution-length rule
  (aarch64: smallest 5-smooth M ≥ 2n−1; x86: cost model, 7-smooth M in [2n−1, 1.4·(2n−1)]) and
  added from the header was the FFTW-padded real layout of in-place r2c/c2r in libfftl.
- **Figures:** CairoMakie in the tutorials. Literate runs with `execute = false` (its default for
  markdown), so the scripts execute as Documenter `@example` blocks during `makedocs`; a figure
  cell ends with `fig` (no semicolon), because Documenter captures the block result — explicit
  `display(fig)` calls would be discarded. Documenter renders SVG first (PNG is the LaTeX
  fallback); Makie's default theme background is white.
- **Benchmarks:** Tobias asked for the benchmark page (2026-11-06). It carries the two figures
  of the paper (the standing of every library against its best counterpart, and the speed-up
  over the fastest other library — the paper's Fig. 5 and Fig. 1, standing first), regenerated
  from the committed overview data in `bench/results` (copied from the workspace harness; ~0.4
  MB) on every build — `make.jl` runs `bench/fig_standing.jl` and `bench/fig_speedup.jl` as
  their own processes, because the scripts are standalone with their own constants. The scripts
  also write `speedup.toml` and `standing.toml` next to the figures; the numbers on the page are
  read from them in `@eval` blocks, and the provenance table from the `.meta` files, so no number
  on the page is typed by hand. The scripts
  are adapted from the paper's `scripts/fig5-dominance.jl` and `fig1.jl` (docs sizing, PNG
  output; "dominance" is called "standing" here, the summary statistic is the "fold factor").
  The numbers are labelled as the development state (loaded machines, one round per case); the
  campaign on quiet machines replaces the data, and the figures follow it.
- **libfftl:** size (23 MB on macOS arm64: library 14 MB, Julia runtime 10 MB), build requirements (Julia 1.12+
  with juliac), build variables (`FFTL_THREADS`, `FFTL_TRIM`, `JULIA_CPU_TARGET`), baked thread pool,
  column-major data, unnormalised backward transform, error handling via `fftl_last_error`, concurrency
  rules. The C example in the how-to was compiled and run against the build in `FFTL.jl/libfftl`.
- **Logo:** three hand-built SVGs in `docs/assets/`, each by size of the place it sits in:
  `fftl-main.svg` (wordmark with spectrum underline, everyday use), `fftl-icon.svg` (bars only,
  16-48 px), `fftl-logo.svg` (wide: butterfly, wordmark and spectrum, for headers with room; the
  README links it). `fftl-main-dark.svg` and `fftl-logo-dark.svg` are the dark-ground variants.
  Copies in `docs/src/assets/`: `logo.svg` and `logo-dark.svg` (main; DocumenterVitepress moves
  every `*logo*` file to `public/` and uses `logo.svg` in the navbar; `logo-dark.svg` is unused
  until a custom `.vitepress/config.mts` sets a light/dark pair), `fftl-wide.svg` (home page header), `favicon.ico`
  (from `fftl-icon.svg`: `rsvg-convert` to 16, 32 and 48 px PNGs, then `magick` to combine them;
  `magick` alone drops the stems). No other raster images. If the SVGs change, copy them over.

## Verification rules

- Every numerical or behavioural claim is checked by running Julia (scratch scripts outside the repo),
  or it is marked as not verified in the text. Examples run in the build, but look at their printed
  output, not only at the exit code: several output lines were wrong before being checked
  (for example `brfft` normalisation is the product of the transformed lengths, not `d`).
- Claims about GPUs cannot be run in the build; they come from the package README and notes and are
  labelled as such.
- Docstrings were added in `src/` (`fft`, `bfft`, `ifft`, `rfft`, `brfft`, `irfft`, `R2RKind`). The
  docs branch does not change any behaviour in `src/`.

## Build and git

- Build: `julia --project=docs docs/make.jl` (after `Pkg.instantiate` in `docs/`; the docs env
  carries FFTL as a path dep plus Documenter, Literate and CairoMakie). Output in
  `docs/build` (ignored). `docs/.gitignore` covers build output, generated pages, Manifest and logs.
- Commit on a branch off `main`. Do not push and do not merge into `main` without Tobias's OK.
- Commit trailer: `Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>`.

## Open items

- Accuracy: the manual claims accuracy checks but shows no numbers; a page from the workspace's
  `quality.csv` (error against BigFloat) is missing.
- Dark mode: the VitePress theme offers a dark-mode switch; Tobias wants light mode only. The
  tutorial figures have a white background, so they read in both themes.
- Benchmarks: the campaign results (quiet machines, several rounds, held-out sizes) replace the
  interim data in `bench/results`; the figures and the numbers on the page follow it.
