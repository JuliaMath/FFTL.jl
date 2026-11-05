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
- Plain Documenter theme, no fancy landing page. Good examples over decoration. Light mode.
- Examples are real: getting started and tutorials are Literate scripts that run during the build,
  how-to pages use `@example` blocks. A broken example fails the build.

## Structure (Diátaxis, same split as NFFT.jl)

- **Home** (`src/index.md`): logo, what you get, quick start, status.
- **Getting Started**: "Your first transform".
- **Tutorials** (Literate, `literate/tutorials/`): spectrum of a sampled signal, compressing an image
  with the DCT, Poisson with the DST (second-order convergence shown), fast convolution. Each has
  at least one figure (CairoMakie) next to its printed results.
- **How-to Guides** (`src/howto/`): plans and reuse, transforms along dimensions (including
  `plan_pass` with pruning, which NFFT.jl uses), real data, cosine/sine/Hartley transforms, threads,
  number types, precompilation, GPU, using FFTL next to FFTW.jl, C library (libfftl), contribute.
- **Explanation** (`src/explanation/`): the DFT and its conventions, algorithms (Stockham chains,
  codelets, pruning, Bluestein, multidimensional without transposition, four-step, real and
  real-to-real reductions), implementation (type specialisation, SIMD, kernel cache, cache sizes,
  the two chain forms, why libfftl needs `RChain`), performance (measurement protocol).
- **Reference** (`src/reference/`): Julia API (`@docs`), AbstractFFTs (what is implemented),
  libfftl C API (prose; the header page is generated from `libfftl/include/fftl.h`), platforms and
  limits, benchmarks (stub: protocol and machines, no numbers yet).

## Decisions

- **AbstractFFTs:** all plans are `AbstractFFTs.Plan`s (`mul!`, `*`, `inv`, `\`, `fftdims`,
  `plan_inv`). FFTL does **not** register `AbstractFFTs.plan_fft`/`fft` for standard arrays, so
  loading it next to FFTW.jl does not take over `fft`. The docs say this explicitly. Niklas's remark
  "we implement AbstractFFTs" is true for the plan interface only. Changing this is Tobias's call.
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
- **Benchmarks:** only a stub until Tobias asks for the dedicated benchmark work. The page lists the
  protocol (A/B on one machine, interleaved medians, A/A noise band, held-out sizes, accuracy against a
  high-precision reference, first plan and execution timed separately) and the machines. No numbers.
- **libfftl:** size (about 23 MB with the Julia runtime, macOS arm64), build requirements (Julia 1.12+
  with juliac), build variables (`FFTL_THREADS`, `FFTL_TRIM`, `JULIA_CPU_TARGET`), baked thread pool,
  column-major data, unnormalised backward transform, error handling via `fftl_last_error`, concurrency
  rules. The C example in the how-to was compiled and run against the build in `FFTL.jl/libfftl`.
- **Logo:** from `FFTL.jl/docs/assets/fftl-logo.svg` (created by the package agent on branch
  `radix-order`, untracked). Copied, not moved. `src/assets/logo.png` is rendered from the SVG with a
  transparent background (`rsvg-convert`). If the SVG changes, re-render the PNG.

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

- GitHub workflow for the docs (`deploydocs` is in `make.jl`; no CI job yet).
- Dark mode: Documenter's theme still offers a dark-mode switch; Tobias wants light mode only.
  The tutorial figures have a white background, so they read in both themes.
- Benchmarks: results once the benchmark release is done.
- Package README still says "work in progress" and does not link the docs.
