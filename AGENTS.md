# FFTL.jl: notes for contributors and coding agents

FFTL.jl is a Fast Fourier transform written in Julia (MIT licence): complex, real and real-to-real
transforms, any dimension, threads, GPU through KernelAbstractions, other number types, and `libfftl`,
a C library with FFTW3's interface that is compiled from the same Julia code with `juliac --trim`.
Read this file before changing the package. The documentation holds the rest:

- `docs/src/howto/contribute.md`: set up, tests, where things are, what a change must come with.
- `docs/src/explanation/design-decisions.md`: what was decided and why, with the evidence.
- `docs/src/explanation/image-size.md`: why the package image was large and what keeps it small.
- `docs/src/explanation/implementation.md`, `performance.md`: how it works and how it is measured.

## Rules

- **Licence.** The package is MIT-licensed. Never read, copy, translate or paraphrase GPL code.
  - FFTW and KFR are GPL: benchmark competitors only. Concepts from published papers are fine; code
    structure, naming and style are not modelled on FFTW.
  - ducc0 is GPL-2.0-or-later as a whole. Only files with the SPDX header
    `BSD-3-Clause OR GPL-2.0-or-later` (its FFT component) may be read; the odd DCT-IV code in
    `fft1d_impl.h` is credited to FFTW's authors and must not be read.
  - VkFFT (MIT) is a benchmark competitor: read its paper and documentation, never its source code.
  - Permissive sources (FFTA.jl, pocketfft, RustFFT, AbstractFFTs.jl) may be read. Prefer an own
    implementation; if an algorithm is taken over closely, name the source in the code and keep the
    required notice. Check the licence of every new reference before reading it.
- **Claims are measured.** Every statement about speed, accuracy or size is checked by running Julia
  before it is written down, or it is marked as not verified. A change that claims to be faster comes
  with an interleaved A/B measurement on one machine (medians, and the spread of the same code against
  itself); see `docs/src/explanation/performance.md`.
- **Kernels of a fixed set are plain methods, not `@generated`.** A package image keeps the complete
  optimised IR of every precompiled `@generated` specialisation. `@generated` methods stay as the
  fallback for combinations outside the precompiled set. See `docs/src/explanation/image-size.md`.
- **Nothing private in the repository.** Code, documentation, data files, commit messages and pull request
  descriptions contain no host names or accounts of machines, no local paths, no private contact data and no
  names of people beyond the authors of the package. A machine is named by its processor ("AMD EPYC 7702",
  "Apple M1 Max"). Check before every push, including the `.meta` files next to benchmark data.
- **Decisions leave a trace in the repository.** A decision or a measured finding that other
  contributors need goes into `docs/src/explanation/design-decisions.md` or the matching explanation
  page, and into the pull request description and the commit message, in the same change.

## Build and test

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.test()'                     # quick selection
FFTL_TEST_FULL=true julia --project=. -e 'using Pkg; Pkg.test()' # full lists of sizes and types
```

CI runs the quick selection with `JULIA_NUM_THREADS=2` and the full lists once a week.
`FFTL_TEST_VERBOSE=true` prints the time of every test set. The C library is built with
`libfftl/build.sh` and tested with `libfftl/test/fftw3_ctest.sh` (see `docs/src/howto/libfftl.md`).
Documentation: `julia --project=docs docs/make.jl`.

## Code and commits

- Everything in the repository is in English: code, comments, commit messages, documentation.
- A comment explains the code as it is now: an invariant, a reason that is not obvious, a limit a later
  editor would break. It does not refer to plans, chat, or earlier versions, and it must be true for a
  reader who has only the repository. History belongs in the commit log.
- A commit message says what the code does now and why, not how the change came about.
- Work on a branch and open a pull request against `main`; do not push to `main`.

# Documentation conventions

Conventions for the package documentation (README, `docs/`).

## Logo

- Files: `docs/assets/fftl-logo.svg` (light) and `docs/assets/fftl-logo-dark.svg` (dark), both
  versioned. Figures for the README live in `docs/assets/`, nowhere else. The docs sidebar uses
  the copies in `docs/src/assets/` (`logo.svg`, `logo-dark.svg`); Documenter picks those up
  automatically, including the dark-theme switch.
- Name: "FFTL", without ".jl". The package is also built as a C library, so the name must read on
  its own.
- Layout: butterfly left, wordmark right, the spectrum underline below the wordmark only. Both
  blocks are 187 units tall, so the lockup lines up.
- Hand-built SVG, no fonts: the wordmark is drawn with strokes, so the file renders the same
  everywhere. Edit the SVG directly; the generator script is not part of the repository.
- Symbol: a radix-2 decimation-in-time butterfly for 8 points, three stages with strides 1, 2, 4
  (bit-reversed input), drawn as pure butterflies (crosses, no straight lines). Node rows cycle
  through the Julia colours; stage opacity 1, 0.55, and 0.3 (dotted) from input to output. Outer
  columns have larger nodes than the inner ones.
- Spectrum underline: the |DFT| of the first 8 digits of pi (3, 1, 4, 1, 5, 9, 2, 6), zero-padded
  to 16 points, DC bin dropped: 15 stems with a node dot on top. Dot colours cycle through the
  Julia colours, starting green.
- Colours: green `#389826`, purple `#9558B2`, red `#CB3C33`, blue `#4063D8`, ink `#1A1A1A` (light)
  or `#F2F2EE` (dark).
- Wordmark: 10-unit strokes, cap height 60, letter pitch 6 units. The "L" is Julia purple, the
  other letters are ink. The wordmark is 2× the unit scale in the lockup (120 px high next to a
  200 px symbol).
- The light file goes on light pages, the dark file on dark ones; the docs sidebar switches on its
  own. On the README and in print, use the light file.
- Favicon: `docs/assets/fftl-favicon-spectrum.svg`, the 8-point spectrum in a 32 px square.
