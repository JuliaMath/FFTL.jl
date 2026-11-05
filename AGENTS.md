# Documentation conventions

Conventions for the package documentation (README, `docs/`). Code conventions are in the workspace `AGENTS.md`.

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
  everywhere. Edit the SVG directly; the generator (`logo/variants/make_variants7.py` in the
  workspace) is not part of the repository.
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
