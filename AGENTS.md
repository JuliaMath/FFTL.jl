# Documentation conventions

Conventions for the package documentation (README, `docs/`). Code conventions are in the workspace `AGENTS.md`.

## Logo

- File: `docs/assets/fftl-logo.svg` (versioned). Figures for the README live in `docs/assets/`, nowhere else.
- Name: "FFTL", without ".jl". The package is also built as a C library, so the name must read on its own.
- Hand-built SVG, no fonts: the wordmark is drawn with strokes, so the file renders the same everywhere.
  Edit the SVG directly; there is no generator in the repository.
- Symbol: a radix-2 decimation-in-time butterfly for 8 points, three stages with strides 1, 2, 4
  (bit-reversed input). Node rows cycle through the Julia colours; stage opacity 1, 0.55, and 0.3
  (dotted) from input to output. Outer columns have larger nodes than the inner ones.
- Colours: green `#389826`, purple `#9558B2`, red `#CB3C33`, blue `#4063D8`, ink `#1A1A1A`.
- Wordmark: 10-unit strokes, cap height 60, letter pitch 6 units. The "L" is Julia purple, the other
  letters are ink. The wordmark is 2× the unit scale in the lockup (120 px high next to a 200 px symbol).
- Use on a light background. The strokes are dark ink, so a dark theme needs a white plate behind the logo.
- The stage opacities carry the depth; at 24 px the symbol reads as a pattern, which is enough for a favicon.
