using Documenter, DocumenterVitepress, Literate, FFTL

const DOCS = @__DIR__
const ROOT = dirname(DOCS)
const GENERATED = joinpath(DOCS, "src", "generated")

# Literate scripts are executed when the documentation is built: every example in
# getting started and in the tutorials is run, and a broken one fails the build.
const LITERATE = joinpath(DOCS, "literate")
for (dir, _, files) in walkdir(LITERATE), file in filter(endswith(".jl"), files)
  rel = relpath(dir, LITERATE)
  outdir = joinpath(GENERATED, rel)
  mkpath(outdir)
  Literate.markdown(joinpath(dir, file), outdir; documenter = true)
end

# The C API reference is the headers themselves, so the pages cannot drift from the code.
mkpath(GENERATED)
for (h, page) in (("fftw3.h", "libfftl_fftw3_header.md"), ("fftl.h", "libfftl_header.md"))
  header = read(joinpath(ROOT, "libfftl", "include", h), String)
  write(joinpath(GENERATED, page), "# C header ($h)\n\n```c\n" * header * "```\n")
end

# The two figures of the benchmarks page are generated from the committed overview data
# (bench/results) on every build, so a fresh clone reproduces them without any measurement.
# Each script runs in its own process: they are standalone scripts that define their own
# constants and print their summary to the build log.
for f in ("fig_standing.jl", "fig_speedup.jl")
    run(`$(Base.julia_cmd()) --project=$(joinpath(ROOT, "docs")) $(joinpath(ROOT, "bench", f))`)
end

DocMeta.setdocmeta!(FFTL, :DocTestSetup, :(using FFTL, LinearAlgebra); recursive = true)

makedocs(;
  modules = [FFTL],
  sitename = "FFTL.jl",
  authors = "Tobias Knopp and contributors",
  repo = Remotes.GitHub("JuliaMath", "FFTL.jl"),
  format = DocumenterVitepress.MarkdownVitepress(;
    repo = "https://github.com/JuliaMath/FFTL.jl",
    devurl = "dev",
    deploy_url = "JuliaMath.github.io/FFTL.jl",
  ),
  # Diátaxis, as in NFFT.jl: learn, do, understand, look up
  pages = [
    "Home" => "index.md",
    "Getting Started" => [
      "Your first transform" => "generated/getting_started.md",
    ],
    "Tutorials" => [
      "Spectrum of a sampled signal" => "generated/tutorials/spectrum.md",
      "Compressing an image with the DCT" => "generated/tutorials/image_dct.md",
      "Solving Poisson's equation with the DST" => "generated/tutorials/poisson.md",
      "Fast convolution" => "generated/tutorials/convolution.md",
    ],
    "How-to Guides" => [
      "Coming from FFTW.jl" => "howto/fftw.md",
      "Plans and reuse" => "howto/plans.md",
      "Transforms along dimensions" => "howto/dimensions.md",
      "Real data" => "howto/real.md",
      "Cosine, sine and Hartley transforms" => "howto/cosine_sine.md",
      "Threads" => "howto/threads.md",
      "Number types" => "howto/number_types.md",
      "Precompilation" => "howto/precompile.md",
      "GPU" => "howto/gpu.md",
      "C library (libfftl)" => "howto/libfftl.md",
      "Compiling ahead of time" => "howto/trim.md",
      "Contribute" => "howto/contribute.md",
    ],
    "Explanation" => [
      "The discrete Fourier transform" => "explanation/dft.md",
      "FFTW and FFTL" => "explanation/fftw.md",
      "Algorithms" => "explanation/algorithms.md",
      "Implementation" => "explanation/implementation.md",
      "Performance" => "explanation/performance.md",
    ],
    "Reference" => [
      "Julia API" => "reference/api.md",
      "AbstractFFTs" => "reference/abstractffts.md",
      "libfftl C API" => "reference/libfftl.md",
      "C header (fftw3.h)" => "generated/libfftl_fftw3_header.md",
      "C header (fftl.h)" => "generated/libfftl_header.md",
      "Platforms and limits" => "reference/platforms.md",
      "Benchmarks" => "reference/benchmarks.md",
    ],
  ],
  # FFTL exports nothing, so there are no exports for checkdocs to walk. The names in the
  # `@docs` blocks of reference/api.md are the enforced list: a binding without a docstring
  # fails the build.
  checkdocs = :none,
)

DocumenterVitepress.deploydocs(; repo = "github.com/JuliaMath/FFTL.jl.git", devbranch = "main", push_preview = true)
