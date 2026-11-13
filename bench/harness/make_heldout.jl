# Draws the held-out sizes once, with a fixed seed, and writes heldout.txt. The file is committed
# before any campaign run; the campaign reads it and never draws again. Re-running this script
# with the same seed reproduces the same file.
#
#   julia --project=. harness/make_heldout.jl
#
# Ranges follow the tuned catalog: 1-D lengths from 16 to 2^21 (log-uniform), 2-D dimensions from
# 16 to 1500, 3-D dimensions from 16 to 160. Smooth and non-smooth 1-D lengths are drawn separately
# (four non-smooth ones, twelve smooth). Duplicates of tuned sizes are dropped and replaced.

using Random

include(joinpath(@__DIR__, "..", "overview", "run_kinds.jl"))
include(joinpath(@__DIR__, "catalog.jl"))

const SEED = 20261110
const OUT = joinpath(@__DIR__, "heldout.txt")

function main()
  rng = MersenneTwister(SEED)
  tuned = Set(vcat(C2C_SHAPES, REAL_SHAPES, DCT_SHAPES, OTHER_SHAPES))
  logunif(lo, hi) = round(Int, exp(log(lo) + rand(rng) * (log(hi) - log(lo))))
  # Draws shapes until `want` distinct ones pass `keep` and are not tuned sizes.
  function draw(make, keep, want)
    got = Tuple[]
    while length(got) < want
      s = make()
      (keep(s) && !(s in tuned) && !(s in got)) && push!(got, s)
    end
    got
  end
  smooth = draw(() -> (logunif(16, 2^21),), s -> !is_nonsmooth(s), 12)
  nonsmooth_1d = draw(() -> (logunif(2^10, 2^20),), is_nonsmooth, 4)
  two_d = draw(() -> (rand(rng, 16:1500), rand(rng, 16:1500)), _ -> true, 4)
  three_d = draw(() -> (rand(rng, 16:160), rand(rng, 16:160), rand(rng, 16:160)), _ -> true, 3)

  open(OUT, "w") do io
    println(io, "# Held-out sizes: seed $SEED, drawn by harness/make_heldout.jl. Format: dims, then the shape.")
    println(io, "# Never used for tuning; do not edit, draw a new set under a new seed instead.")
    for s in vcat(smooth, nonsmooth_1d, two_d, three_d)
      println(io, length(s), " ", join(s, " "))
    end
  end
  println("wrote ", OUT, ": ", length(smooth), " smooth 1-D, ", length(nonsmooth_1d), " non-smooth 1-D, ",
          length(two_d), " 2-D, ", length(three_d), " 3-D")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
