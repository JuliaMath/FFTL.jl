# Catalog of benchmark cases: (kind, shape, family).
#
# Families:
#   tuned      the sizes of the old overview (overview/run_kinds.jl). FFTL's thresholds were measured
#              on them, so they are not a held-out test.
#   nonsmooth  lengths with a prime factor above 31 (FFTL's codelet limit, MAX_RADIX_PRIME in
#              src/factor.jl), 1-D and multi-dimensional.
#   heldout    sizes drawn once with a fixed seed (harness/make_heldout.jl, committed in
#              heldout.txt before the campaign). Never used for tuning.
#
# Included by measure.jl after overview/run_kinds.jl, which provides the tuned shape lists.

const NONSMOOTH_SHAPES = [
  (37,), (74,), (2018,), (4036,), (16144,), (20014,), (37888,), (64576,), (80056,),
  (37, 37), (101, 101), (1009, 16), (1009, 1009), (101, 101, 101)]

# Largest prime factor of n.
function largest_prime_factor(n)
  m, p, last = n, 2, 1
  while p * p <= m
    while m % p == 0
      last = p; m ÷= p
    end
    p += 1
  end
  m > 1 ? max(last, m) : last
end

const CODELET_LIMIT = 31

# Every multi-dimensional shape counts as non-smooth if one of its dimensions is.
is_nonsmooth(shape) = any(d -> largest_prime_factor(d) > CODELET_LIMIT, shape)

const HELDOUT_FILE = joinpath(@__DIR__, "heldout.txt")

# Lines of heldout.txt: the dimension count, then the shape (e.g. "2 1234 77").
function heldout_shapes()
  shapes = Tuple[]
  for l in eachline(HELDOUT_FILE)
    (isempty(l) || startswith(l, "#")) && continue
    v = parse.(Int, split(l))
    push!(shapes, Tuple(v[2:end]))
  end
  shapes
end

# Kinds sharing one shape list. The other r2r kinds only have the tuned shapes (no new families yet).
function full_catalog()
  nonsmooth = NONSMOOTH_SHAPES
  heldout = heldout_shapes()
  for s in nonsmooth
    is_nonsmooth(s) || error("nonsmooth shape $s has no prime factor above $CODELET_LIMIT")
  end
  entry(kind, shapes, fam) = [(kind, s, fam) for s in shapes]
  cat = Tuple{String,Tuple,String}[]
  for (kind, tuned) in (("c2c", C2C_SHAPES), ("r2c", REAL_SHAPES), ("c2r", REAL_SHAPES),
                        ("r2r_REDFT10", DCT_SHAPES))
    append!(cat, entry(kind, tuned, "tuned"))
    append!(cat, entry(kind, nonsmooth, "nonsmooth"))
    append!(cat, entry(kind, heldout, "heldout"))
  end
  for k in R2R_OTHER
    append!(cat, entry("r2r_$k", OTHER_SHAPES, "tuned"))
  end
  cat
end

# Cases of a tier: the kinds it lists, every stride-th case.
function catalog(tier)
  kept = [c for c in full_catalog() if c[1] in tier["kinds"]]
  kept[1:tier["stride"]:end]
end
