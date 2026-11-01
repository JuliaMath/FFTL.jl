# Choice of the radix sequence for a Stockham chain, without measurements.
#
# Rules from brute-force searches over all decompositions (M1, Float32 and Float64, n ≤ 4·10^6):
# - n ≤ 16 with small factors: one codelet.
# - Powers of two: radix 4 throughout, one 8 (or 2) for an odd exponent; radix 16 and above lose
#   (register spills, cache-set conflicts at large strides).
# - A factor 3 is merged with a 4 into 12, else with a 2 into 6; two 3s give 9.
# - A factor 5 is merged with a spare 2 into 10, else stays 5. An odd power of two is split as
#   4s and a 2 (not an 8) when there is a 3 or 5 to merge the 2 with.
# - Other primes up to MAX_RADIX_PRIME are radices of their own.
# The first stage takes a power-of-two radix when there is one.
# - Batched transforms with a large stride S (a dimension ≥ 2 of a big array): a radix-R stage
#   reads R streams S·m elements apart, which share cache sets when R ≥ 16 (64³, dimension 3:
#   (4, 16) 1123 µs, (4, 4, 4) 408 µs). Above LARGE_STRIDE no radix is larger than 8.

"""
    MAX_RADIX_PRIME

Largest prime factor handled by a codelet; lengths with larger prime factors use Bluestein's
algorithm.
"""
const MAX_RADIX_PRIME = 31

function factorize(n::Int)
  f = Dict{Int,Int}()
  for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
    while n % p == 0
      f[p] = get(f, p, 0) + 1
      n ÷= p
    end
  end
  f, n          # n > 1: remaining part with prime factors > MAX_RADIX_PRIME
end

smooth(n::Int) = factorize(n)[2] == 1

const LARGE_STRIDE = 256

"""
    radices(n, S = 1)

Radix sequence for a chain of length `n` applied to `S` interleaved sequences.
"""
function radices(n::Int, S::Int = 1)
  n == 1 && return Int[]
  f, rest = factorize(n)
  rest == 1 || throw(ArgumentError("length $n has prime factors larger than $MAX_RADIX_PRIME"))
  small = S > LARGE_STRIDE ? 8 : 16
  if ispow2(n) && n > small && n <= 64
    return S > LARGE_STRIDE ? (n == 64 ? [4, 4, 4] : [4, n ÷ 4]) : [4, n ÷ 4]
  end
  (n <= small || (length(f) == 1 && haskey(f, n))) && return [n]   # small n or a prime
  a = get(f, 2, 0); b = get(f, 3, 0); c = get(f, 5, 0)
  r = Int[]
  eight = 0; two = 0
  if isodd(a)
    (a >= 3 && b + c == 0) ? (eight = 1; a -= 3) : (two = 1; a -= 1)
  end
  fours = a ÷ 2
  while b > 0 && fours > 0
    push!(r, 12); b -= 1; fours -= 1
  end
  while b > 0 && two > 0
    push!(r, 6); b -= 1; two -= 1
  end
  while b >= 2
    push!(r, 9); b -= 2
  end
  b == 1 && push!(r, 3)
  for _ in 1:c
    if two > 0
      push!(r, 10); two -= 1
    else
      push!(r, 5)
    end
  end
  for p in (7, 11, 13, 17, 19, 23, 29, 31), _ in 1:get(f, p, 0)
    push!(r, p)
  end
  pow2 = vcat(fill(8, eight), fill(4, fours), fill(2, two))
  # first stage: a power-of-two radix if available, then the rest, remaining powers of two last
  if isempty(pow2)
    return sort(r; rev = true)
  end
  vcat(pow2[1], r, pow2[2:end])
end
