# Choice of the radix sequence for a Stockham chain, without measurements.
#
# Rules from brute-force searches over all decompositions (M1, Float32 and Float64, n ≤ 4·10^6):
# - n ≤ 16 with small factors: one codelet.
# - Powers of two up to 16384 (stride ≤ LARGE_STRIDE), aarch64 only: radix 8 as often as possible
#   and a 4 (two 4s for one spare bit) for the rest, 2–7 % faster than 4s with one 8 up to 4096
#   (M1, F32 and F64); on Zen 3 (AVX2, 16 vector registers) it is 3–25 % slower from 256 on.
#   Otherwise radix 4 throughout, one 8 (or 2) for an odd exponent; radix 16 and above lose
#   (register spills, cache-set conflicts at large strides).
# - aarch64: a factor 3 is merged with a 4 into 12, else with a 2 into 6; a factor 5 with a spare
#   2 into 10 (see merge_composite for x86). Two 3s give 9. An odd power of two is split as 4s and
#   a 2 (not an 8) when there is a 3 or 5 to merge the 2 with (x86: or a 7).
# - Other primes up to MAX_RADIX_PRIME are radices of their own.
# The first stage takes a power-of-two radix when there is one.
# - Batched transforms with a large stride S (a dimension ≥ 2 of a big array): a radix-R stage
#   reads R streams S·m elements apart, which share cache sets when R ≥ 16 (64³, dimension 3:
#   (4, 16) 1123 µs, (4, 4, 4) 408 µs). Above LARGE_STRIDE powers of two use radices ≤ 8, and a
#   choice whose streams fall into the same L1 sets is replaced by packed radices ≤ 8.

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

# Merging a factor 3 with a 4 or 2 (radix 12, 6) and a 5 with a spare 2 (radix 10) pays on aarch64
# (NEON, 32 registers). For a single contiguous line (S = 1) on x86 with AVX2 (16 registers) the
# radices 4, 3, 5, 7 alone are faster: of the lengths up to 1024 with prime factors ≤ 7 (Zen 2, all
# radix multisets measured), the merged choice is up to 2.4× slower than the best one, plain radices
# at most 1.4× on lengths not used to choose the rule. Batches (S > 1) keep the merged radices; they
# have not been measured on x86 for every length.
merge_composite(S) = Sys.ARCH === :aarch64 || S > 1

"""
    radices(n, S = 1; esize = 16)

Radix sequence for a chain of length `n` applied to `S` interleaved sequences of elements of
`esize` bytes.
"""
function radices(n::Int, S::Int = 1; esize::Int = 16)
  r = heuristic_radices(n, S)
  S > LARGE_STRIDE && set_conflict(r, n, S, esize) || return r
  alt = packed_radices(n)
  set_conflict(alt, n, S, esize) ? r : alt
end

# A stage of radix R reads and writes R streams S·n/R elements apart. When that spacing is a
# multiple of the L1 set stride, the streams share a cache set and more than 8 of them thrash
# (48³ dimension 3: (4, 12) 392 µs, (6, 8) 107 µs; 144³: (12, 12) 12.8 ms, (4, 6, 6) 5.6 ms).
const L1_SET_LOG2 = Sys.ARCH === :aarch64 ? 14 : 12
const RADIX8_POW2 = Sys.ARCH === :aarch64

function set_conflict(rs, n, S, esize = 16)
  for r in rs
    v = trailing_zeros(S * (n ÷ r) * esize)
    cld(r, 1 << clamp(L1_SET_LOG2 - v, 0, L1_SET_LOG2)) > 8 && return true
  end
  false
end

# Prime factors packed into radices of at most 8, powers of two first.
function packed_radices(n::Int)
  f, _ = factorize(n)
  bins = Int[]
  for p in reverse!(sort!(reduce(vcat, (fill(p, e) for (p, e) in f); init = Int[])))
    i = findfirst(b -> b * p <= 8, bins)
    i === nothing ? push!(bins, p) : (bins[i] *= p)
  end
  # powers of two first, each group in decreasing order
  [reverse!(sort!(filter(ispow2, bins))); reverse!(sort!(filter(!ispow2, bins)))]
end

function heuristic_radices(n::Int, S::Int)
  n == 1 && return Int[]
  merge = merge_composite(S)
  f, rest = factorize(n)
  rest == 1 || throw(ArgumentError("length $n has prime factors larger than $MAX_RADIX_PRIME"))
  small = S > LARGE_STRIDE ? 8 : 16
  if ispow2(n) && n > small && n <= 64
    return S > LARGE_STRIDE ? (n == 64 ? [4, 4, 4] : [4, n ÷ 4]) : [4, n ÷ 4]
  end
  if RADIX8_POW2 && ispow2(n) && 64 < n <= 16384 && S <= LARGE_STRIDE
    k = trailing_zeros(n)
    spare = k % 3
    return vcat(spare == 2 ? [4] : spare == 1 ? [4, 4] : Int[], fill(8, (k - (spare == 2 ? 2 : spare == 1 ? 4 : 0)) ÷ 3))
  end
  (n <= small || (length(f) == 1 && haskey(f, n))) && return [n]   # small n or a prime
  a = get(f, 2, 0); b = get(f, 3, 0); c = get(f, 5, 0)
  r = Int[]
  eight = 0; two = 0
  if isodd(a)
    (a >= 3 && b + c == 0 && (merge || !haskey(f, 7))) ? (eight = 1; a -= 3) : (two = 1; a -= 1)
  end
  fours = a ÷ 2
  while merge && b > 0 && fours > 0
    push!(r, 12); b -= 1; fours -= 1
  end
  while merge && b > 0 && two > 0
    push!(r, 6); b -= 1; two -= 1
  end
  while b >= 2
    push!(r, 9); b -= 2
  end
  b == 1 && push!(r, 3)
  for _ in 1:c
    if merge && two > 0
      push!(r, 10); two -= 1
    else
      push!(r, 5)
    end
  end
  for p in (7, 11, 13, 17, 19, 23, 29, 31), _ in 1:get(f, p, 0)
    push!(r, p)
  end
  pow2 = vcat(fill(8, eight), fill(4, fours), fill(2, two))
  # Single lines on x86: powers of two first, then the other radices from the smallest to the largest.
  # The largest radix last is the fastest order: (3, 13) for 39 takes 0.65–0.75 of (13, 3), and the
  # ascending order is the best measured one for 91 % of the lengths up to 1024 (Zen 2).
  if !merge
    return vcat(sort(pow2; rev = true), sort!(r))
  end
  # aarch64 and batches: a power-of-two radix first, then the rest, remaining powers of two last
  if isempty(pow2)
    return reverse!(sort(r))
  end
  vcat(pow2[1], r, pow2[2:end])
end
