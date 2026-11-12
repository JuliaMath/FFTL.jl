# Kernels for one dimension: a Stockham chain (`RChain`, rchain.jl) for lengths whose prime factors
# are ≤ MAX_RADIX_PRIME, a resampled DFT on top of a chain otherwise (resample.jl). Both transform a
# batch of S interleaved sequences (element j of lane i at offset i + S·j), see kernels.jl.

abstract type Kernel{T} end

"""
    TypedKernels

Tag of the plans that carry the types of their kernels in their own type (the default plans), as
opposed to `RChain`, the tag of the run-time plans, whose kernels are of one fixed type.
"""
abstract type TypedKernels end

"""
    run!(py, px, pw1, pw2, P::Kernel, S)

Transform S interleaved sequences of length `P.n` from `px` into `py` (which may equal `px`),
using the work buffers `pw1`, `pw2` (S·n complex values each).
"""
function run! end

"""
    run_lanes!(py, px, pw1, pw2, P::Kernel, Sx, Sy, B)

Transform a block of B lanes; element j of lane i at `px + i + Sx·j` and `py + i + Sy·j`
(py may equal px). The work buffers hold B·n complex values each.
"""
function run_lanes! end

# Read-only parts of kernels (chains, four-step twiddles, resampling weights) are shared
# between plans and threads: a second plan of a size costs a lookup. Bounded; when full, an
# arbitrary entry is dropped. `clear_cache!()` frees everything.
# Keys and values are concretely typed at every call site (kind, element type, sizes; values are
# asserted by the caller), so the cache compiles ahead of time (juliac --trim).
struct CacheKey
  kind::Symbol
  T::DataType
  n::Int
  σ::Int
  rs::Vector{Int}
end
Base.hash(k::CacheKey, h::UInt) = hash(k.rs, hash(k.σ, hash(k.n, hash(k.T, hash(k.kind, h)))))
Base.:(==)(a::CacheKey, b::CacheKey) = a.kind === b.kind && a.T === b.T && a.n == b.n && a.σ == b.σ && a.rs == b.rs

# Entries carry the tick of their last use; a full cache drops the least recently used one.
# The ticks live in their own concrete dict: the cache hit must not pass the plan element
# (typed Any) to any call, which juliac --trim=safe leaves unresolved. The plans are stored
# once per miss (a concrete value per call site) and read through the caller's type
# assertion.
const KCACHE = Dict{CacheKey,Tuple{Int,Any}}()
const KTICK = Dict{CacheKey,Int}()
const KCACHE_LOCK = ReentrantLock()
const KCACHE_MAX = 128   # a chain and its tables are two entries
const KCACHE_TICK = Ref(0)

function cached(f::F, key::CacheKey) where {F}
  @lock KCACHE_LOCK begin
    if haskey(KCACHE, key)
      KCACHE_TICK[] += 1
      KTICK[key] = KCACHE_TICK[]
      return KCACHE[key][2]
    end
    if length(KCACHE) >= KCACHE_MAX
      oldest, tick = first(KTICK)[1], typemax(Int)
      for (k, t) in KTICK
        t < tick && ((oldest, tick) = (k, t))
      end
      delete!(KTICK, oldest)
      delete!(KCACHE, oldest)
    end
    w = f()
    KCACHE_TICK[] += 1
    KTICK[key] = KCACHE_TICK[]
    KCACHE[key] = (KCACHE_TICK[], w)
    w
  end
end

"""
    clear_cache!()

Drop the twiddle and filter tables that FFTL keeps for reuse by later plans.
"""
clear_cache!() = (@lock KCACHE_LOCK empty!(KCACHE); nothing)

# The first of several stages of a single line runs stage_small! (the widest vector, C complex
# values, outputs interleaved in registers) where the grouped first stage (vector index group
# of the complex counts ncs, 0 for none) would be narrower.
small_first(ncs, group::Int, r::Int, n::Int) =
  r < n && ispow2(r) && (n ÷ r) % ncs[1] == 0 && ncs[1] > (group == 0 ? 1 : ncs[group])

# Stage twiddles and first-stage table of a radix sequence.
chain_tables(::Type{T}, n::Int, σ::Int, rs::AbstractVector{Int}) where {T} =
  cached(() -> build_tables(T, n, σ, rs), CacheKey(:tables, T, n, σ, collect(Int, rs)))::Tuple{Vector{Vector{Complex{T}}},Vector{Complex{T}}}

function build_tables(::Type{T}, n::Int, σ::Int, rs::AbstractVector{Int}) where {T}
  prod(rs; init = 1) == n || throw(ArgumentError("radices do not multiply to the length"))
  W = unit_roots(σ, n)
  tw = Vector{Complex{T}}[]
  L = n
  for r in rs
    push!(tw, stage_twiddles(T, W, r, L))
    L ÷= r
  end
  (tw, first_twiddles(T, W, isempty(rs) ? 1 : rs[1]))
end

"""
    unit_roots(σ, n)

`W[j+1] = exp(σ 2πi j/n)` for `j < n` in Float64. `cispi` is evaluated on the first octant only;
the rest follows exactly from the symmetries of the unit circle (swap and sign changes), so the
table is as accurate as `cispi` itself and costs n/8 evaluations.
"""
function unit_roots(σ::Int, n::Int)
  W = Vector{ComplexF64}(undef, n)
  n == 0 && return W
  if n % 8 == 0
    q = n ÷ 4; o = n ÷ 8
    for j in 0:o
      W[j+1] = cispi(2σ * j / n)
    end
    for j in o+1:q                      # second octant: w^j = σ i conj(w^(q-j))
      c = W[q-j+1]
      W[j+1] = σ > 0 ? complex(imag(c), real(c)) : complex(-imag(c), -real(c))
    end
  elseif n % 4 == 0
    q = n ÷ 4
    for j in 0:q
      W[j+1] = cispi(2σ * j / n)
    end
  else
    for j in 0:n÷2
      W[j+1] = cispi(2σ * j / n)
    end
    for j in n÷2+1:n-1
      W[j+1] = conj(W[n-j+1])
    end
    return W
  end
  q = n ÷ 4
  for j in q+1:2q                       # second quarter: w^j = σ i w^(j-q)
    c = W[j-q+1]
    W[j+1] = σ > 0 ? complex(-imag(c), real(c)) : complex(imag(c), -real(c))
  end
  for j in 2q+1:n-1                     # second half: w^j = -w^(j-n/2)
    W[j+1] = -W[j-2q+1]
  end
  W
end

# w_L^(k p) = W_n^(k p n/L) at (r-1) p + k for k < r, p < L/r. A function of its own: a
# comprehension inside the loop above would capture the reassigned L in a box.
function stage_twiddles(::Type{T}, W::Vector{ComplexF64}, r::Int, L::Int) where {T}
  n = length(W); m = L ÷ r; step = n ÷ L
  t = Vector{Complex{T}}(undef, (r - 1) * m)
  for p in 0:m-1, k in 1:r-1
    t[(r-1)*p+k] = Complex{T}(W[(k*p*step)%n+1])
  end
  t
end

# first stage: w_n^(k p) at (k-1) m + p
function first_twiddles(::Type{T}, W::Vector{ComplexF64}, r::Int) where {T}
  n = length(W); m = n ÷ r
  t = Vector{Complex{T}}(undef, (r - 1) * m)
  for k in 1:r-1, p in 0:m-1
    t[(k-1)*m+p+1] = Complex{T}(W[(k*p)%n+1])
  end
  t
end

# 5-smooth numbers: the grid lengths of the aarch64 rule.
function next_smooth(n::Int)
  m = n
  while true
    f, rest = factorize(m)
    rest == 1 && all(p -> p <= 5, keys(f)) && return m
    m += 1
  end
end

# Grid length of a resampled DFT (see resample.jl): a 7-smooth M in [L, 1.4 L] with the lowest
# estimated time of a chain of length M, time = M·(c0 + Σ_stages w[radix]) in µs. The stage costs are
# fitted to measured chains (Zen 2, M = 70…300000, one thread); a length with many radix-9, -5 or -3
# stages transforms several times slower than a power of two a few percent larger (Float32:
# 2025 = 3^4·5^2 takes 11 µs, 2048 takes 2.5 µs). The aarch64 stage costs have not been measured:
# there the smallest 5-smooth M is kept.
const GRID_RADIX = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
# weights for GRID_RADIX, then any other radix, then the constant per point
const GRID_W = (Float64 = (0.0, 1.11e-3, 3.5e-4, 7.4e-4, 2.9e-4, 9.9e-4, 1.7e-4, 1.52e-3, 5.6e-4, 7.6e-4, 2.38e-3, 2.6e-4),
                Float32 = (4.1e-4, 8.2e-4, 9.0e-5, 4.0e-4, 4.9e-4, 6.2e-4, 0.0, 1.08e-3, 3.4e-4, 1.6e-4, 1.6e-3, 5.1e-4))

function grid_cost(::Type{T}, m::Int) where {T}
  w = GRID_W[T === Float32 ? :Float32 : :Float64]
  c = w[end]
  for r in radices(m)
    i = findfirst(==(r), GRID_RADIX)
    c += i === nothing ? w[end-1] : w[i]
  end
  m * c
end

function grid_length(::Type{T}, L::Int) where {T}
  Sys.ARCH === :aarch64 && return next_smooth(L)
  hi = ceil(Int, 1.4L)
  best = 0; bestc = Inf
  p2 = 1
  while p2 <= hi
    p3 = p2
    while p3 <= hi
      p5 = p3
      while p5 <= hi
        p7 = p5
        while p7 <= hi
          if p7 >= L
            c = grid_cost(T, p7)
            c < bestc && (best = p7; bestc = c)
          end
          p7 *= 7
        end
        p5 *= 5
      end
      p3 *= 3
    end
    p2 *= 2
  end
  best
end

# Grid transforms of at least this many bytes run as a four-step on one thread: on Zen 2 a
# Stockham chain beyond the cache drops to 7 GFLOP/s, the four-step is 1.06× faster at 2^19
# (8 MiB, Float64) and 1.3× from 2^20 on, 0.8× at 3·2^17 and below. On the M1 Max the chain is
# faster at every length up to 2^22.
const GRID_FOURSTEP_BYTES = Sys.ARCH === :aarch64 ? typemax(Int) : 1 << 23

"""
    Composite{T}

DFT of a length n = s·q with s > 1 and q coprime to it (see `composite_split`), as one
Cooley-Tukey step: with j = j2 + q·j1 and
k = k1 + s·k2, a kernel of length s over j1 (a chain if s is smooth, else again a composite or
resampling kernel), the twiddles w_n^(j2·k1), and a resampled DFT of length q over j2. The
transforms have length about 2q instead of 2n and stay in cache.
"""
struct Composite{T,C,B} <: Kernel{T}
  n::Int
  s::Int
  q::Int
  chain::C                        # length s, applied to the q lanes of every slab
  rs::B                           # length q over the lanes (RsBatch, or Resample for few lanes), with its own buffers
  tw::Vector{Complex{T}}          # w_n^m, m = 0…n-1
  z::Vector{Complex{T}}           # output of the first stage, S·n values
end

function Composite{T}(n::Int, s::Int, q::Int, σ::Int, S::Int) where {T}
  chain = first_kernel(T, s, σ, S * q, S)
  rs = leftover_kernel(T, q, σ, s)
  tw = cached(() -> Complex{T}.(unit_roots(σ, n)), CacheKey(:composite, T, n, σ, Int[]))::Vector{Complex{T}}
  Composite{T,typeof(chain),typeof(rs)}(n, s, q, chain, rs, tw, Vector{Complex{T}}(undef, S * n))
end

# Kernel of the first stage of a composite kernel: a chain over L lanes, or a kernel with buffers of
# its own (composite, resampling) that sees at most one lane block, or S lanes, at a time.
first_kernel(::Type{T}, s::Int, σ::Int, L::Int, S::Int) where {T} =
  smooth(s) ? RChain{T}(s, σ, radices(s, L; esize = sizeof(Complex{T}))) : kernel(T, s, σ, max(composite_block(L, s), S))

# Leftovers (prime factors above MAX_RADIX_PRIME) whose grid takes at most this many bytes stay one
# resampled transform. Measured with Bluestein's algorithm (Zen 2, one thread): splitting a leftover
# with an 8 MiB or larger convolution into its prime powers is 1.02–1.4× faster (591159 = 3·193·1021:
# 1.4×, 1194467 = 179·6673: 1.2×), splitting one of 128 KiB to 2.7 MiB is 1.15–1.5× slower
# (262145 = 65·37·109, 1588381 = 19·41·2039): each extra level is one more pass over memory.
const COMPOSITE_SPLIT_BYTES = 1 << 22

# The split n = s·q of the composite kernel: q the whole leftover r (the product of the prime
# factors above MAX_RADIX_PRIME) while its grid is small, else the full power of the largest prime
# factor of r. (n, 1) for smooth n; s = 1 means a single resampled transform.
function composite_split(::Type{T}, n::Int) where {T}
  _, r = factorize(n)
  r == 1 && return (n, 1)
  grid_length(T, 2r) * sizeof(Complex{T}) <= COMPOSITE_SPLIT_BYTES && return (n ÷ r, r)
  p = r; d = MAX_RADIX_PRIME + 2
  while d * d <= r                  # largest prime factor of r (all its factors are above MAX_RADIX_PRIME)
    while r % d == 0
      p = d; r ÷= d
    end
    d += 2
  end
  r > 1 && (p = r)
  q = 1
  m = n
  while m % p == 0
    q *= p; m ÷= p
  end
  (n ÷ q, q)
end

# Block of B lanes, element j of lane i at px + i + Sx·j and py + i + Sy·j. The chain stage runs
# on lane blocks (as the threaded kernel does), so that serial and threaded plans agree bitwise.
function run_lanes!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::Composite{T}, Sx::Int, Sy::Int, B::Int) where {T}
  n = P.n; q = P.q
  length(P.z) >= B * n || throw(ArgumentError("composite kernel built for fewer lanes"))
  CT = 2 * sizeof(T)
  GC.@preserve P begin
    pz = cptr(P.z)
    if Sx == B
      chain_blocks!(pz, px, pw1, pw2, P.chain, B * q, 0, B * q)
    else
      for j2 in 0:q-1
        run_lanes!(pz + B * j2 * CT, px + Sx * j2 * CT, pw1, pw2, P.chain, Sx * q, B * q, B)
      end
    end
    leftover!(py, pz, P.tw, P.rs, n, P.s, B, Sy)
  end
  nothing
end

# Lanes i0 … i1-1 of the first stage of a composite kernel (L lanes, contiguous), in blocks that
# fit in cache.
function chain_blocks!(pz::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, C::Kernel{T}, L::Int, i0::Int, i1::Int) where {T}
  CT = 2 * sizeof(T)
  Bb = composite_block(L, C.n)
  for a in i0:Bb:i1-1
    run_lanes!(pz + a * CT, px + a * CT, pw1, pw2, C, L, L, min(Bb, i1 - a))
  end
  nothing
end
composite_block(L, s) = min(L, max(LANE_ALIGN, LANE_ALIGN * fld(TUNING[].block_budget ÷ s, LANE_ALIGN)))

run!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::Composite{T}, S::Int) where {T} =
  run_lanes!(py, px, pw1, pw2, P, S, S, S)

function kernel(::Type{T}, n::Int, σ::Int, S::Int = 1) where {T}
  smooth(n) && return RChain{T}(n, σ, radices(n, S; esize = sizeof(Complex{T})))
  s, q = composite_split(T, n)
  s == 1 && return Resample{T}(n, σ)
  Composite{T}(n, s, q, σ, S)
end

# y[i + Sy·j] = x[i + Sx·j] for B lanes and n elements
function lane_copy!(py::Ptr{T}, px::Ptr{T}, Sy::Int, Sx::Int, B::Int, n::Int) where {T}
  cy = Ptr{Complex{T}}(py); cx = Ptr{Complex{T}}(px)
  for j in 0:n-1
    unsafe_copyto!(cy + Sy * j * sizeof(Complex{T}), cx + Sx * j * sizeof(Complex{T}), B)
  end
end
