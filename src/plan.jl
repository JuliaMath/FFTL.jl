# Kernels for one dimension: a Stockham chain (`RChain`, rchain.jl) for lengths whose prime factors
# are ≤ MAX_RADIX_PRIME, Bluestein's algorithm on top of two chains otherwise. Both transform a batch
# of S interleaved sequences (element j of lane i at offset i + S·j), see kernels.jl.

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

# Read-only parts of kernels (chains, four-step twiddles, Bluestein chirp and filter) are shared
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

"""
    Bluestein{T}

DFT of any length n as a cyclic convolution of length M ≥ 2n - 1 (M smooth):
  X_k = c_k · Σ_j (x_j c_j) · conj(c_{k-j}),  c_j = exp(σ·πi·j²/n).
"""
struct Bluestein{T,F,B} <: Kernel{T}
  n::Int
  chirp::Vector{Complex{T}}       # c_j, j = 0…n-1
  kernel::Vector{Complex{T}}      # DFT of the convolution kernel, scaled by 1/M
  fwd::F
  bwd::B
  buf::Vector{Complex{T}}
  work1::Vector{Complex{T}}
  work2::Vector{Complex{T}}
end

# 5-smooth numbers: the convolution lengths of the aarch64 rule.
function next_smooth(n::Int)
  m = n
  while true
    f, rest = factorize(m)
    rest == 1 && all(p -> p <= 5, keys(f)) && return m
    m += 1
  end
end

# Convolution length of Bluestein's algorithm: a 7-smooth M in [L, 1.4 L] with the lowest estimated
# time of a chain of length M, time = M·(c0 + Σ_stages w[radix]) in µs. The stage costs are fitted
# to measured chains (Zen 2, M = 70…300000, one thread); a length with many radix-9, -5 or -3
# stages transforms several times slower than a power of two a few percent larger (Float32:
# 2025 = 3^4·5^2 takes 11 µs, 2048 takes 2.5 µs). The aarch64 stage costs have not been measured:
# there the smallest 5-smooth M is kept.
const BLUESTEIN_RADIX = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
# weights for BLUESTEIN_RADIX, then any other radix, then the constant per point
const BLUESTEIN_W = (Float64 = (0.0, 1.11e-3, 3.5e-4, 7.4e-4, 2.9e-4, 9.9e-4, 1.7e-4, 1.52e-3, 5.6e-4, 7.6e-4, 2.38e-3, 2.6e-4),
                     Float32 = (4.1e-4, 8.2e-4, 9.0e-5, 4.0e-4, 4.9e-4, 6.2e-4, 0.0, 1.08e-3, 3.4e-4, 1.6e-4, 1.6e-3, 5.1e-4))

function bluestein_cost(::Type{T}, m::Int) where {T}
  w = BLUESTEIN_W[T === Float32 ? :Float32 : :Float64]
  c = w[end]
  for r in radices(m)
    i = findfirst(==(r), BLUESTEIN_RADIX)
    c += i === nothing ? w[end-1] : w[i]
  end
  m * c
end

function bluestein_length(::Type{T}, L::Int) where {T}
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
            c = bluestein_cost(T, p7)
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

# C: TypedKernels, or RChain for run-time plans (their convolution transforms are chains)
function Bluestein{T}(n::Int, σ::Int, ::Type{C} = TypedKernels) where {T,C}
  M = bluestein_length(T, 2n - 1)
  # one assignment each: the closure below captures fwd, and a variable assigned in two
  # branches would be boxed (untyped, which juliac --trim cannot resolve)
  fwd, bwd = convolution_transforms(T, M, C)
  w1 = Vector{Complex{T}}(undef, M); w2 = Vector{Complex{T}}(undef, M)
  chirp, b = cached(() -> bluestein_tables(T, n, σ, fwd, w1, w2), CacheKey(:bluestein, T, n, σ, Int[])
                   )::Tuple{Vector{Complex{T}},Vector{Complex{T}}}
  Bluestein{T,typeof(fwd),typeof(bwd)}(n, chirp, b, fwd, bwd, zeros(Complex{T}, M), w1, w2)
end

# The forward and backward transforms of length M of a Bluestein kernel: four-steps on one thread
# where a line of length M would take one (`serial_fourstep`). The run-time plans keep their chains
# (one concrete type for juliac --trim).
convolution_transforms(::Type{T}, M::Int, ::Type{C}) where {T,C} = (C{T}(M, -1), C{T}(M, 1))
function convolution_transforms(::Type{T}, M::Int, ::Type{TypedKernels}) where {T}
  if serial_fourstep(T, M) && fourstep_split(M) > 0
    return (FourStep{T}(M, -1, 1), FourStep{T}(M, 1, 1))
  end
  (RChain{T}(M, -1), RChain{T}(M, 1))
end

# chirp c_j = exp(σ iπ j²/n) = W_2n^(j² mod 2n) and the filter FFT_M(conj(c) wrapped) / M
function bluestein_tables(::Type{T}, n::Int, σ::Int, fwd, w1::Vector{Complex{T}}, w2::Vector{Complex{T}}) where {T}
  M = length(w1)
  W = unit_roots(σ, 2n)
  c = Complex{T}[Complex{T}(W[mod(j * j, 2n)+1]) for j in 0:n-1]
  b = zeros(Complex{T}, M)
  b[1] = conj(c[1])
  for j in 1:n-1
    b[j+1] = b[M-j+1] = conj(c[j+1])
  end
  GC.@preserve b w1 w2 run!(cptr(b), cptr(b), cptr(w1), cptr(w2), fwd, 1)
  b ./= M
  (c, b)
end

# Lane by lane: gather, convolve, scatter. Element j of lane i at px + i + Sx·j, py + i + Sy·j.
# Uses the plan's own buffers, so one Bluestein kernel runs on one thread at a time.
function run_lanes!(py::Ptr{T}, px::Ptr{T}, ::Ptr{T}, ::Ptr{T}, P::Bluestein{T}, Sx::Int, Sy::Int, B::Int) where {T}
  n = P.n; buf = P.buf; M = length(buf)
  cx = Ptr{Complex{T}}(px); cy = Ptr{Complex{T}}(py)
  GC.@preserve P begin
    pb = cptr(buf); pw1 = cptr(P.work1); pw2 = cptr(P.work2)
    for i in 1:B
      @inbounds for j in 1:n
        buf[j] = unsafe_load(cx, i + Sx * (j - 1)) * P.chirp[j]
      end
      fill!(view(buf, n+1:M), zero(Complex{T}))
      run!(pb, pb, pw1, pw2, P.fwd, 1)
      buf .*= P.kernel
      run!(pb, pb, pw1, pw2, P.bwd, 1)
      @inbounds for k in 1:n
        unsafe_store!(cy, buf[k] * P.chirp[k], i + Sy * (k - 1))
      end
    end
  end
  nothing
end

run!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::Bluestein{T}, S::Int) where {T} =
  run_lanes!(py, px, pw1, pw2, P, S, S, S)

"""
    Composite{T}

DFT of a length n = s·q with s > 1 and q coprime to it (see `composite_split`), as one
Cooley-Tukey step: with j = j2 + q·j1 and
k = k1 + s·k2, a kernel of length s over j1 (a chain if s is smooth, else again a composite or
Bluestein kernel), the twiddles w_n^(j2·k1), and Bluestein's algorithm of length q over j2. The
convolutions have length about 2q instead of 2n and stay in cache.
"""
struct Composite{T,C,B} <: Kernel{T}
  n::Int
  s::Int
  q::Int
  chain::C                        # length s, applied to the q lanes of every slab
  blue::B                         # Bluestein of length q, with its own buffers
  tw::Vector{Complex{T}}          # w_n^m, m = 0…n-1
  z::Vector{Complex{T}}           # output of the first stage, S·n values
end

function Composite{T}(n::Int, s::Int, q::Int, σ::Int, S::Int) where {T}
  chain = first_kernel(T, s, σ, S * q, S)
  blue = Bluestein{T}(q, σ)
  tw = cached(() -> Complex{T}.(unit_roots(σ, n)), CacheKey(:composite, T, n, σ, Int[]))::Vector{Complex{T}}
  Composite{T,typeof(chain),typeof(blue)}(n, s, q, chain, blue, tw, Vector{Complex{T}}(undef, S * n))
end

# Kernel of the first stage of a composite kernel: a chain over L lanes, or a kernel with buffers of
# its own (composite, Bluestein) that sees at most one lane block, or S lanes, at a time.
first_kernel(::Type{T}, s::Int, σ::Int, L::Int, S::Int) where {T} =
  smooth(s) ? RChain{T}(s, σ, radices(s, L; esize = sizeof(Complex{T}))) : kernel(T, s, σ, max(composite_block(L, s), S))

# Leftovers (prime factors above MAX_RADIX_PRIME) whose convolution takes at most this many bytes
# stay one Bluestein transform. Zen 2, one thread: splitting a leftover with an 8 MiB or larger
# convolution into its prime powers is 1.02–1.4× faster (591159 = 3·193·1021: 1.4×, 1194467 = 179·6673: 1.2×),
# splitting one of 128 KiB to 2.7 MiB is 1.15–1.5× slower (262145 = 65·37·109, 1588381 = 19·41·2039):
# each extra level is one more pass over memory.
const COMPOSITE_SPLIT_BYTES = 1 << 22

# The split n = s·q of the composite kernel: q the whole leftover r (the product of the prime
# factors above MAX_RADIX_PRIME) while its convolution is small, else the full power of the
# largest prime factor of r. (n, 1) for smooth n; s = 1 means a single Bluestein transform.
function composite_split(::Type{T}, n::Int) where {T}
  _, r = factorize(n)
  r == 1 && return (n, 1)
  bluestein_length(T, 2r - 1) * sizeof(Complex{T}) <= COMPOSITE_SPLIT_BYTES && return (n ÷ r, r)
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
    for i in 0:B-1, k1 in 0:P.s-1
      twiddled_bluestein!(py, pz, P.tw, P.blue, n, P.s, B, i, k1, Sy)
    end
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

# Bluestein of length q on lane (i, k1) of the chain output z (element j2 at i + B·(j2 + q·k1)),
# after the twiddles w_n^(j2·k1); output element k2 at y + i + Sy·(k1 + s·k2).
function twiddled_bluestein!(py::Ptr{T}, pz::Ptr{T}, tw::Vector{Complex{T}}, bl::Bluestein{T}, n::Int, s::Int,
                             B::Int, i::Int, k1::Int, Sy::Int) where {T}
  q = bl.n; buf = bl.buf; M = length(buf); chirp = bl.chirp
  cz = Ptr{Complex{T}}(pz); cy = Ptr{Complex{T}}(py)
  GC.@preserve bl tw begin
    pb = cptr(buf)
    zb = i + B * q * k1 + 1
    m = 0                           # (j2·k1) mod n
    @inbounds for j in 1:q
      buf[j] = unsafe_load(cz, zb + B * (j - 1)) * tw[m+1] * chirp[j]
      m += k1; m >= n && (m -= n)
    end
    fill!(view(buf, q+1:M), zero(Complex{T}))
    run!(pb, pb, cptr(bl.work1), cptr(bl.work2), bl.fwd, 1)
    buf .*= bl.kernel
    run!(pb, pb, cptr(bl.work1), cptr(bl.work2), bl.bwd, 1)
    @inbounds for k in 1:q
      unsafe_store!(cy, buf[k] * chirp[k], i + Sy * (k1 + s * (k - 1)) + 1)
    end
  end
  nothing
end

run!(py::Ptr{T}, px::Ptr{T}, pw1::Ptr{T}, pw2::Ptr{T}, P::Composite{T}, S::Int) where {T} =
  run_lanes!(py, px, pw1, pw2, P, S, S, S)

function kernel(::Type{T}, n::Int, σ::Int, S::Int = 1) where {T}
  smooth(n) && return RChain{T}(n, σ, radices(n, S; esize = sizeof(Complex{T})))
  s, q = composite_split(T, n)
  s == 1 && return Bluestein{T}(n, σ)
  Composite{T}(n, s, q, σ, S)
end

# y[i + Sy·j] = x[i + Sx·j] for B lanes and n elements
function lane_copy!(py::Ptr{T}, px::Ptr{T}, Sy::Int, Sx::Int, B::Int, n::Int) where {T}
  cy = Ptr{Complex{T}}(py); cx = Ptr{Complex{T}}(px)
  for j in 0:n-1
    unsafe_copyto!(cy + Sy * j * sizeof(Complex{T}), cx + Sx * j * sizeof(Complex{T}), B)
  end
end
