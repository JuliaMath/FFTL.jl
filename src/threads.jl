# Execution of one dimension pass with S lanes, length n and `outer` slices, on one or several
# threads.
#
# With enough lanes the pass runs in lane blocks (`run_lanes!`): a block of B lanes is read from
# the array into a compact buffer, all stages run there, and the last stage writes back. B is
# chosen so that the block and its buffers stay in the L2 cache, so a pass over a large array
# costs one read and one write of memory instead of one per stage; the blocks are distributed
# over the threads. Passes with few lanes (S small, e.g. the first dimension) run whole slices,
# on several threads when there are enough slices.

# Lane blocks are multiples of 8 complex values (a 128-byte cache line of ComplexF64).
const LANE_ALIGN = 8
# Complex values per lane block buffer (a task holds two or three): 2^16 measured best on the
# M1 (3 MB of L2 per core) and on Zen 3 with one thread, which then has the 32 MB L3 to itself
# (`TUNING`, field block_budget).
# With several threads, long lines (n >= 1024) take blocks of half the L2 per core in ComplexF64
# values: 2^14 on Zen 3 (512 KB L2 per core, shared L3; 16 threads: 2^22 Float64 1.6× and 2048²
# Float64 1.3× faster than with 2^16), 2^16 on the M1. One thread keeps the large blocks (Zen 3
# 2048² 1.6× slower with the small ones), as do short lines.
block_budget(n, nt) = nt > 1 && n >= 1024 ?
  clamp(prevpow2(CACHES[].l2 ÷ 32), 1 << 12, TUNING[].block_budget) : TUNING[].block_budget

# Blocks per thread: threads take blocks from a shared counter, so a slow core (or one busy
# with something else) delays the pass by one block, not by a fixed share.
const BLOCKS_PER_THREAD = 4

# Lane block width: a multiple of LANE_ALIGN, within the cache budget, and small enough that
# there are BLOCKS_PER_THREAD blocks per thread.
function lane_block(S, n, nt)
  Bc = max(LANE_ALIGN, LANE_ALIGN * fld(block_budget(n, nt) ÷ n, LANE_ALIGN))
  min(Bc, thread_block(S, nt))
end
thread_block(S, nt) = LANE_ALIGN * cld(cld(S, BLOCKS_PER_THREAD * nt), LANE_ALIGN)

# Bytes per row of a four-step lane block. A block of a four-step phase is n rows of B values that
# lie a whole matrix row apart in memory (phase 1 reads them, phase 2 writes them), so every row is
# a memory access of its own and narrow rows cost more than the bytes they bring: a property of the
# memory system, not of the vector width, so the width is set in bytes (Float32 gets twice the lanes
# of Float64). Blocks are at least this wide, beyond the cache budget of `lane_block`, while every
# thread keeps BLOCKS_PER_THREAD blocks. EPYC 9354:
# - one thread (FOURSTEP_ROW_BYTES): 2^26 Float64, 14.1 → 11.5 ns per element with 256-byte rows
#   (16 lanes) instead of 128;
# - 16 threads (TUNING[].fourstep_row_bytes, 128): Float32 with 128-byte rows (16 lanes) instead of
#   64 took 0.86–0.88 of the time at 2^24 and 6^10, 0.99–1.04 at 10^6 to 2^22; 256-byte rows were
#   slower in Float64 (6^10 1.15–1.19, 2^26 1.06–1.16), where a task's three buffers no longer fit
#   its share of the cache.
const FOURSTEP_ROW_BYTES = 256
fourstep_block(::Type{T}, S, n, nt) where {T} =
  max(lane_block(S, n, nt), min((nt > 1 ? TUNING[].fourstep_row_bytes : FOURSTEP_ROW_BYTES) ÷ sizeof(Complex{T}),
                                thread_block(S, nt)))

use_lanes(S, n, outer, nt) = S >= 2LANE_ALIGN && (S * n > TUNING[].block_budget || (nt > 1 && outer < nt))
threaded(S, n, outer, nt) = nt > 1 && S * n * outer >= TUNING[].thread_min_work

# k work buffers of n complex values (one or more per task)
thread_buffers(::Type{T}, n::Int, k::Int) where {T} = Vector{Complex{T}}[Vector{Complex{T}}(undef, n) for _ in 1:k]

# S: product of the sizes before d (lanes), outer: product of the sizes after d. Loops, not
# prod(sz[1:d-1]): a tuple sliced by a run-time range has an unknown length.
function dim_extent(sz, d)
  S = 1; outer = 1
  for i in 1:d-1
    S *= sz[i]
  end
  for i in d+1:length(sz)
    outer *= sz[i]
  end
  S, outer
end

# work buffer size per thread for dimension d
function thread_work(sz, d, nt)
  S, outer = dim_extent(sz, d)
  n = sz[d]
  use_lanes(S, n, outer, nt) ? lane_block(S, n, nt) * n :
  threaded(S, n, outer, nt) && outer >= nt ? S * n : 0
end

# Checks shared by the complex plans (FFTLPlan, RPlan); the number of threads.
function check_plan(sz, σ::Integer, threads::Integer)
  all(>=(1), sz) || throw(ArgumentError("array size must be positive"))
  (σ == 1 || σ == -1) || throw(ArgumentError("sign must be 1 or -1"))
  max(1, Int(threads))
end

# Lengths of the serial work buffers and of each task's buffers for transforms along `reg`
# (plane fusion of dimensions 1 and 2 included).
function work_sizes(sz, reg, nt::Int)
  nwork = 0; ntwork = 0
  for d in reg
    S, _ = dim_extent(sz, d)
    nwork = max(nwork, S * sz[d])
    ntwork = max(ntwork, thread_work(sz, d, nt))
  end
  if length(reg) >= 2 && reg[1] == 1 && reg[2] == 2 && fuse_planes(sz[1], sz[2], dim_extent(sz, 2)[2], nt)
    ntwork = max(ntwork, sz[1] * sz[2])
  end
  nwork, ntwork
end

# Whether dimension d is one long contiguous line (all other dimensions of length 1) for the
# four-step algorithm on nt threads.
fourstep_dim(::Type{T}, sz, d::Int, nt::Int) where {T} =
  (so = dim_extent(sz, d); so[1] == 1 && so[2] == 1 && use_fourstep(T, sz[d], nt))

chunk_range(c, nchunks, n) = (div((c - 1) * n, nchunks)):(div(c * n, nchunks) - 1)

# f(c, item) for item = 0…nitems-1 on ntasks tasks; task c owns buffer set c and takes items
# from a shared counter.
function parallel_items(f::F, nitems, ntasks) where {F}
  if min(ntasks, nitems) <= 1
    for it in 0:nitems-1
      f(1, it)
    end
    return nothing
  end
  spawn_items(f, nitems, min(ntasks, nitems))
end

# Separate function: the task closure in the same body slows the serial path above.
@noinline function spawn_items(f::F, nitems, ntasks) where {F}
  next = Threads.Atomic{Int}(0)
  # tasks waited for one by one, not with @sync (its Channel{Any} does not compile ahead of time);
  # the calling task works as task 1 instead of waiting idle
  ts = Vector{Task}(undef, ntasks - 1)
  for c in 2:ntasks
    ts[c-1] = Threads.@spawn take_items(f, next, nitems, c)
  end
  take_items(f, next, nitems, 1)
  foreach(wait, ts)
  nothing
end

function take_items(f::F, next, nitems, c) where {F}
  it = Threads.atomic_add!(next, 1)
  while it < nitems
    f(c, it)
    it = Threads.atomic_add!(next, 1)
  end
end

# Two dependent phases on the same tasks: f1(c, item) for item = 0…n1-1, then f2(c, item) for
# item = 0…n2-1, with one fork-join. A task starts on phase 2 once every item of phase 1 is
# finished; it waits only for items that running tasks have already claimed, so it cannot wait
# for a task that has not started.
function parallel_phases(ntasks, n1, f1::F1, n2, f2::F2) where {F1,F2}
  nt = min(ntasks, max(n1, n2))
  if nt <= 1
    for it in 0:n1-1
      f1(1, it)
    end
    for it in 0:n2-1
      f2(1, it)
    end
    return nothing
  end
  spawn_phases(nt, n1, f1, n2, f2)
end

@noinline function spawn_phases(ntasks, n1, f1::F1, n2, f2::F2) where {F1,F2}
  next1 = Threads.Atomic{Int}(0); done1 = Threads.Atomic{Int}(0); next2 = Threads.Atomic{Int}(0)
  ts = Vector{Task}(undef, ntasks - 1)
  for c in 2:ntasks
    ts[c-1] = Threads.@spawn take_phases(f1, n1, f2, n2, next1, done1, next2, c)
  end
  take_phases(f1, n1, f2, n2, next1, done1, next2, 1)
  foreach(wait, ts)
  nothing
end

function take_phases(f1::F1, n1, f2::F2, n2, next1, done1, next2, c) where {F1,F2}
  it = Threads.atomic_add!(next1, 1)
  while it < n1
    f1(c, it)
    Threads.atomic_add!(done1, 1)
    it = Threads.atomic_add!(next1, 1)
  end
  while done1[] < n1
    GC.safepoint()
    ccall(:jl_cpu_pause, Cvoid, ())
  end
  it = Threads.atomic_add!(next2, 1)
  while it < n2
    f2(c, it)
    it = Threads.atomic_add!(next2, 1)
  end
end


"""
    FourStep{T}

One-dimensional DFT of length n = n1·n2 for long lines (several threads, or one thread from
`serial_fourstep`). With j = j1 + n1·j2 and k = k2 + n2·k1:
  X[k2 + n2 k1] = Σ_j1 W_n1^(j1 k1) · [W_n^(j1 k2) Σ_j2 x[j1 + n1 j2] W_n2^(j2 k2)].
1. For blocks of j1 (lanes of x seen as an n1 × n2 matrix): the block's rows copied in order into a
   cache buffer, DFTs of length n2 there, multiplication by W_n^(j1 k2), written transposed into
   A[k2 + n2·j1] (`twiddle_transpose!`, the transposition in registers).
2. DFTs of length n1 over blocks of the n2 lanes k2 of A; for outputs from
   `TUNING[].fourstep_nt_bytes` into a cache buffer whose rows go to the output with non-temporal
   stores (`rows_out!`), otherwise straight to the output in natural order.
Each step reads and writes memory once; the transposition happens in cache. x is read only by the
copy, not by a stage: its row stride n1 is a power of two for powers of two, and a stage reading
with that stride ran into cache-set conflicts (2^26 Float64, 16 threads: 1.41 times the time
without the copy). Rows of A are padded to n2 + row_pad(T) so that step 2 does not read with a
power-of-two stride either. Lane blocks have rows of at least `FOURSTEP_ROW_BYTES` (one thread) or
`TUNING[].fourstep_row_bytes` bytes.
"""
struct FourStep{T,K1,K2} <: Kernel{T}
  n::Int
  n1::Int
  n2::Int
  k1::K1                          # length n1, n2 interleaved lanes
  k2::K2                          # length n2, n1 interleaved lanes
  tw::Vector{Complex{T}}          # W_n^(j1 k2) at k2 + n2 j1 (length n), or factor tables (n1 + n2)
  buf::Vector{Complex{T}}         # A, rows of n2 + row_pad(T)
  tbuf::Vector{Vector{Complex{T}}}  # three per thread
end

# Split n = n1·n2 with n1 as close to √n as possible, both at least LANE_ALIGN (lane blocks
# take any width; a last block narrower than the vector width uses narrower vectors).
function fourstep_split(n)
  best = 0
  for d in 1:isqrt(n)
    n % d == 0 || continue
    for n1 in (d, n ÷ d)
      n2 = n ÷ n1
      n1 >= LANE_ALIGN && n2 >= LANE_ALIGN && smooth(n1) && smooth(n2) || continue
      if best == 0 || abs(log(n1 / sqrt(n))) < abs(log(best / sqrt(n)))
        best = n1
      end
    end
  end
  best
end

function FourStep{T}(n::Int, σ::Int, nt::Int, ::Type{C} = RChain) where {T,C}
  n1 = fourstep_split(n)
  n2 = n ÷ n1
  k1 = C{T}(n1, σ, radices(n1, n2; esize = sizeof(Complex{T})))
  k2 = C{T}(n2, σ, radices(n2, n1; esize = sizeof(Complex{T})))
  # Above `TUNING[].factor_twiddles` bytes the table of W_n^(j1 k2) would be as large as the data and cost a
  # memory round trip: W_n^(j1 k2) = W_n1^a · W_n^b with j1 k2 = a n2 + b, from tables of n1 + n2
  # values (2^22 Float64 on 8 threads: 5 % faster on the M1). Cache-resident sizes keep the full
  # table: the extra product costs more there (65536 Float32: 4× slower with factor tables).
  tw = cached(() -> fourstep_twiddles(T, n, σ, n1), CacheKey(:fourstep, T, n, σ, [n1]))::Vector{Complex{T}}
  nw = max(fourstep_block(T, n1, n2, nt) * n2, fourstep_block(T, n2, n1, nt) * n1)
  FourStep{T,typeof(k1),typeof(k2)}(n, n1, n2, k1, k2, tw, Vector{Complex{T}}(undef, n1 * (n2 + row_pad(T))),
                                    thread_buffers(T, nw, 3nt))
end

function fourstep_twiddles(::Type{T}, n::Int, σ::Int, n1::Int) where {T}
  n2 = n ÷ n1
  W = unit_roots(σ, n)
  # measured on the M1; on Zen 3 a threshold at its 32 MB L3 instead measures the same within noise (0.93–1.07)
  if n * sizeof(Complex{T}) > TUNING[].factor_twiddles
    t = Vector{Complex{T}}(undef, n1 + n2)
    for a in 0:n1-1; t[a+1] = Complex{T}(W[a*n2+1]); end
    for b in 0:n2-1; t[n1+b+1] = Complex{T}(W[b+1]); end
  else
    t = Vector{Complex{T}}(undef, n)
    for j1 in 0:n1-1, k2 in 0:n2-1
      t[j1*n2+k2+1] = Complex{T}(W[(j1*k2)%n+1])
    end
  end
  t
end

# Padding of the rows of A: 64 bytes, so that every row starts on a cache line (the vectorised
# twiddle-transpose stores 64 bytes at a time) and step 2 does not read with a power-of-two stride.
row_pad(::Type{T}) where {T} = 64 ÷ sizeof(Complex{T})

# One line on the calling thread (the convolution transforms of a large Bluestein kernel).
function run!(py::Ptr{T}, px::Ptr{T}, ::Ptr{T}, ::Ptr{T}, P::FourStep{T}, S::Int) where {T}
  S == 1 || throw(ArgumentError("a four-step kernel transforms one line"))
  run_fourstep!(py, px, P, 1)
end

# A single smooth line takes the four-step algorithm on several threads from TUNING[].fourstep_min
# values, and on one thread where `serial_fourstep` says so.
use_fourstep(::Type{T}, n, nt) where {T} = smooth(n) && fourstep_split(n) > 0 &&
  (nt > 1 ? n >= TUNING[].fourstep_min : serial_fourstep(T, n))

# One line of length n on one thread (also the grid of a large Bluestein kernel): the four-step
# from SERIAL_FOURSTEP_LLC times the last-level cache of a core (`CACHES[].llc`), in Float64 for any
# smooth length, in Float32 for powers of two only. Beyond the cache a chain makes one pass over
# memory per stage (10–13 stages at these lengths); the four-step makes two but is bound by its
# work in cache, about 11 ns per element in Float64 and 6–7 in Float32 from 32 MiB on. A chain of a
# large power of two also loses bandwidth to its strides (2^26 Float64: 22 GB/s, 2^24 and 10^8:
# 34 GB/s). EPYC 9354 (32 MiB L3 per core cluster), one thread, four-step time over chain time:
#   Float64  2^19 1.88  2^22 1.08  2^24 0.96  3·2^23 0.81  2^25 0.80  2^26 0.64
#            10^7 1.01  6^8 0.99  6^10 0.98  10^8 0.95
#            near 2^25 with power-of-two parts 2^11, 2^14, 2^17: 0.91, 0.97, 0.88
#   Float32  2^20 1.61  2^24 1.05  2^25 1.09  2^26 0.94  2^27 0.76
#            3·2^24 1.05  10^7 1.21  10^8 1.17
#            near 2^26 with power-of-two parts 2^12, 2^15, 2^18: 1.11, 1.14, 1.16
# In Float32 the chain moves half the bytes and stays ahead unless its strides collapse. From
# 16 × 32 MiB = 512 MiB the rule takes every measured case where the four-step is faster and none
# where it is slower; Float64 lengths of 384–506 MiB (0.81–0.97) stay on the chain. On other
# machines the multiple of the LLC is an estimate (not measured on aarch64).
const SERIAL_FOURSTEP_LLC = 16
serial_fourstep(::Type{T}, n::Int) where {T} =
  n * sizeof(Complex{T}) >= SERIAL_FOURSTEP_LLC * CACHES[].llc && (sizeof(T) >= 8 || ispow2(n))

function run_fourstep!(py::Ptr{T}, px::Ptr{T}, P::FourStep{T}, nt::Int) where {T}
  n1, n2 = P.n1, P.n2
  CT = 2 * sizeof(T)
  GC.@preserve P begin
    pa = Ptr{Complex{T}}(cptr(P.buf))
    ptw = Ptr{Complex{T}}(cptr(P.tw))
    B = min(fourstep_block(T, n1, n2, nt), fld(length(P.tbuf[1]), n2))
    nb = cld(n1, B)
    B2 = min(fourstep_block(T, n2, n1, nt), fld(length(P.tbuf[1]), n1))
    nb2 = cld(n2, B2)
    pa2 = cptr(P.buf)
    nts = P.n * CT >= TUNING[].fourstep_nt_bytes
    f1 = (c, b) -> begin
        local pw3 = cptr(P.tbuf[3c])
        local cw3 = Ptr{Complex{T}}(pw3)
        local i0 = b * B
        local Bc = min(B, n1 - i0)
        lane_copy!(pw3, px + i0 * CT, Bc, n1, Bc, n2)
        run_lanes!(pw3, pw3, cptr(P.tbuf[3c-2]), cptr(P.tbuf[3c-1]), P.k2, Bc, Bc, Bc)
        if length(P.tw) == P.n
          twiddle_transpose!(pa, cw3, ptw, i0, Bc, n2)
        else
          twiddle_transpose!(pa, cw3, ptw, i0, Bc, n1, n2)
        end
      end
    f2 = (c, b) -> begin
        local i0 = b * B2
        local Bc = min(B2, n2 - i0)
        if nts
          local pw3 = cptr(P.tbuf[3c])
          run_lanes!(pw3, pa2 + i0 * CT, cptr(P.tbuf[3c-2]), cptr(P.tbuf[3c-1]), P.k1, n2 + row_pad(T), Bc, Bc)
          rows_out!(py + i0 * CT, pw3, n2, Bc, n1)
        else
          run_lanes!(py + i0 * CT, pa2 + i0 * CT, cptr(P.tbuf[3c-2]), cptr(P.tbuf[3c-1]), P.k1, n2 + row_pad(T), n2, Bc)
        end
      end
    parallel_phases(nt, nb, f1, nb2, f2)
  end
  nothing
end

# n rows of B complex values from a buffer (row stride B) to y (row stride Sy): non-temporal 64-byte
# stores where every row of the block starts on a 64-byte boundary, then a fence (they are weakly
# ordered); otherwise ordinary copies. Non-temporal stores skip reading each line of y before it is
# overwritten, for outputs far beyond what the caches hold (`TUNING[].fourstep_nt_bytes`).
function rows_out!(py::Ptr{T}, pw::Ptr{T}, Sy::Int, B::Int, n::Int) where {T}
  CT = 2 * sizeof(T)
  if (B * CT) % 64 == 0 && (Sy * CT) % 64 == 0 && UInt(py) % 64 == 0
    V = Vec{64 ÷ sizeof(T),T}
    step = 64 ÷ CT
    for j in 0:n-1
      src = pw + j * B * CT; dst = py + j * Sy * CT
      for v in 0:step:B-1
        vstorent(vload(V, src + v * CT), dst + v * CT)
      end
    end
    Threads.atomic_fence()
  else
    lane_copy!(py, pw, Sy, B, B, n)
  end
  nothing
end

"""
    BluesteinMT{T}

Bluestein's algorithm for one long contiguous line of a length with a prime factor above
`MAX_RADIX_PRIME`, on several threads: the two convolution transforms of length M run as
four-step transforms and the three element-wise passes (chirp product with zero padding,
filter product, chirp product) are split over the threads.
"""
struct BluesteinMT{T,F1,F2} <: Kernel{T}
  n::Int
  chirp::Vector{Complex{T}}       # c_j, j = 0…n-1
  kernel::Vector{Complex{T}}      # DFT of the convolution kernel, scaled by 1/M
  fwd::F1
  bwd::F2
  buf::Vector{Complex{T}}         # length M
end

# Smallest 5-smooth M ≥ 2n - 1 that the four-step algorithm can split.
function bluestein_mt_length(::Type{T}, n::Int) where {T}
  m = bluestein_length(T, 2n - 1)
  while fourstep_split(m) == 0
    m = bluestein_length(T, m + 1)
  end
  m
end

function BluesteinMT{T}(n::Int, σ::Int, nt::Int) where {T}
  M = bluestein_mt_length(T, n)
  ser = RChain{T}(M, -1)
  w1 = Vector{Complex{T}}(undef, M); w2 = Vector{Complex{T}}(undef, M)
  chirp, b = cached(() -> bluestein_tables(T, n, σ, ser, w1, w2), CacheKey(:bluestein, T, n, σ, Int[])
                   )::Tuple{Vector{Complex{T}},Vector{Complex{T}}}
  fwd = FourStep{T}(M, -1, nt); bwd = FourStep{T}(M, 1, nt)
  BluesteinMT{T,typeof(fwd),typeof(bwd)}(n, chirp, b, fwd, bwd, Vector{Complex{T}}(undef, M))
end

"""
    CompositeMT{T}

The composite kernel (first stage of length s, Bluestein of length q) for one long contiguous
line on several threads: the first stage in lane blocks over the tasks, then the s Bluestein
transforms spread over the tasks. Kernels with buffers of their own exist once per task. The
blocks and the arithmetic are those of the serial kernel, so the results agree bitwise.
"""
struct CompositeMT{T,C,B} <: Kernel{T}
  n::Int
  s::Int
  q::Int
  chains::Vector{C}                 # first stage, one per task
  blues::Vector{B}                  # one per task
  tw::Vector{Complex{T}}            # w_n^m, m = 0…n-1
  z::Vector{Complex{T}}             # output of the first stage
  tbuf::Vector{Vector{Complex{T}}}  # two per task, one lane block of the first stage each
end

function CompositeMT{T}(n::Int, σ::Int, nt::Int) where {T}
  s, q = composite_split(T, n)
  c1 = first_kernel(T, s, σ, q, 1)
  chains = [c1; [c1 isa RChain ? c1 : first_kernel(T, s, σ, q, 1) for _ in 2:nt]]
  b1 = Bluestein{T}(q, σ)
  blues = [b1; [Bluestein{T}(q, σ) for _ in 2:nt]]
  tw = cached(() -> Complex{T}.(unit_roots(σ, n)), CacheKey(:composite, T, n, σ, Int[]))::Vector{Complex{T}}
  CompositeMT{T,typeof(c1),typeof(b1)}(n, s, q, chains, blues, tw, Vector{Complex{T}}(undef, n),
                                        thread_buffers(T, composite_block(q, s) * s, 2nt))
end

# A long line n = s·q (see composite_split) with at least one Bluestein transform per thread.
function use_composite_mt(::Type{T}, n, nt) where {T}
  (nt > 1 && n >= TUNING[].bluestein_mt_min) || return false
  s, q = composite_split(T, n)
  q > 1 && s >= nt
end
composite_mt_dim(::Type{T}, sz, d::Int, nt::Int) where {T} =
  (so = dim_extent(sz, d); so[1] == 1 && so[2] == 1 && use_composite_mt(T, sz[d], nt))

function run_composite_mt!(py::Ptr{T}, px::Ptr{T}, P::CompositeMT{T}, nt::Int) where {T}
  n, s, q = P.n, P.s, P.q
  CT = 2 * sizeof(T)
  Bb = composite_block(q, s)
  nb = cld(q, Bb)
  ntasks = min(nt, length(P.blues))
  GC.@preserve P begin
    pz = cptr(P.z)
    parallel_phases(ntasks,
      nb, (c, b) -> begin
        local a = b * Bb
        run_lanes!(pz + a * CT, px + a * CT, cptr(P.tbuf[2c-1]), cptr(P.tbuf[2c]), P.chains[c], q, q, min(Bb, q - a))
      end,
      s, (c, k1) -> twiddled_bluestein!(py, pz, P.tw, P.blues[c], n, s, 1, 0, k1, 1))
  end
  nothing
end

"""
    LinesMT{T}

Many lines of a length with a prime factor above `MAX_RADIX_PRIME` on several threads: the kernels
(Bluestein or composite) hold buffers of their own, so each task has one, and the tasks take blocks
of lanes of every slice from a shared counter.
"""
struct LinesMT{T,K} <: Kernel{T}
  n::Int
  B::Int                            # lanes per block
  kernels::Vector{K}                # one per task
  tbuf::Vector{Vector{Complex{T}}}  # two per task, for the first stage of a composite kernel
end

function LinesMT{T}(n::Int, σ::Int, S::Int, nt::Int) where {T}
  # min: a composite kernel holds buffers for B lanes, so B must not exceed the S lanes there are
  B = min(S, lane_block(S, n, nt))
  k1 = kernel(T, n, σ, B)
  LinesMT{T,typeof(k1)}(n, B, [k1; [kernel(T, n, σ, B) for _ in 2:nt]], thread_buffers(T, k1 isa Composite ? B * n : 0, 2nt))
end

lines_mt_dim(sz, d::Int, nt::Int) =
  !smooth(sz[d]) && (so = dim_extent(sz, d); so[1] * so[2] > 1 && threaded(so[1], sz[d], so[2], nt))

# From this length on, a Bluestein line runs on several threads.

use_bluestein_mt(n, nt) = nt > 1 && n >= TUNING[].bluestein_mt_min && !smooth(n)

bluestein_mt_dim(sz, d::Int, nt::Int) = (so = dim_extent(sz, d); so[1] == 1 && so[2] == 1 && use_bluestein_mt(sz[d], nt))

# f(i) for i in chunk `it` of `nchunks` chunks of 0…N-1, the chunks over nt tasks.
@inline function chunked(f::F, N, nt) where {F}
  nch = min(BLOCKS_PER_THREAD * nt, cld(N, 1 << 12))
  parallel_items(nch, nt) do _, it
    for i in chunk_range(it + 1, nch, N)
      f(i)
    end
  end
end

function run_bluestein_mt!(py::Ptr{T}, px::Ptr{T}, P::BluesteinMT{T}, nt::Int) where {T}
  n = P.n; M = length(P.buf)
  GC.@preserve P begin
    cx = Ptr{Complex{T}}(px); cy = Ptr{Complex{T}}(py)
    pb = Ptr{Complex{T}}(cptr(P.buf))
    pc = Ptr{Complex{T}}(cptr(P.chirp)); pk = Ptr{Complex{T}}(cptr(P.kernel))
    chunked(M, nt) do j
      unsafe_store!(pb, j < n ? unsafe_load(cx, j + 1) * unsafe_load(pc, j + 1) : zero(Complex{T}), j + 1)
    end
    run_fourstep!(cptr(P.buf), cptr(P.buf), P.fwd, nt)
    chunked(M, nt) do j
      unsafe_store!(pb, unsafe_load(pb, j + 1) * unsafe_load(pk, j + 1), j + 1)
    end
    run_fourstep!(cptr(P.buf), cptr(P.buf), P.bwd, nt)
    chunked(n, nt) do k
      unsafe_store!(cy, unsafe_load(pb, k + 1) * unsafe_load(pc, k + 1), k + 1)
    end
  end
  nothing
end

# A[k2 + (n2 + row_pad(T))·(i0 + i)] = w[i + Bc·k2] · tw[k2 + n2·(i0 + i)] for i < Bc, k2 < n2:
# contiguous writes along k2, the transposition in registers. Blocks of C lanes × C positions (C
# complex values per vector of the widest type): C vectors of the cache buffer (one position, C
# lanes each) are transposed (`ctranspose`) into one vector per lane, multiplied by the lane's C
# twiddles and stored into the lane's row of A. Lanes and positions beyond multiples of C take the
# loop over single complex values.
function twiddle_transpose!(pa::Ptr{Complex{T}}, pw::Ptr{Complex{T}}, ptw::Ptr{Complex{T}}, i0, Bc, n2) where {T}
  tt_full!(pa, pw, ptw, i0, Bc, n2, Vec{REG_BYTES ÷ sizeof(T),T})
end

# The blocks of C lanes, unrolled per vector type (plain methods, see `tt_methods`).
function tt_full_body(N)
  C = N ÷ 2
  rs = [Symbol(:r, r) for r in 1:C]; cs = [Symbol(:c, l) for l in 1:C]
  loads = [:($(rs[r]) = vload(V, Ptr{T}(pw + (ig + Bc * (k + $(r - 1))) * CT))) for r in 1:C]
  stores = [quote
              j1 = i0 + ig + $(l - 1)
              tt_store!(ccmul($(cs[l]), vload(V, Ptr{T}(ptw + (n2 * j1 + k) * CT))), Ptr{T}(pa + ((n2 + row_pad(T)) * j1 + k) * CT), nts)
            end for l in 1:C]
  quote
    for ig in 0:$C:Bv-1, k in 0:$C:nv-1
      $(loads...)
      ($(cs...),) = ctranspose(($(rs...),))
      $(stores...)
    end
  end
end

# lanes ilo … ihi-1, positions klo … n2-1, one complex value per vector
function tt_full_single!(pa::Ptr{Complex{T}}, pw::Ptr{Complex{T}}, ptw::Ptr{Complex{T}}, i0, Bc, n2, ilo, ihi, klo) where {T}
  V = single_vec(T)
  CT = sizeof(Complex{T})
  for i in ilo:ihi-1
    rowa = (n2 + row_pad(T)) * (i0 + i)
    rowt = n2 * (i0 + i)
    for k2 in klo:n2-1
      a = vload(V, Ptr{T}(pw + (i + Bc * k2) * CT))
      w = vload(V, Ptr{T}(ptw + (rowt + k2) * CT))
      vstore(ccmul(a, w), Ptr{T}(pa + (rowa + k2) * CT))
    end
  end
end

# The same with factor tables: W_n^(j1 k2), j1 = i0 + i, is tw[a] · tw[n1 + b] with
# j1 k2 = a n2 + b. In the blocks of C positions k … k + C - 1 a lane's twiddles are
# W_n^(j1 k) · W_n^(j1 r), r < C: one anchor from the tables per lane and block (a and b advance
# by j1·C) times the lane's vector of W_n^(j1 r), built once from the tables. Each twiddle is then
# a product of two products of table values instead of one.
function twiddle_transpose!(pa::Ptr{Complex{T}}, pw::Ptr{Complex{T}}, ptw::Ptr{Complex{T}}, i0, Bc, n1, n2) where {T}
  tt_factor!(pa, pw, ptw, i0, Bc, n1, n2, Vec{REG_BYTES ÷ sizeof(T),T})
end

# W_n^e from the factor tables at ct (W_n^(a n2), a < n1) and cb (W_n^b, b < n2)
@inline tt_root(ct::Ptr{Complex{T}}, cb::Ptr{Complex{T}}, e, n2) where {T} =
  (q = divrem(e, n2); unsafe_load(ct, q[1] + 1) * unsafe_load(cb, q[2] + 1))
# the vector of W_n^(j1 r), r < C
@inline tt_steps(::Type{Vec{N,T}}, ct, cb, j1, n2) where {N,T} =
  Vec{N,T}(ntuple(i -> (z = tt_root(ct, cb, j1 * ((i - 1) >> 1), n2); isodd(i) ? real(z) : imag(z)), Val(N)))

function tt_factor_body(N)
  C = N ÷ 2
  rs = [Symbol(:r, r) for r in 1:C]; cs = [Symbol(:c, l) for l in 1:C]
  js = [Symbol(:j, l) for l in 1:C]; ss = [Symbol(:s, l) for l in 1:C]
  as = [Symbol(:a, l) for l in 1:C]; bs = [Symbol(:b, l) for l in 1:C]
  qs = [Symbol(:q, l) for l in 1:C]; ds = [Symbol(:d, l) for l in 1:C]
  init = Any[]
  for l in 1:C
    push!(init, :($(js[l]) = i0 + ig + $(l - 1)), :($(ss[l]) = tt_steps(V, ct, cb, $(js[l]), n2)),
          :(($(qs[l]), $(ds[l])) = divrem($(js[l]) * $C, n2)), :($(as[l]) = 0), :($(bs[l]) = 0))
  end
  loads = [:($(rs[r]) = vload(V, Ptr{T}(pw + (ig + Bc * (k + $(r - 1))) * CT))) for r in 1:C]
  stores = Any[]
  for l in 1:C
    push!(stores, quote
      z = unsafe_load(ct, $(as[l]) + 1) * unsafe_load(cb, $(bs[l]) + 1)
      tt_store!(ccmul($(cs[l]), ccmul($(ss[l]), cconst(V, real(z), imag(z)))), Ptr{T}(pa + ((n2 + row_pad(T)) * $(js[l]) + k) * CT), nts)
      $(as[l]) += $(qs[l]); $(bs[l]) += $(ds[l])
      if $(bs[l]) >= n2
        $(bs[l]) -= n2; $(as[l]) += 1
      end
    end)
  end
  quote
    for ig in 0:$C:Bv-1
      $(init...)
      for k in 0:$C:nv-1
        $(loads...)
        ($(cs...),) = ctranspose(($(rs...),))
        $(stores...)
      end
    end
  end
end

# A is written with non-temporal stores where every store is one whole cache line: 64-byte vectors
# and rows of A that start on a line (A aligned, row length a multiple of 64 bytes; k advances by
# whole vectors). A is read in step 2 by other threads than the ones that wrote it, so with ordinary
# stores a line of A sits modified in the writer's cache and every write first takes ownership of a
# line another core holds; whole-line non-temporal stores need no ownership. 16 threads, EPYC 9354,
# time with non-temporal stores over the time with ordinary ones: Float32 2^13 0.90, 2^16 0.83,
# 2^22 0.67, 6^10 0.89; Float64 2^16 0.81, 10^6 0.54, 2^22 0.73, 2^26 0.96 (4 threads: 0.56–0.85).
# With narrower vectors a store is part of a line, and the stores stay ordinary (not measured).
tt_nontemporal(::Type{Vec{N,T}}, pa, n2) where {N,T} =
  N * sizeof(T) == 64 && UInt(pa) % 64 == 0 && ((n2 + row_pad(T)) * 2 * sizeof(T)) % 64 == 0
@inline tt_store!(v, q, nts::Bool) = nts ? vstorent(v, q) : vstore(v, q)

# Plain methods for the vector types of the kernels (C = 1, 2, 4, 8 complex values).
for T in (Float32, Float64), N in (2, 4, 8, 16)
  N * sizeof(T) <= 64 || continue
  @eval function tt_full!(pa::Ptr{Complex{$T}}, pw::Ptr{Complex{$T}}, ptw::Ptr{Complex{$T}}, i0, Bc, n2, ::Type{Vec{$N,$T}})
    T = $T; V = Vec{$N,$T}; CT = 2 * sizeof(T)
    Bv = Bc - Bc % $(N ÷ 2); nv = n2 - n2 % $(N ÷ 2)
    nts = tt_nontemporal(V, pa, n2)
    $(tt_full_body(N))
    nts && Threads.atomic_fence()
    tt_full_single!(pa, pw, ptw, i0, Bc, n2, 0, Bv, nv)
    tt_full_single!(pa, pw, ptw, i0, Bc, n2, Bv, Bc, 0)
  end
  @eval function tt_factor!(pa::Ptr{Complex{$T}}, pw::Ptr{Complex{$T}}, ptw::Ptr{Complex{$T}}, i0, Bc, n1, n2, ::Type{Vec{$N,$T}})
    T = $T; V = Vec{$N,$T}; CT = 2 * sizeof(T)
    ct = ptw; cb = ptw + n1 * CT
    Bv = Bc - Bc % $(N ÷ 2); nv = n2 - n2 % $(N ÷ 2)
    nts = tt_nontemporal(V, pa, n2)
    $(tt_factor_body(N))
    nts && Threads.atomic_fence()
    tt_factor_single!(pa, pw, ptw, i0, Bc, n1, n2, 0, Bv, nv)
    tt_factor_single!(pa, pw, ptw, i0, Bc, n1, n2, Bv, Bc, 0)
  end
end

# lanes ilo … ihi-1, positions klo … n2-1, one complex value per vector
function tt_factor_single!(pa::Ptr{Complex{T}}, pw::Ptr{Complex{T}}, ptw::Ptr{Complex{T}}, i0, Bc, n1, n2, ilo, ihi, klo) where {T}
  V = single_vec(T)
  CT = sizeof(Complex{T})
  pb = ptw + n1 * CT
  for i in ilo:ihi-1
    j1 = i0 + i
    rowa = (n2 + row_pad(T)) * j1
    qa, rb = divrem(j1, n2)
    ta, tb = divrem(j1 * klo, n2)
    for k2 in klo:n2-1
      x = vload(V, Ptr{T}(pw + (i + Bc * k2) * CT))
      w = ccmul(vload(V, Ptr{T}(ptw + ta * CT)), vload(V, Ptr{T}(pb + tb * CT)))
      vstore(ccmul(x, w), Ptr{T}(pa + (rowa + k2) * CT))
      ta += qa; tb += rb
      if tb >= n2
        tb -= n2; ta += 1
      end
    end
  end
end

# Passes along dimensions 1 and 2 fused per plane: when a plane (n1 × n2 values) fits in cache,
# each plane gets both transforms while it is cached, so the two dimensions cost one memory
# round trip. Planes are the work items; with several threads there must be enough of them.
# Complex values per plane (2 MB ComplexF64). Not the L2 per core: on Zen 3 (512 KB L2) planes of
# 2^15 values are up to 1.37× slower than 2^17 with 16 threads (256³), the planes live in L3.

fuse_planes(n1, n2, outer, nt) = n1 * n2 <= TUNING[].plane_budget && (nt == 1 || outer >= nt)

function run_planes!(py::Ptr{T}, px::Ptr{T}, p, k1::RChain{T}, k2::RChain{T},
                     n1::Int, n2::Int, outer::Int) where {T}
  CT = 2 * sizeof(T)
  step = n1 * n2 * CT
  ntasks = threaded(n1, n2, outer, p.nthreads) ? p.nthreads : 1
  parallel_items(outer, ntasks) do c, o
    local pw1 = cptr(p.tbuf[2c-1]); local pw2 = cptr(p.tbuf[2c])
    local base = o * step
    run_rows!(py + base, px + base, pw1, pw2, k1, n2, n1)
    # dimension 2 of the plane in lane blocks that stay in L2
    local B = lane_block(n1, n2, 1)
    for i0 in 0:B:n1-1
      run_lanes!(py + base + i0 * CT, py + base + i0 * CT, pw1, pw2, k2, n1, n1, min(B, n1 - i0))
    end
  end
  nothing
end

# One dimension pass by kernel type; the driver (apply_dims! in interface.jl) calls run_dim!.
function run_dim!(py::Ptr{T}, px::Ptr{T}, p, k::Kernel{T}, S::Int, n::Int, outer::Int) where {T}
  run_kernel_dim!(py, px, p, k, S, n, outer)
end

run_kernel_dim!(py::Ptr{T}, px::Ptr{T}, p, k::FourStep{T}, S::Int, n::Int, outer::Int) where {T} =
  run_fourstep!(py, px, k, p.nthreads)

run_kernel_dim!(py::Ptr{T}, px::Ptr{T}, p, k::BluesteinMT{T}, S::Int, n::Int, outer::Int) where {T} =
  run_bluestein_mt!(py, px, k, p.nthreads)

run_kernel_dim!(py::Ptr{T}, px::Ptr{T}, p, k::CompositeMT{T}, S::Int, n::Int, outer::Int) where {T} =
  run_composite_mt!(py, px, k, p.nthreads)

function run_kernel_dim!(py::Ptr{T}, px::Ptr{T}, p, k::LinesMT{T}, S::Int, n::Int, outer::Int) where {T}
  B = k.B
  nb = cld(S, B)
  step = 2 * S * n * sizeof(T)
  GC.@preserve k parallel_items(nb * outer, min(p.nthreads, length(k.kernels))) do c, it
    local o, b; o, b = divrem(it, nb)
    local i0 = b * B
    local off = o * step + i0 * 2 * sizeof(T)
    run_lanes!(py + off, px + off, cptr(k.tbuf[2c-1]), cptr(k.tbuf[2c]), k.kernels[c], S, S, min(B, S - i0))
  end
  nothing
end

# Stockham chains: lane blocks when there are enough lanes, slices in groups when there are enough
# slices, otherwise one slice after the other.
function run_kernel_dim!(py::Ptr{T}, px::Ptr{T}, p, k::RChain{T}, S::Int, n::Int, outer::Int) where {T}
  nt = p.nthreads
  step = 2 * S * n * sizeof(T)
  if use_lanes(S, n, outer, nt)
    # never wider than the buffers made at plan time (the cache sizes may have changed since)
    B = min(lane_block(S, n, nt), fld(length(p.tbuf[1]), n))
    nb = cld(S, B)
    ntasks = threaded(S, n, outer, nt) ? nt : 1
    parallel_items(nb * outer, ntasks) do c, it
      local o, b; o, b = divrem(it, nb)
      local i0 = b * B
      local off = o * step + i0 * 2 * sizeof(T)
      run_lanes!(py + off, px + off, cptr(p.tbuf[2c-1]), cptr(p.tbuf[2c]), k, S, S, min(B, S - i0))
    end
  elseif threaded(S, n, outer, nt) && outer >= nt
    # slices in groups, BLOCKS_PER_THREAD groups per thread
    ng = min(outer, BLOCKS_PER_THREAD * nt)
    parallel_items(ng, nt) do c, g
      for o in chunk_range(g + 1, ng, outer)
        run!(py + o * step, px + o * step, cptr(p.tbuf[2c-1]), cptr(p.tbuf[2c]), k, S)
      end
    end
  else
    run_serial_dim!(py, px, p, k, S, n, outer)
  end
  nothing
end

# Other kernels (Bluestein) and chains that are not threaded: one slice after the other.
run_kernel_dim!(py::Ptr{T}, px::Ptr{T}, p, k::Kernel{T}, S::Int, n::Int, outer::Int) where {T} =
  run_serial_dim!(py, px, p, k, S, n, outer)

function run_serial_dim!(py::Ptr{T}, px::Ptr{T}, p, k::Kernel{T}, S::Int, n::Int, outer::Int) where {T}
  step = 2 * S * n * sizeof(T)
  pw1 = cptr(p.work1); pw2 = cptr(p.work2)
  for o in 0:outer-1
    run!(py + o * step, px + o * step, pw1, pw2, k, S)
  end
  nothing
end
