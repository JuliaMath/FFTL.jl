# Execution of one dimension pass with S lanes, length n and `outer` slices, on one or several
# threads.
#
# With enough lanes the pass runs in lane blocks (`run_lanes!`): a block of B lanes is read from
# the array into a compact buffer, all stages run there, and the last stage writes back. B is
# chosen so that the block and its buffers stay in the L2 cache, so a pass over a large array
# costs one read and one write of memory instead of one per stage; the blocks are distributed
# over the threads. Passes with few lanes (S small, e.g. the first dimension) run whole slices,
# on several threads when there are enough slices.

# Below this many complex values per pass, starting tasks costs more than it gains.
const THREAD_MIN_WORK = 1 << 15
# Lane blocks are multiples of 8 complex values (a 128-byte cache line of ComplexF64).
const LANE_ALIGN = 8
# Complex values per lane block buffer (a task holds two or three): 2^16 measured best on the
# M1 (3 MB of L2 per core) and on Zen 3 with one thread, which then has the 32 MB L3 to itself.
const BLOCK_BUDGET = 1 << 16
# With several threads, long lines (n >= 1024) take blocks of half the L2 per core in ComplexF64
# values: 2^14 on Zen 3 (512 KB L2 per core, shared L3; 16 threads: 2^22 Float64 1.6× and 2048²
# Float64 1.3× faster than with 2^16), 2^16 on the M1. One thread keeps the large blocks (Zen 3
# 2048² 1.6× slower with the small ones), as do short lines.
block_budget(n, nt) = nt > 1 && n >= 1024 ?
  clamp(prevpow2(CACHES[].l2 ÷ 32), 1 << 12, BLOCK_BUDGET) : BLOCK_BUDGET

# Blocks per thread: threads take blocks from a shared counter, so a slow core (or one busy
# with something else) delays the pass by one block, not by a fixed share.
const BLOCKS_PER_THREAD = 4

# Lane block width: a multiple of LANE_ALIGN, within the cache budget, and small enough that
# there are BLOCKS_PER_THREAD blocks per thread.
function lane_block(S, n, nt)
  Bc = max(LANE_ALIGN, LANE_ALIGN * fld(block_budget(n, nt) ÷ n, LANE_ALIGN))
  Bt = LANE_ALIGN * cld(cld(S, BLOCKS_PER_THREAD * nt), LANE_ALIGN)
  min(Bc, Bt)
end

use_lanes(S, n, outer, nt) = S >= 2LANE_ALIGN && (S * n > BLOCK_BUDGET || (nt > 1 && outer < nt))
threaded(S, n, outer, nt) = nt > 1 && S * n * outer >= THREAD_MIN_WORK

# k work buffers of n complex values (one or more per task)
thread_buffers(::Type{T}, n::Int, k::Int) where {T} = Vector{Complex{T}}[Vector{Complex{T}}(undef, n) for _ in 1:k]

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
fourstep_dim(sz, d::Int, nt::Int) = (so = dim_extent(sz, d); so[1] == 1 && so[2] == 1 && use_fourstep(sz[d], nt))

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

function run_dim!(py::Ptr{T}, px::Ptr{T}, p, k::Kernel{T}, S::Int, n::Int, outer::Int) where {T}
  nt = p.nthreads
  step = 2 * S * n * sizeof(T)
  if k isa FourStep
    run_fourstep!(py, px, k, nt)
  elseif (k isa Chain || k isa RChain) && use_lanes(S, n, outer, nt)
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
  elseif (k isa Chain || k isa RChain) && threaded(S, n, outer, nt) && outer >= nt
    # slices in groups, BLOCKS_PER_THREAD groups per thread
    ng = min(outer, BLOCKS_PER_THREAD * nt)
    parallel_items(ng, nt) do c, g
      for o in chunk_range(g + 1, ng, outer)
        run!(py + o * step, px + o * step, cptr(p.tbuf[2c-1]), cptr(p.tbuf[2c]), k, S)
      end
    end
  else
    pw1 = cptr(p.work1); pw2 = cptr(p.work2)
    for o in 0:outer-1
      run!(py + o * step, px + o * step, pw1, pw2, k, S)
    end
  end
  nothing
end

"""
    FourStep{T}

One-dimensional DFT of length n = n1·n2 for several threads. With j = j1 + n1·j2 and
k = k2 + n2·k1:
  X[k2 + n2 k1] = Σ_j1 W_n1^(j1 k1) · [W_n^(j1 k2) Σ_j2 x[j1 + n1 j2] W_n2^(j2 k2)].
1. For blocks of j1 (lanes of x seen as an n1 × n2 matrix): DFTs of length n2 in a cache buffer,
   multiplication by W_n^(j1 k2), written transposed: A[k2 + n2·j1].
2. DFTs of length n1 over the n2 lanes k2 of A, written to the output in natural order.
Each step reads and writes memory once; the transposition happens in cache. Rows of A are
padded to n2 + ROW_PAD so that step 2 does not read with a power-of-two stride (cache-set
conflicts).
"""
struct FourStep{T,K1,K2} <: Kernel{T}
  n::Int
  n1::Int
  n2::Int
  k1::K1                          # length n1, n2 interleaved lanes
  k2::K2                          # length n2, n1 interleaved lanes
  tw::Vector{Complex{T}}          # W_n^(j1 k2) at k2 + n2 j1 (length n), or factor tables (n1 + n2)
  buf::Vector{Complex{T}}         # A, rows of n2 + ROW_PAD
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

# C: the chain type of the two factors (Chain, or RChain for run-time plans)
function FourStep{T}(n::Int, σ::Int, nt::Int, ::Type{C} = Chain) where {T,C}
  n1 = fourstep_split(n)
  n2 = n ÷ n1
  k1 = C{T}(n1, σ, radices(n1, n2; esize = sizeof(Complex{T})))
  k2 = C{T}(n2, σ, radices(n2, n1; esize = sizeof(Complex{T})))
  # Above FACTOR_TWIDDLES bytes the table of W_n^(j1 k2) would be as large as the data and cost a
  # memory round trip: W_n^(j1 k2) = W_n1^a · W_n^b with j1 k2 = a n2 + b, from tables of n1 + n2
  # values (2^22 Float64 on 8 threads: 5 % faster on the M1). Cache-resident sizes keep the full
  # table: the extra product costs more there (65536 Float32: 4× slower with factor tables).
  tw = cached(() -> fourstep_twiddles(T, n, σ, n1), CacheKey(:fourstep, T, n, σ, [n1]))::Vector{Complex{T}}
  nw = max(lane_block(n1, n2, nt) * n2, lane_block(n2, n1, nt) * n1)
  FourStep{T,typeof(k1),typeof(k2)}(n, n1, n2, k1, k2, tw, Vector{Complex{T}}(undef, n1 * (n2 + ROW_PAD)),
                                    thread_buffers(T, nw, 3nt))
end

function fourstep_twiddles(::Type{T}, n::Int, σ::Int, n1::Int) where {T}
  n2 = n ÷ n1
  W = unit_roots(σ, n)
  if n * sizeof(Complex{T}) > FACTOR_TWIDDLES
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

const FOURSTEP_MIN = 1 << 13
const ROW_PAD = 4
# Bytes above which the four-step uses factor twiddle tables (measured on the M1; on Zen 3 a
# threshold at its 32 MB L3 instead measures the same within noise, 0.93–1.07).
const FACTOR_TWIDDLES = 1 << 23

use_fourstep(n, nt) = nt > 1 && n >= FOURSTEP_MIN && smooth(n) && fourstep_split(n) > 0

function run_fourstep!(py::Ptr{T}, px::Ptr{T}, P::FourStep{T}, nt::Int) where {T}
  n1, n2 = P.n1, P.n2
  CT = 2 * sizeof(T)
  GC.@preserve P begin
    pa = Ptr{Complex{T}}(cptr(P.buf))
    ptw = Ptr{Complex{T}}(cptr(P.tw))
    # 1. blocks of j1: DFTs of length n2 into a cache buffer, twiddles, transposed into A
    B = min(lane_block(n1, n2, nt), fld(length(P.tbuf[1]), n2))
    nb = cld(n1, B)
    parallel_items(nb, nt) do c, b
      local pw3 = cptr(P.tbuf[3c])
      local cw3 = Ptr{Complex{T}}(pw3)
      local i0 = b * B
      local Bc = min(B, n1 - i0)
      run_lanes!(pw3, px + i0 * CT, cptr(P.tbuf[3c-2]), cptr(P.tbuf[3c-1]), P.k2, n1, Bc, Bc)
      if length(P.tw) == P.n
        twiddle_transpose!(pa, cw3, ptw, i0, Bc, n2)
      else
        twiddle_transpose!(pa, cw3, ptw, i0, Bc, n1, n2)
      end
    end
    # 2. DFTs of length n1 over lane blocks of k2: A[k2 + n2 j1] → y[k2 + n2 k1]
    B2 = min(lane_block(n2, n1, nt), fld(length(P.tbuf[1]), n1))
    nb2 = cld(n2, B2)
    pa2 = cptr(P.buf)
    parallel_items(nb2, nt) do c, b
      local i0 = b * B2
      run_lanes!(py + i0 * CT, pa2 + i0 * CT, cptr(P.tbuf[3c-2]), cptr(P.tbuf[3c-1]), P.k1, n2 + ROW_PAD, n2, min(B2, n2 - i0))
    end
  end
  nothing
end

# A[k2 + (n2 + ROW_PAD)·(i0 + i)] = w[i + Bc·k2] · tw[k2 + n2·(i0 + i)] for i < Bc, k2 < n2:
# contiguous writes along k2, strided reads from the cache buffer, one complex value per vector.
function twiddle_transpose!(pa::Ptr{Complex{T}}, pw::Ptr{Complex{T}}, ptw::Ptr{Complex{T}}, i0, Bc, n2) where {T}
  V = single_vec(T)
  CT = sizeof(Complex{T})
  for i in 0:Bc-1
    rowa = (n2 + ROW_PAD) * (i0 + i)
    rowt = n2 * (i0 + i)
    for k2 in 0:n2-1
      a = vload(V, Ptr{T}(pw + (i + Bc * k2) * CT))
      w = vload(V, Ptr{T}(ptw + (rowt + k2) * CT))
      vstore(ccmul(a, w), Ptr{T}(pa + (rowa + k2) * CT))
    end
  end
end

# The same with factor tables: W_n^(j1 k2), j1 = i0 + i, is tw[a] · tw[n1 + b] with
# j1 k2 = a n2 + b; a and b advance by j1 per step.
function twiddle_transpose!(pa::Ptr{Complex{T}}, pw::Ptr{Complex{T}}, ptw::Ptr{Complex{T}}, i0, Bc, n1, n2) where {T}
  V = single_vec(T)
  CT = sizeof(Complex{T})
  pb = ptw + n1 * CT
  for i in 0:Bc-1
    j1 = i0 + i
    rowa = (n2 + ROW_PAD) * j1
    qa, rb = divrem(j1, n2)
    ta = 0; tb = 0
    for k2 in 0:n2-1
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
const PLANE_BUDGET = 1 << 17

fuse_planes(n1, n2, outer, nt) = n1 * n2 <= PLANE_BUDGET && (nt == 1 || outer >= nt)

function run_planes!(py::Ptr{T}, px::Ptr{T}, p, k1::Union{Chain{T},RChain{T}}, k2::Union{Chain{T},RChain{T}},
                     n1::Int, n2::Int, outer::Int) where {T}
  CT = 2 * sizeof(T)
  step = n1 * n2 * CT
  ntasks = threaded(n1, n2, outer, p.nthreads) ? p.nthreads : 1
  parallel_items(outer, ntasks) do c, o
    local pw1 = cptr(p.tbuf[2c-1]); local pw2 = cptr(p.tbuf[2c])
    local base = o * step
    for j in 0:n2-1
      local off = base + j * n1 * CT
      run!(py + off, px + off, pw1, pw2, k1, 1)
    end
    # dimension 2 of the plane in lane blocks that stay in L2
    local B = lane_block(n1, n2, 1)
    for i0 in 0:B:n1-1
      run_lanes!(py + base + i0 * CT, py + base + i0 * CT, pw1, pw2, k2, n1, n1, min(B, n1 - i0))
    end
  end
  nothing
end
