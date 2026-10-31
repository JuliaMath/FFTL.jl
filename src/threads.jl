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
# Complex values per lane-block buffer (2^15 ComplexF64 = 512 KB; two buffers plus the block in L2).
const BLOCK_BUDGET = 1 << 15

# Lane block width: a multiple of LANE_ALIGN, within the cache budget, and small enough that
# every thread gets a block.
function lane_block(S, n, nt)
  Bc = max(LANE_ALIGN, LANE_ALIGN * fld(BLOCK_BUDGET ÷ n, LANE_ALIGN))
  Bt = LANE_ALIGN * cld(cld(S, nt), LANE_ALIGN)
  min(Bc, Bt)
end

use_lanes(S, n, outer, nt) = S >= 2LANE_ALIGN && (S * n > BLOCK_BUDGET || (nt > 1 && outer < nt))
threaded(S, n, outer, nt) = nt > 1 && S * n * outer >= THREAD_MIN_WORK

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

function parallel_chunks(f, nchunks)
  if nchunks == 1
    f(1)
  else
    @sync for c in 1:nchunks
      Threads.@spawn f(c)
    end
  end
  nothing
end

chunk_range(c, nchunks, n) = (div((c - 1) * n, nchunks)):(div(c * n, nchunks) - 1)

function run_dim!(py::Ptr{T}, px::Ptr{T}, p, k::Kernel{T}, S::Int, n::Int, outer::Int) where {T}
  nt = p.nthreads
  step = 2 * S * n * sizeof(T)
  if k isa FourStep
    run_fourstep!(py, px, k, nt)
  elseif k isa Chain && use_lanes(S, n, outer, nt)
    B = lane_block(S, n, nt)
    nb = cld(S, B)
    nchunks = threaded(S, n, outer, nt) ? min(nt, nb * outer) : 1
    parallel_chunks(nchunks) do c
      # `local`: an assignment to a name of the enclosing function would share it among tasks
      local pw1 = cptr(p.tbuf[2c-1]); local pw2 = cptr(p.tbuf[2c])
      for it in chunk_range(c, nchunks, nb * outer)
        local o, b; o, b = divrem(it, nb)
        local i0 = b * B
        local off = o * step + i0 * 2 * sizeof(T)
        run_lanes!(py + off, px + off, pw1, pw2, k, S, S, min(B, S - i0))
      end
    end
  elseif k isa Chain && threaded(S, n, outer, nt) && outer >= nt
    parallel_chunks(nt) do c
      local pw1 = cptr(p.tbuf[2c-1]); local pw2 = cptr(p.tbuf[2c])
      for o in chunk_range(c, nt, outer)
        run!(py + o * step, px + o * step, pw1, pw2, k, S)
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
  X[k2 + n2 k1] = Σ_j1 W_n^(j1 k2) W_n1^(j1 k1) Σ_j2 x[j1 + n1 j2] W_n2^(j2 k2).
1. DFTs of length n2 over the n1 lanes of x seen as an n1 × n2 matrix (lane blocks in parallel),
2. multiplication by W_n^(j1 k2) and DFTs of length n1 along the columns (columns in parallel),
3. transposition into the output (column blocks in parallel).
"""
struct FourStep{T,K1,K2} <: Kernel{T}
  n::Int
  n1::Int
  n2::Int
  k1::K1                          # length n1, contiguous columns
  k2::K2                          # length n2, n1 interleaved lanes
  tw::Vector{Complex{T}}          # W_n^(j1 k2) at j1 + n1·k2
  buf::Vector{Complex{T}}
  tbuf::Vector{Vector{Complex{T}}}
end

# Split n = n1·n2 with n1 as close to √n as possible, n1 a multiple of LANE_ALIGN.
function fourstep_split(n)
  best = 0
  for d in 1:isqrt(n)
    n % d == 0 || continue
    for n1 in (d, n ÷ d)
      n1 % LANE_ALIGN == 0 && smooth(n1) && smooth(n ÷ n1) || continue
      if best == 0 || abs(log(n1 / sqrt(n))) < abs(log(best / sqrt(n)))
        best = n1
      end
    end
  end
  best
end

function FourStep{T}(n::Int, σ::Int, nt::Int) where {T}
  n1 = fourstep_split(n)
  n2 = n ÷ n1
  k1 = Chain{T}(n1, σ, radices(n1))
  k2 = Chain{T}(n2, σ, radices(n2, n1))
  tw = vec([Complex{T}(cispi(2σ * mod(j1 * k2, n) / n)) for j1 in 0:n1-1, k2 in 0:n2-1])
  nw = max(lane_block(n1, n2, nt) * n2, n1)
  FourStep{T,typeof(k1),typeof(k2)}(n, n1, n2, k1, k2, tw, Vector{Complex{T}}(undef, n),
                                    [Vector{Complex{T}}(undef, nw) for _ in 1:2nt])
end

const FOURSTEP_MIN = 1 << 16

use_fourstep(n, nt) = nt > 1 && n >= FOURSTEP_MIN && smooth(n) && fourstep_split(n) > 0

function run_fourstep!(py::Ptr{T}, px::Ptr{T}, P::FourStep{T}, nt::Int) where {T}
  n1, n2 = P.n1, P.n2
  CT = 2 * sizeof(T)
  GC.@preserve P begin
    pa = cptr(P.buf)
    # 1. length-n2 DFTs over lane blocks of x → buf
    B = lane_block(n1, n2, nt)
    nb = cld(n1, B)
    parallel_chunks(min(nt, nb)) do c
      local pw1 = cptr(P.tbuf[2c-1]); local pw2 = cptr(P.tbuf[2c])
      for b in chunk_range(c, min(nt, nb), nb)
        local i0 = b * B
        run_lanes!(pa + i0 * CT, px + i0 * CT, pw1, pw2, P.k2, n1, n1, min(B, n1 - i0))
      end
    end
    # 2. twiddles and length-n1 DFTs along the columns of buf, in place
    parallel_chunks(nt) do c
      local pw1 = cptr(P.tbuf[2c-1]); local pw2 = cptr(P.tbuf[2c])
      for k2 in chunk_range(c, nt, n2)
        local off = k2 * n1
        @inbounds @simd for j1 in 1:n1
          P.buf[off+j1] *= P.tw[off+j1]
        end
        run!(pa + off * CT, pa + off * CT, pw1, pw2, P.k1, 1)
      end
    end
    # 3. y[k2 + n2 k1] = buf[k1 + n1 k2]
    transpose_blocks!(Ptr{Complex{T}}(py), Ptr{Complex{T}}(pa), n1, n2, nt)
  end
  nothing
end

# y (n2 × n1) = transpose of a (n1 × n2), in tiles of 32 × 32, tile columns of a in parallel
function transpose_blocks!(py::Ptr{C}, pa::Ptr{C}, n1::Int, n2::Int, nt::Int) where {C}
  tile = 32
  ntc = cld(n2, tile)
  parallel_chunks(min(nt, ntc)) do c
    for tc in chunk_range(c, min(nt, ntc), ntc)
      k2lo = tc * tile; k2hi = min(k2lo + tile, n2) - 1
      for k1lo in 0:tile:n1-1
        k1hi = min(k1lo + tile, n1) - 1
        for k1 in k1lo:k1hi, k2 in k2lo:k2hi
          unsafe_store!(py, unsafe_load(pa, k1 + n1 * k2 + 1), k2 + n2 * k1 + 1)
        end
      end
    end
  end
end
