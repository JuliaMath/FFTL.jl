# The FFTW3 C interface (include/fftw3.h) over FFTL's run-time plans: double precision
# as fftw_*, single precision as fftwf_*, both exported by the same library. Signatures and
# semantics follow the FFTW manual (https://www.fftw.org/fftw3_doc/), the constant values
# FFTW.jl (src/fft.jl). One generic implementation per entry point; the @ccallable methods at the
# end instantiate it for Float64 and Float32.
#
# Covered: every layout of the basic, advanced (plan_many) and guru/guru64 planners, interleaved
# and split complex arrays, fftw_execute and the new-array execute functions, plan copies,
# memory, wisdom (none is kept), threads, plan information. Planning returns C NULL with a
# message in fftl_last_error for invalid arguments and for FFTW_WISDOM_ONLY (there is never
# wisdom).
#
# Layouts. When the transform and vector dimensions form one dense array (in some order of
# dimensions, with the same layout for input and output; r2c/c2r: FFTW's real and complex
# layouts of that array, padded in place, with the halved dimension first among the transform
# dimensions), the FFTL plan runs on the user's arrays. Every other layout (gaps, embedded
# subarrays, different input and output layouts, negative strides, the halved dimension not
# the fastest-varying one, split arrays) goes through the copy fallback: gather into a dense
# buffer, run the plan in place there, scatter into the output layout.
#
# Mapping. FFTW arrays are row major (n[0] varies slowest), FFTL's column major; an iodim list is
# turned into one FFTL array by sorting the dimensions by stride (solve_layout), so the basic
# interface ends up with sz = reverse(n) and FFTW's last dimension, the one r2c/c2r halve, as
# FFTL's region[1], the one RRFFTPlan halves.
#
# An fftw_plan is a handle (registry index | precision << 32 | type << 40) plus one, so
# that it is never NULL. Each plan record keeps idle FFTL plans: an execution takes one (or
# builds another when all are busy) and returns it afterwards, so that the new-array execute
# functions can run one plan on several threads at once, as FFTW allows. FFTL plans own their
# work buffers (and the buffer of the copy fallback); the copies share the cached tables.

const W_C2C, W_R2C, W_C2R, W_R2R = 0, 1, 2, 3

# FFTW planner flags and kinds (values from FFTW.jl, src/fft.jl)
const W_WISDOM_ONLY = UInt32(1) << 21
# FFTW r2r kind codes R2HC = 0, HC2R, DHT, REDFT00, REDFT01, REDFT10, REDFT11, RODFT00, RODFT01,
# RODFT10, RODFT11 = 10 → FFTL.R2RKind values
const W_KINDS = Int[9, 10, 8, 0, 2, 1, 3, 4, 6, 5, 7]
for (code, k) in ((0, FFTL.R2HC), (1, FFTL.HC2R), (2, FFTL.DHT), (3, FFTL.REDFT00), (4, FFTL.REDFT01),
                  (5, FFTL.REDFT10), (6, FFTL.REDFT11), (7, FFTL.RODFT00), (8, FFTL.RODFT01),
                  (9, FFTL.RODFT10), (10, FFTL.RODFT11))
  FFTL.R2RKind(W_KINDS[code + 1]) == k || error("FFTW kind table out of sync with FFTL.R2RKind")
end

# fftw_iodim and fftw_iodim64 (ptrdiff_t: 64-bit platforms only)
struct WIodim
  n::Cint
  is::Cint
  os::Cint
end
struct WIodim64
  n::Int
  is::Int
  os::Int
end

# The copy fallback of a layout that is not one dense array: the user's arrays are gathered into
# a dense column-major buffer, the FFTL plan runs in place there, and the result is scattered
# into the output layout. Per buffer dimension: the counts on the input and output side (they
# differ only along the halved dimension of r2c/c2r), the strides in the user's arrays (FFTW's
# units: complex numbers for interleaved arrays, real numbers for real and split arrays), and
# the strides in the buffer in the same units.
struct WCopy
  nin::Vector{Int}
  nout::Vector{Int}
  istr::Vector{Int}
  ostr::Vector{Int}
  bin::Vector{Int}
  bout::Vector{Int}
  buflen::Int                     # real numbers
end
WCopy() = WCopy(Int[], Int[], Int[], Int[], Int[], Int[], 0)

# An idle FFTL plan of a plan record with its own buffer for the copy fallback (empty if the
# record needs none), so that concurrent executions never share a buffer.
struct WSlot{P,T}
  plan::P
  buf::Vector{T}
end

# One FFTW plan. P is the FFTL plan type, T its real type; `free` holds the idle FFTL plans of
# this record. Split arrays: pin2/pout2 are the imaginary parts (C_NULL for interleaved and real
# arrays).
mutable struct WPlan{P,T}
  typ::Int
  inplace::Bool
  split::Bool
  pin::Ptr{Cvoid}
  pin2::Ptr{Cvoid}
  pout::Ptr{Cvoid}
  pout2::Ptr{Cvoid}
  noop::Bool                      # a vector dimension of length 0: nothing to compute
  sz::Vector{Int}
  region::Vector{Int}
  kinds::Vector{FFTL.R2RKind}
  sign::Int
  threads::Int
  copy::Bool
  cp::WCopy
  free::Vector{WSlot{P,T}}
  lock::ReentrantLock
  refs::Int                       # fftw_copy_plan references
end

const WREG_C2C_F64 = Union{Nothing,WPlan{RPlan{Float64},Float64}}[]
const WREG_C2C_F32 = Union{Nothing,WPlan{RPlan{Float32},Float32}}[]
const WREG_RF_F64 = Union{Nothing,WPlan{RRFFTPlan{Float64},Float64}}[]
const WREG_RF_F32 = Union{Nothing,WPlan{RRFFTPlan{Float32},Float32}}[]
const WREG_RR_F64 = Union{Nothing,WPlan{RR2RPlan{Float64},Float64}}[]
const WREG_RR_F32 = Union{Nothing,WPlan{RR2RPlan{Float32},Float32}}[]
wreg_c2c(::Type{Float64}) = WREG_C2C_F64
wreg_c2c(::Type{Float32}) = WREG_C2C_F32
wreg_rf(::Type{Float64}) = WREG_RF_F64
wreg_rf(::Type{Float32}) = WREG_RF_F32
wreg_rr(::Type{Float64}) = WREG_RR_F64
wreg_rr(::Type{Float32}) = WREG_RR_F32
wprec(::Type{Float64}) = 1
wprec(::Type{Float32}) = 2

# fftw_plan_with_nthreads, per precision
const WTHREADS_F64 = Ref(1)
const WTHREADS_F32 = Ref(1)
wthreads(::Type{Float64}) = WTHREADS_F64
wthreads(::Type{Float32}) = WTHREADS_F32

# --- layout: iodims → one dense FFTL array ------------------------------------------------

# A dimension of the iodim lists: length, stride in the real array (or the only stride for c2c
# and r2r), stride in the complex array, and its index among the transform dimensions (0 for a
# vector dimension).
struct WDim
  n::Int
  rs::Int
  cs::Int
  t::Int
end

# True if the dimensions in this order are one dense column-major array for input and output.
# h: position of the halved dimension (r2c/c2r) in `order`, 0 if none. Its real extent is
# 2(n ÷ 2 + 1) in place (FFTW's padding), n out of place; its complex extent n ÷ 2 + 1.
function wdense(order::Vector{WDim}, h::Int, inplace::Bool)
  accr = 1; accc = 1
  for i in eachindex(order)
    d = order[i]
    (d.n == 1 || (d.rs == accr && d.cs == accc)) || return false
    if i == h
      accr *= inplace ? 2 * (d.n ÷ 2 + 1) : d.n
      accc *= d.n ÷ 2 + 1
    else
      accr *= d.n
      accc *= d.n
    end
  end
  true
end

# Sort by the magnitude of the real stride (a handful of dimensions; sort! with `by` does not
# compile ahead of time).
function wsort!(v::Vector{WDim})
  for i in 2:length(v)
    x = v[i]; j = i - 1
    while j >= 1 && abs(v[j].rs) > abs(x.rs)
      v[j + 1] = v[j]; j -= 1
    end
    v[j + 1] = x
  end
  v
end

struct WLayout
  noop::Bool
  sz::Vector{Int}
  region::Vector{Int}
  tidx::Vector{Int}       # per region entry: the FFTW transform dimension (1-based)
  copy::Bool
  cp::WCopy
end

# The FFTL array for transform dimensions td and vector dimensions vd, given as (n, is, os)
# with FFTW's units (r2c: is real, os complex; c2r: is complex, os real). If the dimensions form
# one dense array (and dense_ok), the plan runs on the user's arrays; otherwise on a buffer
# through the copy fallback (wcopy_layout).
function solve_layout(td::Vector{NTuple{3,Int}}, vd::Vector{NTuple{3,Int}}, typ::Int, inplace::Bool,
                      dense_ok::Bool)
  for x in td
    x[1] >= 1 || throw(ArgumentError("FFTW interface: transform sizes must be positive"))
  end
  for x in vd
    x[1] >= 0 || throw(ArgumentError("FFTW interface: vector sizes must be nonnegative"))
    x[1] == 0 && return WLayout(true, Int[], Int[], Int[], false, WCopy())
  end
  real = typ == W_R2C || typ == W_C2R
  rank = length(td)
  hdim = WDim(1, 0, 0, 1)       # rank 0 r2c/c2r: one value, a halved dimension of length 1
  dims = WDim[]
  ones = WDim[]                 # r2r dimensions of length 1 (not the identity: DCT-II scales by 2)
  for i in 1:rank
    n, is, os = td[i]
    rs, cs = typ == W_C2R ? (os, is) : (is, os)
    if real && i == rank
      hdim = WDim(n, rs, cs, i)
    elseif n > 1
      push!(dims, WDim(n, rs, cs, i))
    elseif typ == W_R2R
      push!(ones, WDim(n, rs, cs, i))
    end
  end
  for x in vd
    n, is, os = x
    rs, cs = typ == W_C2R ? (os, is) : (is, os)
    n > 1 && push!(dims, WDim(n, rs, cs, 0))
  end
  wsort!(dims)
  order = WDim[]
  found = false
  if dense_ok && real
    # The halved dimension must come before the other transform dimensions (FFTL halves the
    # first dimension of the region); try each place up to there.
    ft = length(dims) + 1
    for i in eachindex(dims)
      if dims[i].t > 0
        ft = i
        break
      end
    end
    for pos in 1:ft
      cand = WDim[]
      for i in 1:pos-1
        push!(cand, dims[i])
      end
      push!(cand, hdim)
      for i in pos:length(dims)
        push!(cand, dims[i])
      end
      if wdense(cand, pos, inplace)
        order = cand
        found = true
        break
      end
    end
  elseif dense_ok
    found = wdense(dims, 0, inplace)
    order = dims
  end
  if !found
    # Copy fallback: the halved dimension first, the others by input stride.
    order = WDim[]
    real && push!(order, hdim)
    for d in dims
      push!(order, d)
    end
  end
  for d in ones
    push!(order, d)
  end
  sz = Int[]; region = Int[]; tidx = Int[]
  for i in eachindex(order)
    push!(sz, order[i].n)
    if order[i].t > 0
      push!(region, i)
      push!(tidx, order[i].t)
    end
  end
  cp = found ? WCopy() : wcopy_layout(order, typ)
  isempty(sz) && push!(sz, 1)
  WLayout(false, sz, region, tidx, !found, cp)
end

# The gather and scatter of the copy fallback for the buffer dimensions `order`. The buffer is
# the dense column-major array the FFTL plan runs on in place: complex numbers for c2c, real
# numbers for r2r, and for r2c/c2r FFTW's padded in-place layout (the halved dimension first,
# 2(n ÷ 2 + 1) real or n ÷ 2 + 1 complex numbers long).
function wcopy_layout(order::Vector{WDim}, typ::Int)
  real = typ == W_R2C || typ == W_C2R
  k = length(order)
  nr = Int[]; nc = Int[]; sr = Int[]; sc = Int[]; ur = Int[]; uc = Int[]
  accr = 1; accc = 1
  for i in 1:k
    d = order[i]
    nh = real && i == 1 ? d.n ÷ 2 + 1 : d.n
    push!(nr, d.n); push!(nc, nh)
    push!(ur, d.rs); push!(uc, d.cs)
    push!(sr, accr); push!(sc, accc)
    accr *= real && i == 1 ? 2nh : d.n
    accc *= nh
  end
  buflen = typ == W_R2R ? accr : 2accc          # real numbers
  if typ == W_C2R
    WCopy(nc, nr, uc, ur, sc, sr, buflen)
  elseif typ == W_R2C
    WCopy(nr, nc, ur, uc, sr, sc, buflen)
  else
    WCopy(nr, nr, ur, uc, sr, sr, buflen)
  end
end

# Advanced interface → iodims. nembed empty means NULL: the logical sizes, for r2c/c2r with the
# halved complex and (in place) padded real last dimension.
function many_dims(n::Vector{Int}, howmany::Int, inembed::Vector{Int}, istride::Int, idist::Int,
                   onembed::Vector{Int}, ostride::Int, odist::Int, typ::Int, inplace::Bool)
  rank = length(n)
  rank >= 0 || throw(ArgumentError("FFTW interface: rank must be nonnegative"))
  howmany >= 0 || throw(ArgumentError("FFTW interface: howmany must be nonnegative"))
  ie = isempty(inembed) ? default_embed(n, typ, true, inplace) : inembed
  oe = isempty(onembed) ? default_embed(n, typ, false, inplace) : onembed
  td = Vector{NTuple{3,Int}}(undef, rank)
  si = istride; so = ostride
  for i in rank:-1:1
    td[i] = (n[i], si, so)
    si *= ie[i]; so *= oe[i]
  end
  vd = howmany == 1 ? NTuple{3,Int}[] : [(howmany, idist, odist)]
  td, vd
end

function default_embed(n::Vector{Int}, typ::Int, input::Bool, inplace::Bool)
  e = copy(n)
  isempty(e) && return e
  realside = (typ == W_R2C && input) || (typ == W_C2R && !input)
  cplxside = (typ == W_R2C && !input) || (typ == W_C2R && input)
  h = e[end] ÷ 2 + 1
  if cplxside
    e[end] = h
  elseif realside && inplace
    e[end] = 2h
  end
  e
end

# --- plan records ---------------------------------------------------------------------

new_fftl(::Type{RPlan{T}}, r::WPlan) where {T} = RPlan{T}(r.sz, r.region, r.sign; threads = r.threads)
new_fftl(::Type{RRFFTPlan{T}}, r::WPlan) where {T} =
  RRFFTPlan{T}(r.sz, r.region, r.typ == W_R2C ? -1 : 1; threads = r.threads)
new_fftl(::Type{RR2RPlan{T}}, r::WPlan) where {T} = RR2RPlan{T}(r.sz, r.kinds, r.region; threads = r.threads)

new_slot(r::WPlan{P,T}) where {P,T} = WSlot{P,T}(new_fftl(P, r), Vector{T}(undef, r.copy ? r.cp.buflen : 0))

function wrecord(::Type{P}, ::Type{T}, typ::Int, lay::WLayout, kinds::Vector{FFTL.R2RKind}, sign::Int,
                 threads::Int, pin::Ptr{Cvoid}, pin2::Ptr{Cvoid}, pout::Ptr{Cvoid}, pout2::Ptr{Cvoid},
                 split::Bool) where {P,T}
  r = WPlan{P,T}(typ, pin == pout, split, pin, pin2, pout, pout2, lay.noop, lay.sz, lay.region, kinds,
                 sign, threads, lay.copy, lay.cp, WSlot{P,T}[], ReentrantLock(), 1)
  lay.noop || push!(r.free, new_slot(r))
  r
end

function wregister!(reg::Vector{Union{Nothing,WPlan{P,T}}}, r::WPlan{P,T}, typ::Int, prec::Int) where {P,T}
  idx = @lock REGISTRY_LOCK begin
    push!(reg, r)
    length(reg)
  end
  h = Int64(idx - 1) | (Int64(prec) << 32) | (Int64(typ) << 40)
  Ptr{Cvoid}(UInt(h + 1))
end

# (type, precision, index) of an fftw_plan; type -1 if it is none
function wdecode(p::Ptr{Cvoid})
  p == C_NULL && return (-1, 0, 0)
  h = reinterpret(Int64, UInt64(UInt(p))) - 1      # any pointer: no InexactError
  h < 0 && return (-1, 0, 0)
  typ = Int(h >> 40); prec = Int((h >> 32) & 0xff)
  (0 <= typ <= W_R2R && (prec == 1 || prec == 2)) || return (-1, 0, 0)
  (typ, prec, Int(h & 0xffffffff))
end

wlookup(reg::Vector{Union{Nothing,WPlan{P,T}}}, idx::Int) where {P,T} =
  @lock REGISTRY_LOCK (0 <= idx <= length(reg) - 1 ? reg[idx + 1] : nothing)

function read_kinds_fftw(p::Ptr{Cint}, n::Int)
  v = Int[]
  for i in 1:n
    k = Int(unsafe_load(p, i))
    0 <= k <= 10 || throw(ArgumentError("FFTW interface: unknown r2r kind"))
    push!(v, k)
  end
  v
end

# The common planner: iodims (FFTW order), kinds per transform dimension (r2r). Split arrays
# (split = true): pin2/pout2 are the imaginary parts of the complex side; they always go through
# the copy fallback.
function wplan(::Type{T}, typ::Int, td::Vector{NTuple{3,Int}}, vd::Vector{NTuple{3,Int}},
               kinds::Vector{Int}, pin::Ptr{Cvoid}, pout::Ptr{Cvoid}, sign::Int, flags::UInt32;
               split::Bool = false, pin2::Ptr{Cvoid} = C_NULL, pout2::Ptr{Cvoid} = C_NULL) where {T}
  flags & W_WISDOM_ONLY != 0 &&
    throw(ArgumentError("FFTW interface: FFTW_WISDOM_ONLY, and FFTL keeps no wisdom"))
  (typ != W_C2C || sign == -1 || sign == 1) || throw(ArgumentError("FFTW interface: sign must be -1 or +1"))
  lay = solve_layout(td, vd, typ, pin == pout, !split)
  fk = FFTL.R2RKind[]
  if typ == W_R2R
    for t in lay.tidx
      push!(fk, FFTL.R2RKind(W_KINDS[kinds[t] + 1]))
    end
  end
  nt = wthreads(T)[]
  prec = wprec(T)
  if typ == W_C2C
    wregister!(wreg_c2c(T), wrecord(RPlan{T}, T, typ, lay, fk, sign, nt, pin, pin2, pout, pout2, split), typ, prec)
  elseif typ == W_R2R
    wregister!(wreg_rr(T), wrecord(RR2RPlan{T}, T, typ, lay, fk, sign, nt, pin, pin2, pout, pout2, split), typ, prec)
  else
    wregister!(wreg_rf(T), wrecord(RRFFTPlan{T}, T, typ, lay, fk, sign, nt, pin, pin2, pout, pout2, split), typ, prec)
  end
end

read_ints(p::Ptr{Cint}, n::Int) = Int[Int(unsafe_load(p, i)) for i in 1:n]

function read_iodims(p::Ptr{I}, n::Int) where {I}
  n >= 0 || throw(ArgumentError("FFTW interface: ranks must be nonnegative"))
  v = Vector{NTuple{3,Int}}(undef, n)
  for i in 1:n
    d = unsafe_load(p, i)
    v[i] = (Int(d.n), Int(d.is), Int(d.os))
  end
  v
end

function wplan_many(::Type{T}, typ::Int, rank::Int, n::Ptr{Cint}, howmany::Int, pin::Ptr{Cvoid},
                    inembed::Ptr{Cint}, istride::Int, idist::Int, pout::Ptr{Cvoid}, onembed::Ptr{Cint},
                    ostride::Int, odist::Int, kinds::Ptr{Cint}, sign::Int, flags::UInt32) where {T}
  rank >= 0 || throw(ArgumentError("FFTW interface: rank must be nonnegative"))
  nv = read_ints(n, rank)
  ie = inembed == C_NULL ? Int[] : read_ints(inembed, rank)
  oe = onembed == C_NULL ? Int[] : read_ints(onembed, rank)
  ks = typ == W_R2R ? read_kinds_fftw(kinds, rank) : Int[]
  td, vd = many_dims(nv, howmany, ie, istride, idist, oe, ostride, odist, typ, pin == pout)
  wplan(T, typ, td, vd, ks, pin, pout, sign, flags)
end

function wplan_guru(::Type{T}, typ::Int, rank::Int, dims::Ptr{I}, hrank::Int, hdims::Ptr{I},
                    pin::Ptr{Cvoid}, pout::Ptr{Cvoid}, kinds::Ptr{Cint}, sign::Int, flags::UInt32;
                    split::Bool = false, pin2::Ptr{Cvoid} = C_NULL, pout2::Ptr{Cvoid} = C_NULL) where {T,I}
  td = read_iodims(dims, rank)
  vd = read_iodims(hdims, hrank)
  ks = typ == W_R2R ? read_kinds_fftw(kinds, rank) : Int[]
  wplan(T, typ, td, vd, ks, pin, pout, sign, flags; split = split, pin2 = pin2, pout2 = pout2)
end

# --- execution --------------------------------------------------------------------------

wrun!(p::RPlan{T}, ::Int, out::Ptr{Cvoid}, in_::Ptr{Cvoid}) where {T} =
  execute!(p, Ptr{Complex{T}}(out), Ptr{Complex{T}}(in_))
wrun!(p::RRFFTPlan{T}, typ::Int, out::Ptr{Cvoid}, in_::Ptr{Cvoid}) where {T} =
  typ == W_R2C ? execute_r2c!(p, Ptr{Complex{T}}(out), Ptr{T}(in_)) :
                 execute_c2r!(p, Ptr{T}(out), Ptr{Complex{T}}(in_))
wrun!(p::RR2RPlan{T}, ::Int, out::Ptr{Cvoid}, in_::Ptr{Cvoid}) where {T} =
  execute_r2r!(p, Ptr{T}(out), Ptr{T}(in_))

# Copies the strided array src (counts n, strides ss) to dst (strides ds); strides in elements
# of E. The first dimension is the inner loop. No dimensions: one element.
function wstrided_copy!(dst::Ptr{E}, ds::Vector{Int}, src::Ptr{E}, ss::Vector{Int}, n::Vector{Int}) where {E}
  k = length(n)
  if k == 0
    unsafe_store!(dst, unsafe_load(src))
    return nothing
  end
  for x in n
    x == 0 && return nothing
  end
  idx = zeros(Int, k)
  n1 = n[1]; d1 = ds[1]; s1 = ss[1]
  doff = 0; soff = 0
  while true
    for i in 0:n1-1
      unsafe_store!(dst, unsafe_load(src, soff + i * s1 + 1), doff + i * d1 + 1)
    end
    j = 2
    while j <= k
      idx[j] += 1; doff += ds[j]; soff += ss[j]
      idx[j] < n[j] && break
      doff -= ds[j] * n[j]; soff -= ss[j] * n[j]; idx[j] = 0
      j += 1
    end
    j > k && return nothing
  end
end

# One side of the copy fallback between the user's array (p, imaginary part p2 if split) and the
# buffer b. complex: the user's array is complex. gather: user → buffer.
function wcopy_side!(b::Ptr{T}, bs::Vector{Int}, p::Ptr{Cvoid}, p2::Ptr{Cvoid}, us::Vector{Int},
                     n::Vector{Int}, complex::Bool, split::Bool, gather::Bool) where {T}
  if !complex
    gather ? wstrided_copy!(b, bs, Ptr{T}(p), us, n) : wstrided_copy!(Ptr{T}(p), us, b, bs, n)
  elseif !split
    cb = Ptr{Complex{T}}(b); cu = Ptr{Complex{T}}(p)
    gather ? wstrided_copy!(cb, bs, cu, us, n) : wstrided_copy!(cu, us, cb, bs, n)
  else
    # split: real and imaginary parts as real arrays; the buffer holds them interleaved
    bs2 = Int[2s for s in bs]
    re = Ptr{T}(p); im = Ptr{T}(p2); bi = b + sizeof(T)
    if gather
      wstrided_copy!(b, bs2, re, us, n); wstrided_copy!(bi, bs2, im, us, n)
    else
      wstrided_copy!(re, us, b, bs2, n); wstrided_copy!(im, us, bi, bs2, n)
    end
  end
  nothing
end

function wexec(r::WPlan{P,T}, in_::Ptr{Cvoid}, in2::Ptr{Cvoid}, out::Ptr{Cvoid}, out2::Ptr{Cvoid}) where {P,T}
  r.noop && return nothing
  s = @lock r.lock (isempty(r.free) ? nothing : pop!(r.free))
  q = s === nothing ? new_slot(r) : s
  try
    if r.copy
      c = r.cp
      GC.@preserve q begin
        b = pointer(q.buf)
        vb = Ptr{Cvoid}(b)
        wcopy_side!(b, c.bin, in_, in2, c.istr, c.nin, r.typ == W_C2C || r.typ == W_C2R, r.split, true)
        wrun!(q.plan, r.typ, vb, vb)
        wcopy_side!(b, c.bout, out, out2, c.ostr, c.nout, r.typ == W_C2C || r.typ == W_R2C, r.split, false)
      end
    else
      GC.@preserve q wrun!(q.plan, r.typ, out, in_)
    end
  finally
    @lock r.lock push!(r.free, q)
  end
  nothing
end

# want: the plan type an execute function requires (-1: any, fftw_execute); split: whether
# the execute function takes split arrays.
function wexecute(::Type{T}, p::Ptr{Cvoid}, want::Int, split::Bool, in_::Ptr{Cvoid}, in2::Ptr{Cvoid},
                  out::Ptr{Cvoid}, out2::Ptr{Cvoid}, newarrays::Bool) where {T}
  typ, prec, idx = wdecode(p)
  if typ < 0 || prec != wprec(T)
    set_error!("FFTW interface: invalid plan")
    return nothing
  end
  if want >= 0 && typ != want
    set_error!("FFTW interface: the execute function does not match the plan's transform type")
    return nothing
  end
  if typ == W_C2C
    wexec_reg(wreg_c2c(T), idx, split, in_, in2, out, out2, newarrays)
  elseif typ == W_R2R
    wexec_reg(wreg_rr(T), idx, split, in_, in2, out, out2, newarrays)
  else
    wexec_reg(wreg_rf(T), idx, split, in_, in2, out, out2, newarrays)
  end
end

function wexec_reg(reg::Vector{Union{Nothing,WPlan{P,T}}}, idx::Int, split::Bool, in_::Ptr{Cvoid},
                   in2::Ptr{Cvoid}, out::Ptr{Cvoid}, out2::Ptr{Cvoid}, newarrays::Bool) where {P,T}
  r = wlookup(reg, idx)
  if r === nothing
    set_error!("FFTW interface: invalid plan")
    return nothing
  end
  if !newarrays
    wexec(r, r.pin, r.pin2, r.pout, r.pout2)
  elseif split != r.split
    set_error!(split ? "FFTW interface: split execute function on a plan for interleaved arrays" :
                       "FFTW interface: the plan is for split arrays; use the split execute function")
  elseif (in_ == out) != r.inplace
    set_error!("FFTW interface: new arrays must be in place exactly when the plan is")
  else
    wexec(r, in_, in2, out, out2)
  end
  nothing
end

# fftw_destroy_plan (copy = false) and fftw_copy_plan (copy = true): one reference less or more;
# false if p is no plan of this precision.
function wdestroy(::Type{T}, p::Ptr{Cvoid}, copy::Bool) where {T}
  typ, prec, idx = wdecode(p)
  (typ < 0 || prec != wprec(T)) && return false
  typ == W_C2C ? wrefs!(wreg_c2c(T), idx, copy) :
  typ == W_R2R ? wrefs!(wreg_rr(T), idx, copy) : wrefs!(wreg_rf(T), idx, copy)
end

function wrefs!(reg::Vector{Union{Nothing,WPlan{P,T}}}, idx::Int, up::Bool) where {P,T}
  @lock REGISTRY_LOCK begin
    0 <= idx <= length(reg) - 1 || return false
    r = reg[idx + 1]
    r === nothing && return false
    r.refs += up ? 1 : -1
    r.refs == 0 && (reg[idx + 1] = nothing)
    true
  end
end

# --- information --------------------------------------------------------------------------

# Floating-point operations of the plan: the textbook estimate 5 N log2 N for a complex DFT of
# N = the product of the transformed lengths (3 N log2 N additions, 2 N log2 N multiplications),
# half of it for the real transforms, times the number of transforms. Not a count of FFTL's
# operations.
function wops(r::WPlan)
  r.noop && return (0.0, 0.0)
  N = 1
  for d in r.region
    N *= r.sz[d]
  end
  howmany = 1
  for x in r.sz
    howmany *= x
  end
  howmany = howmany ÷ N
  f = N > 1 ? N * log2(Float64(N)) * howmany : 0.0
  s = r.typ == W_C2C ? 1.0 : 0.5
  (3s * f, 2s * f)
end

function wdesc(r::WPlan)
  kind = r.typ == W_C2C ? (r.sign < 0 ? "dft forward" : "dft backward") :
         r.typ == W_R2C ? "r2c" : r.typ == W_C2R ? "c2r" : "r2r"
  s = "(fftl " * kind * (r.inplace ? " in place" : " out of place") * " size "
  for i in eachindex(r.sz)
    s *= (i > 1 ? "x" : "") * string(r.sz[i])
  end
  s *= " (column major) dims"
  for d in r.region
    s *= " " * string(d)
  end
  if r.typ == W_R2R
    s *= " kinds"
    for k in r.kinds
      s *= " " * string(Int(k))
    end
  end
  s * " threads " * string(r.threads) * (r.split ? " split" : "") * (r.copy ? " copied" : "") *
    (r.noop ? " empty" : "") * ")"
end

function winfo(::Type{T}, p::Ptr{Cvoid}) where {T}
  typ, prec, idx = wdecode(p)
  (typ < 0 || prec != wprec(T)) && return ("", (0.0, 0.0), false)
  typ == W_C2C ? winfo_reg(wreg_c2c(T), idx) :
  typ == W_R2R ? winfo_reg(wreg_rr(T), idx) : winfo_reg(wreg_rf(T), idx)
end

function winfo_reg(reg::Vector{Union{Nothing,WPlan{P,T}}}, idx::Int) where {P,T}
  r = wlookup(reg, idx)
  r === nothing && return ("", (0.0, 0.0), false)
  (wdesc(r), wops(r), true)
end

# A malloc'ed copy of s (the caller frees it with free)
function malloc_cstr(s::String)
  n = ncodeunits(s)
  buf = Ptr{UInt8}(Libc.malloc(n + 1))
  buf == C_NULL && return buf
  GC.@preserve s unsafe_copyto!(buf, pointer(s), n)
  unsafe_store!(buf, 0x00, n + 1)
  buf
end

c_fputs(s::String, f::Ptr{Cvoid}) = ccall(:fputs, Cint, (Cstring, Ptr{Cvoid}), s, f)

c_stdout() = Sys.isapple() ? unsafe_load(cglobal(:__stdoutp, Ptr{Cvoid})) :
                             unsafe_load(cglobal(:stdout, Ptr{Cvoid}))

# --- wisdom -----------------------------------------------------------------------------------

# FFTL keeps no wisdom; export writes this placeholder (the manual leaves the format open,
# "LISP-like ASCII text"), import reads nothing and reports success.
const W_WISDOM = "(fftl-wisdom)\n"

function wexport_filename(name::Cstring)
  f = ccall(:fopen, Ptr{Cvoid}, (Cstring, Cstring), name, "w")
  f == C_NULL && return Cint(0)
  ok = c_fputs(W_WISDOM, f) >= 0
  ok &= ccall(:fclose, Cint, (Ptr{Cvoid},), f) == 0
  Cint(ok)
end

function wexport_callback(write_char::Ptr{Cvoid}, data::Ptr{Cvoid})
  for c in codeunits(W_WISDOM)
    ccall(write_char, Cvoid, (Cchar, Ptr{Cvoid}), c % Cchar, data)
  end
  nothing
end

function wimport_filename(name::Cstring)
  f = ccall(:fopen, Ptr{Cvoid}, (Cstring, Cstring), name, "r")
  f == C_NULL && return Cint(0)
  ccall(:fclose, Cint, (Ptr{Cvoid},), f)
  Cint(1)
end

# --- memory -----------------------------------------------------------------------------------

# 64-byte alignment: a cache line, and more than any SIMD width FFTL uses
function wmalloc(n::Csize_t)
  p = Ref{Ptr{Cvoid}}(C_NULL)
  ccall(:posix_memalign, Cint, (Ref{Ptr{Cvoid}}, Csize_t, Csize_t), p, 64, max(n, Csize_t(1))) == 0 ||
    return (set_error!("FFTW interface: out of memory"); C_NULL)
  p[]
end

function walloc(n::Csize_t, size::Int)
  n > typemax(Csize_t) ÷ Csize_t(size) && return (set_error!("FFTW interface: allocation size overflows"); C_NULL)
  wmalloc(n * Csize_t(size))
end

# --- the C entry points ---------------------------------------------------------------------

macro wfailed(e)
  e = esc(e)
  quote
    if $e isa ArgumentError
      set_last_error($e)
    elseif $e isa DimensionMismatch
      set_last_error($e)
    elseif $e isa ErrorException
      set_last_error($e)
    else
      set_error!("FFTW interface: plan creation failed")
    end
    C_NULL
  end
end

dense_iodims(n::Vector{Int}, typ::Int, inplace::Bool) =
  many_dims(n, 1, Int[], 1, 0, Int[], 1, 0, typ, inplace)

function wplan_basic(::Type{T}, typ::Int, n::Vector{Int}, pin::Ptr{Cvoid}, pout::Ptr{Cvoid},
                     kinds::Vector{Int}, sign::Int, flags::UInt32) where {T}
  td, vd = dense_iodims(n, typ, pin == pout)
  for k in kinds
    0 <= k <= 10 || throw(ArgumentError("FFTW interface: unknown r2r kind"))
  end
  wplan(T, typ, td, vd, kinds, pin, pout, sign, flags)
end

function wplan_rank(::Type{T}, typ::Int, rank::Int, n::Ptr{Cint}, pin::Ptr{Cvoid}, pout::Ptr{Cvoid},
                    kinds::Ptr{Cint}, sign::Int, flags::UInt32) where {T}
  rank >= 0 || throw(ArgumentError("FFTW interface: rank must be nonnegative"))
  ks = typ == W_R2R ? read_kinds_fftw(kinds, rank) : Int[]
  wplan_basic(T, typ, read_ints(n, rank), pin, pout, ks, sign, flags)
end

for (T, pre) in ((Float64, "fftw_"), (Float32, "fftwf_"))
  wname(name) = Symbol(pre, name)
  @eval begin
    # --- basic interface
    Base.@ccallable function $(wname(:plan_dft_1d))(n0::Cint, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, sign::Cint, flags::Cuint)::Ptr{Cvoid}
      try wplan_basic($T, W_C2C, Int[n0], in_, out, Int[], Int(sign), UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_dft_2d))(n0::Cint, n1::Cint, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, sign::Cint, flags::Cuint)::Ptr{Cvoid}
      try wplan_basic($T, W_C2C, Int[n0, n1], in_, out, Int[], Int(sign), UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_dft_3d))(n0::Cint, n1::Cint, n2::Cint, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, sign::Cint, flags::Cuint)::Ptr{Cvoid}
      try wplan_basic($T, W_C2C, Int[n0, n1, n2], in_, out, Int[], Int(sign), UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_dft))(rank::Cint, n::Ptr{Cint}, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, sign::Cint, flags::Cuint)::Ptr{Cvoid}
      try wplan_rank($T, W_C2C, Int(rank), n, in_, out, Ptr{Cint}(C_NULL), Int(sign), UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_dft_r2c_1d))(n0::Cint, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Cuint)::Ptr{Cvoid}
      try wplan_basic($T, W_R2C, Int[n0], in_, out, Int[], -1, UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_dft_r2c_2d))(n0::Cint, n1::Cint, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Cuint)::Ptr{Cvoid}
      try wplan_basic($T, W_R2C, Int[n0, n1], in_, out, Int[], -1, UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_dft_r2c_3d))(n0::Cint, n1::Cint, n2::Cint, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Cuint)::Ptr{Cvoid}
      try wplan_basic($T, W_R2C, Int[n0, n1, n2], in_, out, Int[], -1, UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_dft_r2c))(rank::Cint, n::Ptr{Cint}, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Cuint)::Ptr{Cvoid}
      try wplan_rank($T, W_R2C, Int(rank), n, in_, out, Ptr{Cint}(C_NULL), -1, UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_dft_c2r_1d))(n0::Cint, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Cuint)::Ptr{Cvoid}
      try wplan_basic($T, W_C2R, Int[n0], in_, out, Int[], 1, UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_dft_c2r_2d))(n0::Cint, n1::Cint, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Cuint)::Ptr{Cvoid}
      try wplan_basic($T, W_C2R, Int[n0, n1], in_, out, Int[], 1, UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_dft_c2r_3d))(n0::Cint, n1::Cint, n2::Cint, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Cuint)::Ptr{Cvoid}
      try wplan_basic($T, W_C2R, Int[n0, n1, n2], in_, out, Int[], 1, UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_dft_c2r))(rank::Cint, n::Ptr{Cint}, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Cuint)::Ptr{Cvoid}
      try wplan_rank($T, W_C2R, Int(rank), n, in_, out, Ptr{Cint}(C_NULL), 1, UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_r2r_1d))(n0::Cint, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, k0::Cint, flags::Cuint)::Ptr{Cvoid}
      try wplan_basic($T, W_R2R, Int[n0], in_, out, Int[k0], 0, UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_r2r_2d))(n0::Cint, n1::Cint, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, k0::Cint, k1::Cint, flags::Cuint)::Ptr{Cvoid}
      try wplan_basic($T, W_R2R, Int[n0, n1], in_, out, Int[k0, k1], 0, UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_r2r_3d))(n0::Cint, n1::Cint, n2::Cint, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, k0::Cint, k1::Cint, k2::Cint, flags::Cuint)::Ptr{Cvoid}
      try wplan_basic($T, W_R2R, Int[n0, n1, n2], in_, out, Int[k0, k1, k2], 0, UInt32(flags)) catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_r2r))(rank::Cint, n::Ptr{Cint}, in_::Ptr{Cvoid}, out::Ptr{Cvoid}, kinds::Ptr{Cint}, flags::Cuint)::Ptr{Cvoid}
      try wplan_rank($T, W_R2R, Int(rank), n, in_, out, kinds, 0, UInt32(flags)) catch e; @wfailed(e) end
    end

    # --- advanced interface
    Base.@ccallable function $(wname(:plan_many_dft))(rank::Cint, n::Ptr{Cint}, howmany::Cint,
        in_::Ptr{Cvoid}, inembed::Ptr{Cint}, istride::Cint, idist::Cint,
        out::Ptr{Cvoid}, onembed::Ptr{Cint}, ostride::Cint, odist::Cint, sign::Cint, flags::Cuint)::Ptr{Cvoid}
      try
        wplan_many($T, W_C2C, Int(rank), n, Int(howmany), in_, inembed, Int(istride), Int(idist),
                   out, onembed, Int(ostride), Int(odist), Ptr{Cint}(C_NULL), Int(sign), UInt32(flags))
      catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_many_dft_r2c))(rank::Cint, n::Ptr{Cint}, howmany::Cint,
        in_::Ptr{Cvoid}, inembed::Ptr{Cint}, istride::Cint, idist::Cint,
        out::Ptr{Cvoid}, onembed::Ptr{Cint}, ostride::Cint, odist::Cint, flags::Cuint)::Ptr{Cvoid}
      try
        wplan_many($T, W_R2C, Int(rank), n, Int(howmany), in_, inembed, Int(istride), Int(idist),
                   out, onembed, Int(ostride), Int(odist), Ptr{Cint}(C_NULL), -1, UInt32(flags))
      catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_many_dft_c2r))(rank::Cint, n::Ptr{Cint}, howmany::Cint,
        in_::Ptr{Cvoid}, inembed::Ptr{Cint}, istride::Cint, idist::Cint,
        out::Ptr{Cvoid}, onembed::Ptr{Cint}, ostride::Cint, odist::Cint, flags::Cuint)::Ptr{Cvoid}
      try
        wplan_many($T, W_C2R, Int(rank), n, Int(howmany), in_, inembed, Int(istride), Int(idist),
                   out, onembed, Int(ostride), Int(odist), Ptr{Cint}(C_NULL), 1, UInt32(flags))
      catch e; @wfailed(e) end
    end
    Base.@ccallable function $(wname(:plan_many_r2r))(rank::Cint, n::Ptr{Cint}, howmany::Cint,
        in_::Ptr{Cvoid}, inembed::Ptr{Cint}, istride::Cint, idist::Cint,
        out::Ptr{Cvoid}, onembed::Ptr{Cint}, ostride::Cint, odist::Cint, kinds::Ptr{Cint}, flags::Cuint)::Ptr{Cvoid}
      try
        wplan_many($T, W_R2R, Int(rank), n, Int(howmany), in_, inembed, Int(istride), Int(idist),
                   out, onembed, Int(ostride), Int(odist), kinds, 0, UInt32(flags))
      catch e; @wfailed(e) end
    end
  end

  # --- guru and guru64 interfaces
  for (g, I) in ((:guru_, WIodim), (:guru64_, WIodim64))
    @eval begin
      Base.@ccallable function $(wname(Symbol(:plan_, g, :dft)))(rank::Cint, dims::Ptr{$I}, hrank::Cint, hdims::Ptr{$I},
          in_::Ptr{Cvoid}, out::Ptr{Cvoid}, sign::Cint, flags::Cuint)::Ptr{Cvoid}
        try wplan_guru($T, W_C2C, Int(rank), dims, Int(hrank), hdims, in_, out, Ptr{Cint}(C_NULL), Int(sign), UInt32(flags)) catch e; @wfailed(e) end
      end
      Base.@ccallable function $(wname(Symbol(:plan_, g, :dft_r2c)))(rank::Cint, dims::Ptr{$I}, hrank::Cint, hdims::Ptr{$I},
          in_::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Cuint)::Ptr{Cvoid}
        try wplan_guru($T, W_R2C, Int(rank), dims, Int(hrank), hdims, in_, out, Ptr{Cint}(C_NULL), -1, UInt32(flags)) catch e; @wfailed(e) end
      end
      Base.@ccallable function $(wname(Symbol(:plan_, g, :dft_c2r)))(rank::Cint, dims::Ptr{$I}, hrank::Cint, hdims::Ptr{$I},
          in_::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Cuint)::Ptr{Cvoid}
        try wplan_guru($T, W_C2R, Int(rank), dims, Int(hrank), hdims, in_, out, Ptr{Cint}(C_NULL), 1, UInt32(flags)) catch e; @wfailed(e) end
      end
      Base.@ccallable function $(wname(Symbol(:plan_, g, :r2r)))(rank::Cint, dims::Ptr{$I}, hrank::Cint, hdims::Ptr{$I},
          in_::Ptr{Cvoid}, out::Ptr{Cvoid}, kinds::Ptr{Cint}, flags::Cuint)::Ptr{Cvoid}
        try wplan_guru($T, W_R2R, Int(rank), dims, Int(hrank), hdims, in_, out, kinds, 0, UInt32(flags)) catch e; @wfailed(e) end
      end
      Base.@ccallable function $(wname(Symbol(:plan_, g, :split_dft)))(rank::Cint, dims::Ptr{$I}, hrank::Cint, hdims::Ptr{$I},
          ri::Ptr{Cvoid}, ii::Ptr{Cvoid}, ro::Ptr{Cvoid}, io::Ptr{Cvoid}, flags::Cuint)::Ptr{Cvoid}
        # always forward, as in FFTW (backward: swap the real and imaginary parts)
        try
          wplan_guru($T, W_C2C, Int(rank), dims, Int(hrank), hdims, ri, ro, Ptr{Cint}(C_NULL), -1, UInt32(flags);
                     split = true, pin2 = ii, pout2 = io)
        catch e; @wfailed(e) end
      end
      Base.@ccallable function $(wname(Symbol(:plan_, g, :split_dft_r2c)))(rank::Cint, dims::Ptr{$I}, hrank::Cint, hdims::Ptr{$I},
          in_::Ptr{Cvoid}, ro::Ptr{Cvoid}, io::Ptr{Cvoid}, flags::Cuint)::Ptr{Cvoid}
        try
          wplan_guru($T, W_R2C, Int(rank), dims, Int(hrank), hdims, in_, ro, Ptr{Cint}(C_NULL), -1, UInt32(flags);
                     split = true, pout2 = io)
        catch e; @wfailed(e) end
      end
      Base.@ccallable function $(wname(Symbol(:plan_, g, :split_dft_c2r)))(rank::Cint, dims::Ptr{$I}, hrank::Cint, hdims::Ptr{$I},
          ri::Ptr{Cvoid}, ii::Ptr{Cvoid}, out::Ptr{Cvoid}, flags::Cuint)::Ptr{Cvoid}
        try
          wplan_guru($T, W_C2R, Int(rank), dims, Int(hrank), hdims, ri, out, Ptr{Cint}(C_NULL), 1, UInt32(flags);
                     split = true, pin2 = ii)
        catch e; @wfailed(e) end
      end
    end
  end

  @eval begin
    # --- execution
    Base.@ccallable function $(wname(:execute))(p::Ptr{Cvoid})::Cvoid
      try wexecute($T, p, -1, false, C_NULL, C_NULL, C_NULL, C_NULL, false) catch; set_error!("FFTW interface: execution failed") end
      nothing
    end
    Base.@ccallable function $(wname(:execute_dft))(p::Ptr{Cvoid}, in_::Ptr{Cvoid}, out::Ptr{Cvoid})::Cvoid
      try wexecute($T, p, W_C2C, false, in_, C_NULL, out, C_NULL, true) catch; set_error!("FFTW interface: execution failed") end
      nothing
    end
    Base.@ccallable function $(wname(:execute_dft_r2c))(p::Ptr{Cvoid}, in_::Ptr{Cvoid}, out::Ptr{Cvoid})::Cvoid
      try wexecute($T, p, W_R2C, false, in_, C_NULL, out, C_NULL, true) catch; set_error!("FFTW interface: execution failed") end
      nothing
    end
    Base.@ccallable function $(wname(:execute_dft_c2r))(p::Ptr{Cvoid}, in_::Ptr{Cvoid}, out::Ptr{Cvoid})::Cvoid
      try wexecute($T, p, W_C2R, false, in_, C_NULL, out, C_NULL, true) catch; set_error!("FFTW interface: execution failed") end
      nothing
    end
    Base.@ccallable function $(wname(:execute_r2r))(p::Ptr{Cvoid}, in_::Ptr{Cvoid}, out::Ptr{Cvoid})::Cvoid
      try wexecute($T, p, W_R2R, false, in_, C_NULL, out, C_NULL, true) catch; set_error!("FFTW interface: execution failed") end
      nothing
    end
    Base.@ccallable function $(wname(:execute_split_dft))(p::Ptr{Cvoid}, ri::Ptr{Cvoid}, ii::Ptr{Cvoid}, ro::Ptr{Cvoid}, io::Ptr{Cvoid})::Cvoid
      try wexecute($T, p, W_C2C, true, ri, ii, ro, io, true) catch; set_error!("FFTW interface: execution failed") end; nothing
    end
    Base.@ccallable function $(wname(:execute_split_dft_r2c))(p::Ptr{Cvoid}, in_::Ptr{Cvoid}, ro::Ptr{Cvoid}, io::Ptr{Cvoid})::Cvoid
      try wexecute($T, p, W_R2C, true, in_, C_NULL, ro, io, true) catch; set_error!("FFTW interface: execution failed") end; nothing
    end
    Base.@ccallable function $(wname(:execute_split_dft_c2r))(p::Ptr{Cvoid}, ri::Ptr{Cvoid}, ii::Ptr{Cvoid}, out::Ptr{Cvoid})::Cvoid
      try wexecute($T, p, W_C2R, true, ri, ii, out, C_NULL, true) catch; set_error!("FFTW interface: execution failed") end; nothing
    end

    # --- plans
    Base.@ccallable function $(wname(:destroy_plan))(p::Ptr{Cvoid})::Cvoid
      p == C_NULL || wdestroy($T, p, false) || set_error!("FFTW interface: invalid plan")
      nothing
    end
    Base.@ccallable function $(wname(:copy_plan))(p::Ptr{Cvoid})::Ptr{Cvoid}
      wdestroy($T, p, true) ? p : (set_error!("FFTW interface: invalid plan"); C_NULL)
    end
    # Frees the cached tables; plans keep what they reference, so existing plans stay valid.
    Base.@ccallable function $(wname(:cleanup))()::Cvoid
      FFTL.clear_cache!(); nothing
    end
    Base.@ccallable function $(wname(:set_timelimit))(t::Cdouble)::Cvoid
      nothing
    end
    # The estimate of wops, not a measured or FFTW-comparable cost.
    Base.@ccallable function $(wname(:cost))(p::Ptr{Cvoid})::Cdouble
      _, ops, _ = winfo($T, p)
      ops[1] + ops[2]
    end
    # The estimate of wops (no fused multiply-adds), not an operation count of FFTL.
    Base.@ccallable function $(wname(:flops))(p::Ptr{Cvoid}, add::Ptr{Cdouble}, mul::Ptr{Cdouble}, fma::Ptr{Cdouble})::Cvoid
      _, ops, _ = winfo($T, p)
      add == C_NULL || unsafe_store!(add, ops[1])
      mul == C_NULL || unsafe_store!(mul, ops[2])
      fma == C_NULL || unsafe_store!(fma, 0.0)
      nothing
    end
    Base.@ccallable function $(wname(:sprint_plan))(p::Ptr{Cvoid})::Ptr{UInt8}
      s, _, ok = winfo($T, p)
      ok ? malloc_cstr(s) : (set_error!("FFTW interface: invalid plan"); Ptr{UInt8}(C_NULL))
    end
    Base.@ccallable function $(wname(:fprint_plan))(p::Ptr{Cvoid}, file::Ptr{Cvoid})::Cvoid
      s, _, ok = winfo($T, p)
      ok && file != C_NULL && c_fputs(s, file)
      nothing
    end
    Base.@ccallable function $(wname(:print_plan))(p::Ptr{Cvoid})::Cvoid
      s, _, ok = winfo($T, p)
      ok && c_fputs(s, c_stdout())
      nothing
    end

    # --- memory
    Base.@ccallable function $(wname(:malloc))(n::Csize_t)::Ptr{Cvoid}
      wmalloc(n)
    end
    Base.@ccallable function $(wname(:free))(p::Ptr{Cvoid})::Cvoid
      Libc.free(p); nothing
    end
    Base.@ccallable function $(wname(:alloc_real))(n::Csize_t)::Ptr{$T}
      Ptr{$T}(walloc(n, sizeof($T)))
    end
    Base.@ccallable function $(wname(:alloc_complex))(n::Csize_t)::Ptr{Cvoid}
      walloc(n, 2sizeof($T))
    end
    # FFTL imposes no alignment, so all arrays are equivalently aligned.
    Base.@ccallable function $(wname(:alignment_of))(p::Ptr{$T})::Cint
      Cint(0)
    end

    # --- wisdom
    Base.@ccallable function $(wname(:export_wisdom_to_filename))(name::Cstring)::Cint
      wexport_filename(name)
    end
    Base.@ccallable function $(wname(:export_wisdom_to_file))(file::Ptr{Cvoid})::Cvoid
      file == C_NULL || c_fputs(W_WISDOM, file); nothing
    end
    Base.@ccallable function $(wname(:export_wisdom_to_string))()::Ptr{UInt8}
      malloc_cstr(W_WISDOM)
    end
    Base.@ccallable function $(wname(:export_wisdom))(write_char::Ptr{Cvoid}, data::Ptr{Cvoid})::Cvoid
      wexport_callback(write_char, data)
    end
    # There is no system wisdom; FFTW returns 0 in that case as well.
    Base.@ccallable function $(wname(:import_system_wisdom))()::Cint
      Cint(0)
    end
    Base.@ccallable function $(wname(:import_wisdom_from_filename))(name::Cstring)::Cint
      wimport_filename(name)
    end
    Base.@ccallable function $(wname(:import_wisdom_from_file))(file::Ptr{Cvoid})::Cint
      Cint(1)
    end
    Base.@ccallable function $(wname(:import_wisdom_from_string))(s::Cstring)::Cint
      Cint(1)
    end
    Base.@ccallable function $(wname(:import_wisdom))(read_char::Ptr{Cvoid}, data::Ptr{Cvoid})::Cint
      Cint(1)
    end
    Base.@ccallable function $(wname(:forget_wisdom))()::Cvoid
      nothing
    end

    # --- threads. Execution runs on the library's worker pool (include/fftl.h, Threading);
    # the callback of fftw_threads_set_callback is not used.
    Base.@ccallable function $(wname(:init_threads))()::Cint
      Cint(1)
    end
    Base.@ccallable function $(wname(:plan_with_nthreads))(n::Cint)::Cvoid
      wthreads($T)[] = max(1, Int(n)); nothing
    end
    Base.@ccallable function $(wname(:planner_nthreads))()::Cint
      Cint(wthreads($T)[])
    end
    Base.@ccallable function $(wname(:cleanup_threads))()::Cvoid
      FFTL.clear_cache!(); nothing
    end
    # Planning is thread safe already: FFTL's table cache and the registries are locked.
    Base.@ccallable function $(wname(:make_planner_thread_safe))()::Cvoid
      nothing
    end
    Base.@ccallable function $(wname(:threads_set_callback))(loop::Ptr{Cvoid}, data::Ptr{Cvoid})::Cvoid
      nothing
    end
  end
end
