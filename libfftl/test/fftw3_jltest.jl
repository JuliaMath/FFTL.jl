# Tests of the FFTW3 interface (src/fftw3api.jl) without a juliac build: its @ccallable entry
# points are plain Julia methods, called here with pointers to Julia arrays and compared with
# FFTW.jl (FFTW_jll) in the same process. Julia arrays are column major, FFTW's C arrays row
# major: a Julia array of size reverse(n) is the C array n[0] x ... x n[rank-1].
#
# Run from libfftl/: julia --project=test -t 4 test/fftw3_jltest.jl

using FFTW, FFTL, Random
include(joinpath(@__DIR__, "..", "src", "libfftl.jl"))

Random.seed!(17)

ntests = 0
nfailed = 0

macro CHECK(ex)
  Meta.isexpr(ex, :tuple) && length(ex.args) == 2 ||
    error("@CHECK takes the form: @CHECK condition, name")
  esc(quote
    global ntests, nfailed
    local cond, nm = $ex
    ntests += 1
    cond === true || (nfailed += 1; println("FAIL ", nm))
  end)
end

# FFTW constants from FFTW.jl
const FLAGS = (FFTW.ESTIMATE, FFTW.MEASURE, FFTW.PATIENT, FFTW.EXHAUSTIVE,
               FFTW.ESTIMATE | FFTW.DESTROY_INPUT, FFTW.MEASURE | FFTW.PRESERVE_INPUT,
               FFTW.ESTIMATE | FFTW.UNALIGNED)
const ALLKINDS = (FFTW.R2HC, FFTW.HC2R, FFTW.DHT, FFTW.REDFT00, FFTW.REDFT01, FFTW.REDFT10,
                  FFTW.REDFT11, FFTW.RODFT00, FFTW.RODFT01, FFTW.RODFT10, FFTW.RODFT11)

tol(::Type{Float64}) = 1e-12
tol(::Type{Float32}) = 2e-5
relerr(a, b) = maximum(abs.(a .- b); init = 0.0) / max(maximum(abs.(b); init = 0.0), eps())
close(a, b, T) = relerr(ComplexF64.(a), ComplexF64.(b)) < tol(T)

F(pre, name) = getfield(Main, Symbol(pre, name))
vp(a) = Ptr{Cvoid}(pointer(a))
pfx(T) = T == Float64 ? "fftw_" : "fftwf_"
flag() = rand(FLAGS)
cs(f, s::String) = GC.@preserve s f(convert(Cstring, pointer(s)))
lasterr() = (buf = zeros(UInt8, 256); n = fftl_last_error(pointer(buf), Csize_t(256)); String(buf[1:min(n, 255)]))

# Reference DFTs in double precision (FFTW.jl)
cref(x, dims, sign) = sign < 0 ? FFTW.fft(ComplexF64.(x), dims) : FFTW.bfft(ComplexF64.(x), dims)
r2r_ref(x, kinds, dims) = FFTW.r2r(Float64.(x), kinds, dims)

# --- complex DFTs, basic interface ---------------------------------------------------------

function test_c2c_basic(T, n::Vector{Int})
  pre = pfx(T); rank = length(n)
  for sign in (-1, 1), inplace in (false, true)
    name = "$pre c2c n=$n sign=$sign inplace=$inplace"
    x = randn(Complex{T}, reverse(n)...)
    ref = cref(x, 1:rank, sign)
    for via in (:sized, :rank)
      rank > 3 && via == :sized && continue
      a = copy(x); b = inplace ? a : similar(x)
      nn = Cint.(n)
      p = GC.@preserve a b nn begin
        if via == :sized
          rank == 1 ? F(pre, :plan_dft_1d)(nn[1], vp(a), vp(b), Cint(sign), flag()) :
          rank == 2 ? F(pre, :plan_dft_2d)(nn[1], nn[2], vp(a), vp(b), Cint(sign), flag()) :
                      F(pre, :plan_dft_3d)(nn[1], nn[2], nn[3], vp(a), vp(b), Cint(sign), flag())
        else
          F(pre, :plan_dft)(Cint(rank), pointer(nn), vp(a), vp(b), Cint(sign), flag())
        end
      end
      @CHECK p != C_NULL, "$name $via: plan"
      p == C_NULL && continue
      @CHECK a == x, "$name $via: planning leaves the input alone"
      GC.@preserve a b F(pre, :execute)(p)
      @CHECK close(b, ref, T), "$name $via: fftw_execute vs FFTW.jl"
      F(pre, :destroy_plan)(p)
    end
  end
end

# --- r2c / c2r, basic interface -------------------------------------------------------------

# A real array of logical size reverse(n) in FFTW's padded in-place layout (first Julia dimension
# 2(n ÷ 2 + 1)), with the values of x.
function padded(x::Array{T}) where {T}
  m = size(x, 1); h = m ÷ 2 + 1
  a = zeros(T, 2h, size(x)[2:end]...)
  a[1:m, ntuple(_ -> Colon(), ndims(x) - 1)...] .= x
  a
end

function test_real_basic(T, n::Vector{Int})
  pre = pfx(T); rank = length(n)
  jn = reverse(n); h = jn[1] ÷ 2 + 1
  csz = (h, jn[2:end]...)
  for inplace in (false, true), via in (:sized, :rank)
    name = "$pre r2c/c2r n=$n inplace=$inplace $via"
    x = randn(T, jn...)
    ref = FFTW.rfft(Float64.(x))
    nn = Cint.(n)
    plan(kind, a, b) = GC.@preserve a b nn begin
      s = kind == :r2c ? "dft_r2c" : "dft_c2r"
      if via == :sized
        rank == 1 ? F(pre, Symbol("plan_", s, "_1d"))(nn[1], vp(a), vp(b), flag()) :
        rank == 2 ? F(pre, Symbol("plan_", s, "_2d"))(nn[1], nn[2], vp(a), vp(b), flag()) :
                    F(pre, Symbol("plan_", s, "_3d"))(nn[1], nn[2], nn[3], vp(a), vp(b), flag())
      else
        F(pre, Symbol("plan_", s))(Cint(rank), pointer(nn), vp(a), vp(b), flag())
      end
    end
    # r2c
    if inplace
      a = padded(x)
      p = plan(:r2c, a, a)
      @CHECK p != C_NULL, "$name: r2c plan"
      p == C_NULL && continue
      GC.@preserve a F(pre, :execute)(p)
      y = reshape(reinterpret(Complex{T}, vec(a)), csz)
    else
      a = copy(x); y = zeros(Complex{T}, csz)
      p = plan(:r2c, a, y)
      @CHECK p != C_NULL, "$name: r2c plan"
      p == C_NULL && continue
      GC.@preserve a y F(pre, :execute)(p)
    end
    @CHECK close(y, ref, T), "$name: r2c vs FFTW.jl"
    F(pre, :destroy_plan)(p)
    # c2r on the exact half spectrum of x (Hermitian by construction)
    yin = Complex{T}.(ref)
    bref = FFTW.brfft(ref, jn[1])
    if inplace
      c = copy(yin)
      q = plan(:c2r, c, c)
      @CHECK q != C_NULL, "$name: c2r plan"
      q == C_NULL && continue
      GC.@preserve c F(pre, :execute)(q)
      out = reshape(reinterpret(T, vec(c)), (2h, jn[2:end]...))[1:jn[1], ntuple(_ -> Colon(), rank - 1)...]
    else
      c = copy(yin); out = zeros(T, jn...)
      q = plan(:c2r, c, out)
      @CHECK q != C_NULL, "$name: c2r plan"
      q == C_NULL && continue
      GC.@preserve c out F(pre, :execute)(q)
      @CHECK c == yin, "$name: out-of-place c2r keeps its input"
    end
    @CHECK close(out, bref, T), "$name: c2r vs FFTW.jl"
    F(pre, :destroy_plan)(q)
  end
end

# --- r2r, basic interface ----------------------------------------------------------------------

function test_r2r_basic(T, n::Vector{Int}, kinds::Vector{Int})
  pre = pfx(T); rank = length(n)
  for inplace in (false, true), via in (:sized, :rank)
    name = "$pre r2r n=$n kinds=$kinds inplace=$inplace $via"
    x = randn(T, reverse(n)...)
    ref = r2r_ref(x, Tuple(reverse(kinds)), 1:rank)
    a = copy(x); b = inplace ? a : zeros(T, size(x))
    nn = Cint.(n); kk = Cint.(kinds)
    p = GC.@preserve a b nn kk begin
      if via == :sized
        rank == 1 ? F(pre, :plan_r2r_1d)(nn[1], vp(a), vp(b), kk[1], flag()) :
        rank == 2 ? F(pre, :plan_r2r_2d)(nn[1], nn[2], vp(a), vp(b), kk[1], kk[2], flag()) :
                    F(pre, :plan_r2r_3d)(nn[1], nn[2], nn[3], vp(a), vp(b), kk[1], kk[2], kk[3], flag())
      else
        F(pre, :plan_r2r)(Cint(rank), pointer(nn), vp(a), vp(b), pointer(kk), flag())
      end
    end
    @CHECK p != C_NULL, "$name: plan"
    p == C_NULL && continue
    GC.@preserve a b F(pre, :execute)(p)
    @CHECK close(b, ref, T), "$name: vs FFTW.jl"
    F(pre, :destroy_plan)(p)
  end
end

# --- advanced interface ----------------------------------------------------------------------

function test_many(T)
  pre = pfx(T)
  # c2c: howmany arrays one after the other, NULL embeds and embeds equal to n
  for n in ([12], [5, 6], [3, 4, 5]), howmany in (1, 3), inplace in (false, true), emb in (false, true)
    name = "$pre many_dft n=$n howmany=$howmany inplace=$inplace embed=$emb"
    rank = length(n); N = prod(n)
    x = randn(Complex{T}, reverse(n)..., howmany)
    ref = cref(x, 1:rank, -1)
    a = copy(x); b = inplace ? a : similar(x)
    nn = Cint.(n); e = emb ? nn : Cint[]
    p = GC.@preserve a b nn e F(pre, :plan_many_dft)(Cint(rank), pointer(nn), Cint(howmany),
          vp(a), emb ? pointer(e) : Ptr{Cint}(C_NULL), Cint(1), Cint(N),
          vp(b), emb ? pointer(e) : Ptr{Cint}(C_NULL), Cint(1), Cint(N), Cint(-1), flag())
    @CHECK p != C_NULL, "$name: plan"
    p == C_NULL && continue
    GC.@preserve a b F(pre, :execute)(p)
    @CHECK close(b, ref, T), "$name: vs FFTW.jl"
    F(pre, :destroy_plan)(p)
  end
  # the manual's example: each column of a 10 x 3 row-major array (a dense array, transformed
  # along its slower Julia dimension)
  let x = randn(Complex{T}, 3, 10), y = similar(x), n = Cint[10]
    p = GC.@preserve x y n F(pre, :plan_many_dft)(Cint(1), pointer(n), Cint(3), vp(x), pointer(n),
          Cint(3), Cint(1), vp(y), pointer(n), Cint(3), Cint(1), Cint(1), flag())
    @CHECK p != C_NULL, "$pre many_dft columns: plan"
    if p != C_NULL
      GC.@preserve x y F(pre, :execute)(p)
      @CHECK close(y, cref(x, 2, 1), T), "$pre many_dft columns: vs FFTW.jl"
      F(pre, :destroy_plan)(p)
    end
  end
  # r2c / c2r with howmany, default (NULL) and explicit embeds
  for n in ([9], [6, 5], [4, 3, 6]), howmany in (1, 4), inplace in (false, true), emb in (false, true)
    name = "$pre many r2c/c2r n=$n howmany=$howmany inplace=$inplace embed=$emb"
    rank = length(n); jn = reverse(n); h = jn[1] ÷ 2 + 1
    rphys = copy(n); rphys[end] = inplace ? 2h : n[end]
    cphys = copy(n); cphys[end] = h
    idist = prod(rphys); odist = prod(cphys)
    x = randn(T, jn..., howmany)
    ref = FFTW.rfft(Float64.(x), 1:rank)
    nn = Cint.(n); re = Cint.(rphys); ce = Cint.(cphys)
    ie = emb ? pointer(re) : Ptr{Cint}(C_NULL); oe = emb ? pointer(ce) : Ptr{Cint}(C_NULL)
    a = inplace ? padded(x) : copy(x)
    y = inplace ? a : zeros(Complex{T}, h, jn[2:end]..., howmany)
    p = GC.@preserve a y nn re ce F(pre, :plan_many_dft_r2c)(Cint(rank), pointer(nn), Cint(howmany),
          vp(a), ie, Cint(1), Cint(idist), vp(y), oe, Cint(1), Cint(odist), flag())
    @CHECK p != C_NULL, "$name: r2c plan"
    p == C_NULL && continue
    GC.@preserve a y F(pre, :execute)(p)
    yc = inplace ? reshape(reinterpret(Complex{T}, vec(a)), (h, jn[2:end]..., howmany)) : y
    @CHECK close(yc, ref, T), "$name: r2c vs FFTW.jl"
    F(pre, :destroy_plan)(p)
    c = Complex{T}.(ref)
    out = inplace ? c : zeros(T, jn..., howmany)
    q = GC.@preserve c out nn re ce F(pre, :plan_many_dft_c2r)(Cint(rank), pointer(nn), Cint(howmany),
          vp(c), oe, Cint(1), Cint(odist), vp(out), ie, Cint(1), Cint(idist), flag())
    @CHECK q != C_NULL, "$name: c2r plan"
    q == C_NULL && continue
    GC.@preserve c out F(pre, :execute)(q)
    o = inplace ? reshape(reinterpret(T, vec(c)), (2h, jn[2:end]..., howmany))[1:jn[1], ntuple(_ -> Colon(), rank)...] : out
    @CHECK close(o, FFTW.brfft(ref, jn[1], 1:rank), T), "$name: c2r vs FFTW.jl"
    F(pre, :destroy_plan)(q)
  end
  # r2r with howmany
  for n in ([8], [5, 7]), howmany in (1, 3), inplace in (false, true)
    rank = length(n); N = prod(n)
    kinds = Cint[rand(ALLKINDS) for _ in 1:rank]
    name = "$pre many_r2r n=$n howmany=$howmany kinds=$kinds inplace=$inplace"
    x = randn(T, reverse(n)..., howmany)
    ref = r2r_ref(x, Tuple(reverse(Int.(kinds))), 1:rank)
    a = copy(x); b = inplace ? a : zeros(T, size(x))
    nn = Cint.(n)
    p = GC.@preserve a b nn kinds F(pre, :plan_many_r2r)(Cint(rank), pointer(nn), Cint(howmany),
          vp(a), Ptr{Cint}(C_NULL), Cint(1), Cint(N), vp(b), Ptr{Cint}(C_NULL), Cint(1), Cint(N),
          pointer(kinds), flag())
    @CHECK p != C_NULL, "$name: plan"
    p == C_NULL && continue
    GC.@preserve a b F(pre, :execute)(p)
    @CHECK close(b, ref, T), "$name: vs FFTW.jl"
    F(pre, :destroy_plan)(p)
  end
  # howmany = 0: a valid plan that computes nothing
  let x = randn(Complex{T}, 8), y = zeros(Complex{T}, 8), n = Cint[8]
    p = GC.@preserve x y n F(pre, :plan_many_dft)(Cint(1), pointer(n), Cint(0), vp(x), Ptr{Cint}(C_NULL),
          Cint(1), Cint(8), vp(y), Ptr{Cint}(C_NULL), Cint(1), Cint(8), Cint(-1), flag())
    @CHECK p != C_NULL, "$pre many_dft howmany = 0: plan"
    p != C_NULL && (GC.@preserve x y F(pre, :execute)(p); F(pre, :destroy_plan)(p))
    @CHECK all(iszero, y), "$pre many_dft howmany = 0: nothing written"
  end
end

# --- guru interface ---------------------------------------------------------------------------

jstrides(sz) = [prod(sz[1:d-1]; init = 1) for d in eachindex(sz)]

# A dense Julia array of size m; tj: Julia dimensions transformed, in FFTW's order (tj[end] is
# FFTW's last transform dimension, the halved one for r2c/c2r); the other dimensions go to the
# vector loop in random order.
function iodims(I, m, tj, istr, ostr)
  vj = shuffle(setdiff(1:length(m), tj))
  td = [I(m[d], istr[d], ostr[d]) for d in tj]
  vd = [I(m[d], istr[d], ostr[d]) for d in vj]
  td, vd
end

function test_guru(T, I, tag)
  pre = pfx(T)
  for trial in 1:12
    r = rand(2:4)
    m = rand([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16], r)
    k = rand(0:r)
    tj = shuffle(1:r)[1:k]
    # c2c, in place and out of place
    for sign in (-1, 1), inplace in (false, true)
      name = "$pre $tag dft m=$m tj=$tj sign=$sign inplace=$inplace"
      x = randn(Complex{T}, m...)
      st = jstrides(m)
      td, vd = iodims(I, m, tj, st, st)
      a = copy(x); b = inplace ? a : similar(x)
      p = GC.@preserve a b td vd F(pre, Symbol(:plan_, tag, :_dft))(Cint(k), pointer(td), Cint(length(vd)),
            pointer(vd), vp(a), vp(b), Cint(sign), flag())
      @CHECK p != C_NULL, "$name: plan"
      p == C_NULL && continue
      GC.@preserve a b F(pre, :execute)(p)
      @CHECK close(b, isempty(tj) ? x : cref(x, tj, sign), T), "$name: vs FFTW.jl"
      F(pre, :destroy_plan)(p)
    end
    # r2r with a kind per dimension
    for inplace in (false, true)
      kinds = Cint[m[d] == 1 ? rand(setdiff(ALLKINDS, (FFTW.REDFT00,))) : rand(ALLKINDS) for d in tj]
      name = "$pre $tag r2r m=$m tj=$tj kinds=$kinds inplace=$inplace"
      x = randn(T, m...)
      st = jstrides(m)
      td, vd = iodims(I, m, tj, st, st)
      a = copy(x); b = inplace ? a : similar(x)
      p = GC.@preserve a b td vd kinds F(pre, Symbol(:plan_, tag, :_r2r))(Cint(k), pointer(td), Cint(length(vd)),
            pointer(vd), vp(a), vp(b), pointer(kinds), flag())
      @CHECK p != C_NULL, "$name: plan"
      p == C_NULL && continue
      GC.@preserve a b F(pre, :execute)(p)
      ref = isempty(tj) ? x : r2r_ref(x, Tuple(Int.(kinds)), tj)
      @CHECK close(b, ref, T), "$name: vs FFTW.jl"
      F(pre, :destroy_plan)(p)
    end
    # r2c / c2r out of place, with the halved dimension the lowest transformed one (dense)
    k >= 1 || continue
    hj = minimum(tj)
    tjr = [setdiff(tj, hj); hj]
    mc = copy(m); mc[hj] = m[hj] ÷ 2 + 1
    rs = jstrides(m); cs = jstrides(mc)
    x = randn(T, m...)
    ref = FFTW.rfft(Float64.(x), [hj; setdiff(tj, hj)])
    td, vd = iodims(I, m, tjr, rs, cs)
    y = zeros(Complex{T}, mc...)
    name = "$pre $tag r2c m=$m tj=$tjr"
    p = GC.@preserve x y td vd F(pre, Symbol(:plan_, tag, :_dft_r2c))(Cint(k), pointer(td), Cint(length(vd)),
          pointer(vd), vp(x), vp(y), flag())
    @CHECK p != C_NULL, "$name: plan"
    if p != C_NULL
      GC.@preserve x y F(pre, :execute)(p)
      @CHECK close(y, ref, T), "$name: vs FFTW.jl"
      F(pre, :destroy_plan)(p)
    end
    c = Complex{T}.(ref); out = zeros(T, m...)
    tdc = [I(d.n, d.os, d.is) for d in td]; vdc = [I(d.n, d.os, d.is) for d in vd]
    q = GC.@preserve c out tdc vdc F(pre, Symbol(:plan_, tag, :_dft_c2r))(Cint(k), pointer(tdc), Cint(length(vdc)),
          pointer(vdc), vp(c), vp(out), flag())
    @CHECK q != C_NULL, "$pre $tag c2r m=$m tj=$tjr: plan"
    if q != C_NULL
      GC.@preserve c out F(pre, :execute)(q)
      @CHECK close(out, FFTW.brfft(ref, m[hj], [hj; setdiff(tj, hj)]), T), "$pre $tag c2r m=$m tj=$tjr: vs FFTW.jl"
      F(pre, :destroy_plan)(q)
    end
  end
end

# --- every layout: strided, embedded, split (the copy fallback) ------------------------------

# One side (input or output) of a transform in a user layout. kind: :real (real numbers),
# :inter (interleaved complex), :split (separate real and imaginary arrays), :splitinter
# (an interleaved array passed to the split interface: stride 2, imaginary part one real
# further). Strides and base are in elements of the side (complex numbers for :inter and
# :splitinter, real numbers otherwise); wstrides gives the strides FFTW sees.
struct Side{T}
  kind::Symbol
  bufs::Vector{Vector{T}}
  base::Int
  strides::Vector{Int}
  counts::Vector{Int}
end

cplx(s::Side) = s.kind != :real
wunit(s::Side) = s.kind == :splitinter ? 2 : 1
wstrides(s::Side) = wunit(s) .* s.strides
function ptrs(s::Side{T}) where {T}
  if s.kind == :real || s.kind == :split
    p1 = Ptr{Cvoid}(pointer(s.bufs[1])) + s.base * sizeof(T)
    p2 = s.kind == :split ? Ptr{Cvoid}(pointer(s.bufs[2])) + s.base * sizeof(T) : Ptr{Cvoid}(C_NULL)
  else
    p1 = Ptr{Cvoid}(pointer(s.bufs[1])) + 2s.base * sizeof(T)
    p2 = s.kind == :splitinter ? p1 + sizeof(T) : Ptr{Cvoid}(C_NULL)
  end
  p1, p2
end
offset(s::Side, I) = s.base + sum((I[d] - 1) * s.strides[d] for d in eachindex(s.strides); init = 0)
function getv(s::Side{T}, I) where {T}
  o = offset(s, I)
  s.kind == :real ? s.bufs[1][o + 1] :
  s.kind == :split ? Complex(s.bufs[1][o + 1], s.bufs[2][o + 1]) :
                     Complex(s.bufs[1][2o + 1], s.bufs[1][2o + 2])
end
function setv!(s::Side, I, v)
  o = offset(s, I)
  if s.kind == :real
    s.bufs[1][o + 1] = real(v)
  elseif s.kind == :split
    s.bufs[1][o + 1] = real(v); s.bufs[2][o + 1] = imag(v)
  else
    s.bufs[1][2o + 1] = real(v); s.bufs[1][2o + 2] = imag(v)
  end
end
logical(s::Side) = [getv(s, Tuple(I)) for I in CartesianIndices(Tuple(s.counts))]

# Random strides for counts: dimensions in a random order (first: the one at `first`, if
# given), a gap of 0-2 elements after each, some strides negative (neg). Returns the strides,
# the base offset (the position of element 0) and the number of elements spanned.
function rand_strides(counts::Vector{Int}; first::Int = 0, neg::Bool = false)
  k = length(counts)
  order = shuffle(1:k)
  first > 0 && (order = [first; setdiff(order, first)])
  st = zeros(Int, k); acc = 1
  for d in order
    st[d] = acc
    acc *= counts[d] + rand(0:2)
  end
  base = 0
  if neg
    for d in 1:k
      if rand() < 0.4
        base += (counts[d] - 1) * st[d]; st[d] = -st[d]
      end
    end
  end
  st, base, acc
end

newside(::Type{T}, kind, st, base, len, counts) where {T} =
  Side{T}(kind, kind == :split ? [randn(T, len), randn(T, len)] :
                kind == :real ? [randn(T, len)] : [randn(T, 2len)], base, st, counts)

# The transform of one random layout through the guru interface (I: WIodim or WIodim64):
# transform dimensions ns (FFTW's order; the last one is the halved one of r2c/c2r), vector
# dimensions nv, the sides' kinds ikind/okind. mode :oop (random order and gaps), :neg (also
# negative strides) or :inplace (one buffer: the same strides for c2c and r2r; for r2c/c2r
# FFTW's padded layout with the halved dimension fastest and gaps in the others).
function check_layout(T, I, typ::Symbol, ns::Vector{Int}, nv::Vector{Int}, ikind::Symbol,
                      okind::Symbol, mode::Symbol, name::String; newarrays::Bool = false)
  pre = pfx(T); k = length(ns)
  n = [ns; nv]
  h = k >= 1 ? ns[k] ÷ 2 + 1 : 1
  cn = copy(n); (typ == :r2c || typ == :c2r) && k >= 1 && (cn[k] = h)
  incnt = typ == :c2r ? cn : n
  outcnt = typ == :r2c ? cn : n
  if mode == :inplace
    if typ == :c2c || typ == :r2r
      st, base, len = rand_strides(n)
      isd = newside(T, ikind, st, base, len, incnt)
      osd = Side{T}(okind, isd.bufs, base, st, outcnt)
    else
      cst, _, len = rand_strides(cn; first = k)
      rst = [d == k ? 1 : 2cst[d] for d in eachindex(cst)]
      bufs = [randn(T, 2len)]
      rs = Side{T}(:real, bufs, 0, rst, n)
      cs = Side{T}(typ == :r2c ? okind : ikind, bufs, 0, cst, cn)
      isd, osd = typ == :r2c ? (rs, cs) : (cs, rs)
    end
  else
    ist, ib, il = rand_strides(incnt; neg = mode == :neg)
    ost, ob, ol = rand_strides(outcnt; neg = mode == :neg)
    isd = newside(T, ikind, ist, ib, il, incnt)
    osd = newside(T, okind, ost, ob, ol, outcnt)
  end
  kinds = Cint[ns[j] == 1 ? rand(setdiff(ALLKINDS, (FFTW.REDFT00,))) : rand(ALLKINDS) for j in 1:k]
  split = ikind in (:split, :splitinter) || okind in (:split, :splitinter)
  sfx = I == WIodim ? "guru_" : "guru64_"
  iws = wstrides(isd); ows = wstrides(osd)
  td = [I(n[j], iws[j], ows[j]) for j in 1:k]
  vd = [I(n[j], iws[j], ows[j]) for j in k+1:length(n)]
  plan = (i1, i2, o1, o2) -> GC.@preserve td vd kinds begin
    tdp = isempty(td) ? Ptr{I}(C_NULL) : pointer(td); vdp = isempty(vd) ? Ptr{I}(C_NULL) : pointer(vd)
    if typ == :c2c
      split ? F(pre, Symbol(:plan_, sfx, :split_dft))(Cint(k), tdp, Cint(length(vd)), vdp, i1, i2, o1, o2, flag()) :
              F(pre, Symbol(:plan_, sfx, :dft))(Cint(k), tdp, Cint(length(vd)), vdp, i1, o1, Cint(-1), flag())
    elseif typ == :r2c
      split ? F(pre, Symbol(:plan_, sfx, :split_dft_r2c))(Cint(k), tdp, Cint(length(vd)), vdp, i1, o1, o2, flag()) :
              F(pre, Symbol(:plan_, sfx, :dft_r2c))(Cint(k), tdp, Cint(length(vd)), vdp, i1, o1, flag())
    elseif typ == :c2r
      split ? F(pre, Symbol(:plan_, sfx, :split_dft_c2r))(Cint(k), tdp, Cint(length(vd)), vdp, i1, i2, o1, flag()) :
              F(pre, Symbol(:plan_, sfx, :dft_c2r))(Cint(k), tdp, Cint(length(vd)), vdp, i1, o1, flag())
    else
      F(pre, Symbol(:plan_, sfx, :r2r))(Cint(k), tdp, Cint(length(vd)), vdp, i1, o1, pointer(kinds), flag())
    end
  end
  verify_layout(T, typ, ns, nv, isd, osd, kinds, mode == :inplace, plan, name; newarrays = newarrays)
end

# Runs plan(i1, i2, o1, o2) (pointers of the sides; the imaginary parts for split sides) and
# checks the result against FFTW.jl on the logical arrays: the transform dimensions ns first
# (FFTW's order, the last one halved for r2c/c2r), then the vector dimensions nv. c2r: the input
# is replaced by the half spectrum of a real array.
function verify_layout(T, typ::Symbol, ns::Vector{Int}, nv::Vector{Int}, isd::Side, osd::Side,
                       kinds::Vector{Cint}, inplace::Bool, plan, name::String; newarrays::Bool = false)
  pre = pfx(T); k = length(ns); n = [ns; nv]
  split = isd.kind in (:split, :splitinter) || osd.kind in (:split, :splitinter)
  mode = inplace ? :inplace : :oop
  x = logical(isd)
  typ == :r2c || typ == :r2r ? (x = real.(x)) : nothing
  # c2r: the input must be the half spectrum of a real array
  if typ == :c2r
    k >= 1 || return
    xr = randn(Float64, n...)
    x = Complex{T}.(FFTW.rfft(xr, [k; 1:k-1]))
    for J in CartesianIndices(Tuple(isd.counts))
      setv!(isd, Tuple(J), x[J])
    end
  end
  ref = typ == :c2c ? (k == 0 ? ComplexF64.(x) : FFTW.fft(ComplexF64.(x), 1:k)) :
        typ == :r2c ? FFTW.rfft(Float64.(x), [k; 1:k-1]) :
        typ == :c2r ? FFTW.brfft(ComplexF64.(x), n[k], [k; 1:k-1]) :
                      (k == 0 ? Float64.(x) : FFTW.r2r(Float64.(x), Tuple(Int.(kinds)), 1:k))
  outbefore = [copy(b) for b in osd.bufs]
  GC.@preserve isd osd begin
    i1, i2 = ptrs(isd); o1, o2 = ptrs(osd)
    p = plan(i1, i2, o1, o2)
    @CHECK p != C_NULL, "$name: plan ($(lasterr()))"
    p == C_NULL && return
    if newarrays
      # the same layout in fresh buffers, through the new-array execute function
      isd2 = Side{T}(isd.kind, [copy(b) for b in isd.bufs], isd.base, isd.strides, isd.counts)
      osd2 = mode == :inplace ? Side{T}(osd.kind, isd2.bufs, osd.base, osd.strides, osd.counts) :
                                Side{T}(osd.kind, [copy(b) for b in osd.bufs], osd.base, osd.strides, osd.counts)
      GC.@preserve isd2 osd2 begin
        j1, j2 = ptrs(isd2); q1, q2 = ptrs(osd2)
        if typ == :c2c
          split ? F(pre, :execute_split_dft)(p, j1, j2, q1, q2) : F(pre, :execute_dft)(p, j1, q1)
        elseif typ == :r2c
          split ? F(pre, :execute_split_dft_r2c)(p, j1, q1, q2) : F(pre, :execute_dft_r2c)(p, j1, q1)
        elseif typ == :c2r
          split ? F(pre, :execute_split_dft_c2r)(p, j1, j2, q1) : F(pre, :execute_dft_c2r)(p, j1, q1)
        else
          F(pre, :execute_r2r)(p, j1, q1)
        end
      end
      osd = osd2           # its buffers started as copies of outbefore
    else
      F(pre, :execute)(p)
    end
    F(pre, :destroy_plan)(p)
  end
  y = logical(osd)
  got = typ == :c2r || typ == :r2r ? real.(y) : y
  @CHECK close(got, ref, T), "$name: vs FFTW.jl (rel err $(relerr(ComplexF64.(got), ComplexF64.(ref))))"
  # out of place: the gaps of the output keep their values
  if mode != :inplace
    touched = falses(sum(length, osd.bufs))
    lens = length(osd.bufs[1])
    for J in CartesianIndices(Tuple(osd.counts))
      o = offset(osd, Tuple(J))
      if osd.kind == :real
        touched[o + 1] = true
      elseif osd.kind == :split
        touched[o + 1] = true; touched[lens + o + 1] = true
      else
        touched[2o + 1] = true; touched[2o + 2] = true
      end
    end
    after = reduce(vcat, osd.bufs); before = reduce(vcat, outbefore)
    @CHECK after[.!touched] == before[.!touched], "$name: gaps of the output untouched"
  end
end

function test_layouts(T, I, tag)
  pre = pfx(T)
  sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16]
  for trial in 1:40
    k = rand(0:3); v = rand(0:2)
    ns = rand(sizes, k); nv = rand(sizes, v)
    for typ in (:c2c, :r2c, :c2r, :r2r)
      (typ == :r2c || typ == :c2r) && k == 0 && continue
      for mode in (:oop, :neg, :inplace)
        ck = rand([:inter, :split, :splitinter])
        ik, ok = typ == :c2c ? (ck, ck) : typ == :r2c ? (:real, ck) : typ == :c2r ? (ck, :real) : (:real, :real)
        # in place, a real array shares its storage with an interleaved complex one
        mode == :inplace && (typ == :r2c || typ == :c2r) && ck == :split && continue
        nw = rand(Bool)
        check_layout(T, I, typ, ns, nv, ik, ok, mode,
                     "$pre $tag layout $typ ns=$ns nv=$nv $ik→$ok $mode newarrays=$nw"; newarrays = nw)
      end
    end
  end
  # one copied plan on 8 tasks at once: every execution has its own buffer
  let n = Cint[48], x = zeros(Complex{T}, 3, 48), y = zeros(Complex{T}, 3, 48)
    p = GC.@preserve x y n F(pre, :plan_many_dft)(Cint(1), pointer(n), Cint(2), vp(x), Ptr{Cint}(C_NULL),
          Cint(3), Cint(1), vp(y), Ptr{Cint}(C_NULL), Cint(3), Cint(1), Cint(-1), flag())
    @CHECK occursin("copied", unsafe_string(F(pre, :sprint_plan)(p))), "$pre $tag stride 3: copy fallback"
    runs = [Tuple{Matrix{Complex{T}},Matrix{Complex{T}}}[] for _ in 1:8]
    tasks = map(1:8) do t
      Threads.@spawn for _ in 1:10
        local u = randn(Complex{T}, 3, 48)
        local v = zeros(Complex{T}, 3, 48)
        GC.@preserve u v F(pre, :execute_dft)(p, vp(u), vp(v))
        push!(runs[t], (u, v))
      end
    end
    foreach(wait, tasks)
    good = count(((u, v),) -> close(v[1:2, :], cref(u[1:2, :], 2, -1), T) && all(iszero, v[3, :]), reduce(vcat, runs))
    @CHECK good == 80, "$pre $tag one copied plan on 8 tasks: $good of 80 correct"
    F(pre, :destroy_plan)(p)
  end
end

# The advanced interface with embedded arrays: rank, n, howmany, inembed/onembed larger than
# the (real or complex) extents, strides and distances with gaps; out of place.
function test_many_layouts(T)
  pre = pfx(T)
  for trial in 1:30, typ in (:c2c, :r2c, :c2r, :r2r)
    rank = rand(1:3); ns = rand([1, 2, 3, 5, 6, 8, 9], rank); howmany = rand(1:3)
    h = ns[rank] ÷ 2 + 1
    cn = copy(ns); typ in (:r2c, :c2r) && (cn[rank] = h)
    icnt = typ == :c2r ? cn : ns; ocnt = typ == :r2c ? cn : ns
    emb(c) = [c[1]; [c[i] + rand(0:2) for i in 2:length(c)]]
    ie = emb(icnt); oe = emb(ocnt)
    is = rand(1:3); os = rand(1:3)
    ist = [is * prod(ie[j+1:end]; init = 1) for j in 1:rank]
    ost = [os * prod(oe[j+1:end]; init = 1) for j in 1:rank]
    idist = is * prod(ie) + rand(0:3); odist = os * prod(oe) + rand(0:3)
    ik = typ in (:c2c, :c2r) ? :inter : :real
    ok = typ in (:c2c, :r2c) ? :inter : :real
    il = idist * howmany; ol = odist * howmany
    isd = newside(T, ik, [ist; idist], 0, il, [icnt; howmany])
    osd = newside(T, ok, [ost; odist], 0, ol, [ocnt; howmany])
    kinds = Cint[ns[j] == 1 ? rand(setdiff(ALLKINDS, (FFTW.REDFT00,))) : rand(ALLKINDS) for j in 1:rank]
    nn = Cint.(ns); iec = Cint.(ie); oec = Cint.(oe)
    plan = (i1, i2, o1, o2) -> GC.@preserve nn iec oec kinds begin
      args = (Cint(rank), pointer(nn), Cint(howmany), i1, pointer(iec), Cint(is), Cint(idist),
              o1, pointer(oec), Cint(os), Cint(odist))
      typ == :c2c ? F(pre, :plan_many_dft)(args..., Cint(-1), flag()) :
      typ == :r2c ? F(pre, :plan_many_dft_r2c)(args..., flag()) :
      typ == :c2r ? F(pre, :plan_many_dft_c2r)(args..., flag()) :
                    F(pre, :plan_many_r2r)(args..., pointer(kinds), flag())
    end
    verify_layout(T, typ, ns, [howmany], isd, osd, kinds, false, plan,
                  "$pre many layout $typ n=$ns howmany=$howmany inembed=$ie onembed=$oe stride=$is,$os dist=$idist,$odist";
                  newarrays = rand(Bool))
  end
end

# --- new-array execute, concurrency, plan management ---------------------------------------

function test_execute(T)
  pre = pfx(T)
  n = Cint[6, 10]
  x = randn(Complex{T}, 10, 6); y = similar(x)
  p = GC.@preserve x y n F(pre, :plan_dft)(Cint(2), pointer(n), vp(x), vp(y), Cint(-1), FFTW.MEASURE)
  # new arrays, out of place; misaligned (by one element) arrays work as well
  for shift in (0, 1)
    u = randn(Complex{T}, 60 + shift); v = zeros(Complex{T}, 60 + shift)
    GC.@preserve u v F(pre, :execute_dft)(p, vp(u) + shift * sizeof(Complex{T}), vp(v) + shift * sizeof(Complex{T}))
    @CHECK close(v[1+shift:end], vec(cref(reshape(u[1+shift:end], 10, 6), 1:2, -1)), T), "$pre execute_dft shift=$shift"
  end
  # in-place arrays on an out-of-place plan: rejected, output untouched
  w = randn(Complex{T}, 10, 6); w0 = copy(w)
  GC.@preserve w F(pre, :execute_dft)(p, vp(w), vp(w))
  @CHECK w == w0 && occursin("in place", lasterr()), "$pre execute_dft in place on an out-of-place plan: rejected"
  # wrong execute function and wrong precision
  u = zeros(T, 200); v = zeros(T, 200)
  GC.@preserve u v F(pre, :execute_r2r)(p, vp(u), vp(v))
  @CHECK all(iszero, v) && occursin("transform type", lasterr()), "$pre execute_r2r on a c2c plan: rejected"
  other = T == Float64 ? "fftwf_" : "fftw_"
  F(other, :execute)(p)
  @CHECK occursin("invalid plan", lasterr()), "$pre plan executed with the other precision: rejected"
  GC.@preserve u v F(pre, :execute_split_dft)(p, vp(u), vp(u), vp(v), vp(v))
  @CHECK all(iszero, v) && occursin("split", lasterr()), "$pre execute_split_dft: rejected"
  # r2c new-array execute
  xr = randn(T, 11, 3); yr = zeros(Complex{T}, 6, 3); nr = Cint[3, 11]
  pr = GC.@preserve xr yr nr F(pre, :plan_dft_r2c)(Cint(2), pointer(nr), vp(xr), vp(yr), FFTW.ESTIMATE)
  xr2 = randn(T, 11, 3); yr2 = zeros(Complex{T}, 6, 3)
  GC.@preserve xr2 yr2 F(pre, :execute_dft_r2c)(pr, vp(xr2), vp(yr2))
  @CHECK close(yr2, FFTW.rfft(Float64.(xr2)), T), "$pre execute_dft_r2c"
  c2 = Complex{T}.(FFTW.rfft(Float64.(xr2))); o2 = zeros(T, 11, 3)
  pc = GC.@preserve c2 o2 nr F(pre, :plan_dft_c2r)(Cint(2), pointer(nr), vp(c2), vp(o2), FFTW.ESTIMATE)
  GC.@preserve c2 o2 F(pre, :execute_dft_c2r)(pc, vp(c2), vp(o2))
  @CHECK close(o2, 33 .* xr2, T), "$pre execute_dft_c2r"
  F(pre, :destroy_plan)(pr); F(pre, :destroy_plan)(pc)
  # r2r new-array execute
  xk = randn(T, 9); yk = zeros(T, 9); pk = GC.@preserve xk yk F(pre, :plan_r2r_1d)(Cint(9), vp(xk), vp(yk), Cint(FFTW.REDFT10), FFTW.ESTIMATE)
  xk2 = randn(T, 9); yk2 = zeros(T, 9)
  GC.@preserve xk2 yk2 F(pre, :execute_r2r)(pk, vp(xk2), vp(yk2))
  @CHECK close(yk2, r2r_ref(xk2, FFTW.REDFT10, 1), T), "$pre execute_r2r"
  F(pre, :destroy_plan)(pk)

  # the same plan on several threads at once, through the new-array execute function
  for nt in (1, 2)
    F(pre, :plan_with_nthreads)(Cint(nt))
    sz = Cint[64, 48]
    a0 = zeros(Complex{T}, 48, 64); b0 = similar(a0)
    q = GC.@preserve a0 b0 sz F(pre, :plan_dft)(Cint(2), pointer(sz), vp(a0), vp(b0), Cint(1), FFTW.ESTIMATE)
    ok = Threads.Atomic{Int}(0)
    # results are checked after all tasks have finished (FFTW.jl is not called concurrently)
    runs = [Tuple{Matrix{Complex{T}},Matrix{Complex{T}},Matrix{Complex{T}}}[] for _ in 1:8]
    tasks = map(1:8) do t
      Threads.@spawn for _ in 1:10
        # local: u and v name arrays of the enclosing function, which the tasks must not share
        local u = randn(Complex{T}, 48, 64)
        local v = similar(u)
        local u0 = copy(u)
        GC.@preserve u v F(pre, :execute_dft)(q, vp(u), vp(v))
        push!(runs[t], (u0, u, v))
      end
    end
    foreach(wait, tasks)
    for r in runs, (u0, u, v) in r
      u == u0 || println("  input changed")
      ok[] += close(v, cref(u0, 1:2, 1), T)
    end
    @CHECK ok[] == 80, "$pre one plan on 8 tasks (threads=$nt, Julia threads $(Threads.nthreads())): $(ok[]) of 80 correct"
    F(pre, :destroy_plan)(q)
  end
  F(pre, :plan_with_nthreads)(Cint(1))

  # copies and destruction
  @CHECK F(pre, :copy_plan)(p) == p, "$pre copy_plan returns the plan"
  F(pre, :destroy_plan)(p)
  GC.@preserve x y F(pre, :execute)(p)
  @CHECK close(y, cref(x, 1:2, -1), T), "$pre plan alive while a copy is"
  F(pre, :destroy_plan)(p)
  fill!(y, 0)
  GC.@preserve x y F(pre, :execute)(p)
  @CHECK all(iszero, y) && occursin("invalid plan", lasterr()), "$pre plan gone after both destroys"
  F(pre, :destroy_plan)(Ptr{Cvoid}(C_NULL))                     # no-op
  F(pre, :execute)(Ptr{Cvoid}(C_NULL))
  @CHECK occursin("invalid plan", lasterr()), "$pre execute(NULL): rejected"
end

# --- planner errors, flags, threads, memory, wisdom, information --------------------------

function test_misc(T)
  pre = pfx(T)
  x = zeros(Complex{T}, 64); y = zeros(Complex{T}, 64)
  GC.@preserve x y begin
    @CHECK F(pre, :plan_dft_1d)(Cint(16), vp(x), vp(y), Cint(-1), FFTW.WISDOM_ONLY | FFTW.PATIENT) == C_NULL,
      "$pre FFTW_WISDOM_ONLY: NULL"
    @CHECK occursin("WISDOM_ONLY", lasterr()), "$pre FFTW_WISDOM_ONLY: message"
    @CHECK F(pre, :plan_dft_1d)(Cint(0), vp(x), vp(y), Cint(-1), FFTW.ESTIMATE) == C_NULL, "$pre n = 0: NULL"
    @CHECK F(pre, :plan_dft_1d)(Cint(-3), vp(x), vp(y), Cint(-1), FFTW.ESTIMATE) == C_NULL, "$pre n < 0: NULL"
    @CHECK F(pre, :plan_dft_1d)(Cint(8), vp(x), vp(y), Cint(2), FFTW.ESTIMATE) == C_NULL, "$pre sign 2: NULL"
    @CHECK F(pre, :plan_dft)(Cint(-1), Ptr{Cint}(C_NULL), vp(x), vp(y), Cint(-1), FFTW.ESTIMATE) == C_NULL, "$pre rank < 0: NULL"
    kk = Cint[11]
    @CHECK F(pre, :plan_r2r_1d)(Cint(8), vp(x), vp(y), Cint(11), FFTW.ESTIMATE) == C_NULL, "$pre unknown kind: NULL"
    @CHECK F(pre, :plan_r2r_1d)(Cint(1), vp(x), vp(y), Cint(FFTW.REDFT00), FFTW.ESTIMATE) == C_NULL, "$pre DCT-I of length 1: NULL"
    # rank 0: a copy of one value
    x[1] = 3 - 2im
    p = F(pre, :plan_dft)(Cint(0), Ptr{Cint}(C_NULL), vp(x), vp(y), Cint(-1), FFTW.ESTIMATE)
    @CHECK p != C_NULL, "$pre rank 0: plan"
    p == C_NULL || (F(pre, :execute)(p); F(pre, :destroy_plan)(p))
    @CHECK y[1] == 3 - 2im && all(iszero, y[2:end]), "$pre rank 0: copies one value"
    xr = T[2.5, 7]; yc = zeros(Complex{T}, 2)
    p = GC.@preserve xr yc F(pre, :plan_dft_r2c)(Cint(0), Ptr{Cint}(C_NULL), vp(xr), vp(yc), FFTW.ESTIMATE)
    @CHECK p != C_NULL, "$pre r2c rank 0: plan"
    p == C_NULL || (GC.@preserve xr yc F(pre, :execute)(p); F(pre, :destroy_plan)(p))
    @CHECK yc == Complex{T}[2.5, 0], "$pre r2c rank 0: copies one value"
  end

  # threads
  @CHECK F(pre, :init_threads)() != 0, "$pre init_threads"
  F(pre, :plan_with_nthreads)(Cint(3))
  @CHECK F(pre, :planner_nthreads)() == 3, "$pre planner_nthreads"
  p = GC.@preserve x y F(pre, :plan_dft_1d)(Cint(64), vp(x), vp(y), Cint(-1), FFTW.ESTIMATE)
  s = F(pre, :sprint_plan)(p)
  @CHECK s != C_NULL && occursin("threads 3", unsafe_string(s)), "$pre plan takes planner_nthreads"
  s == C_NULL || Libc.free(s)
  F(pre, :plan_with_nthreads)(Cint(1))
  F(pre, :make_planner_thread_safe)()
  F(pre, :threads_set_callback)(C_NULL, C_NULL)

  # information
  add = Ref(-1.0); mul = Ref(-1.0); fma = Ref(-1.0)
  F(pre, :flops)(p, Base.unsafe_convert(Ptr{Cdouble}, add), Base.unsafe_convert(Ptr{Cdouble}, mul),
                 Base.unsafe_convert(Ptr{Cdouble}, fma))
  @CHECK add[] + mul[] ≈ 5 * 64 * 6 && fma[] == 0, "$pre flops: 5 N log2 N"
  @CHECK F(pre, :cost)(p) ≈ 5 * 64 * 6, "$pre cost"
  F(pre, :destroy_plan)(p)
  F(pre, :set_timelimit)(Cdouble(FFTW.NO_TIMELIMIT))
  F(pre, :set_timelimit)(Cdouble(0.5))

  # memory
  ps = [F(pre, :malloc)(Csize_t(n)) for n in (1, 24, 1000)]
  @CHECK all(q -> q != C_NULL && UInt(q) % 16 == 0, ps), "$pre malloc: 16-byte aligned"
  r = F(pre, :alloc_real)(Csize_t(100)); c = F(pre, :alloc_complex)(Csize_t(100))
  @CHECK r != C_NULL && c != C_NULL && UInt(r) % 16 == 0 && UInt(c) % 16 == 0, "$pre alloc_real/alloc_complex"
  unsafe_store!(Ptr{T}(r), T(1), 100); unsafe_store!(Ptr{T}(c), T(1), 200)        # last elements
  @CHECK F(pre, :alignment_of)(Ptr{T}(r)) == F(pre, :alignment_of)(Ptr{T}(c)), "$pre alignment_of"
  foreach(q -> F(pre, :free)(q), ps); F(pre, :free)(Ptr{Cvoid}(r)); F(pre, :free)(c)
  @CHECK F(pre, :alloc_real)(typemax(Csize_t) ÷ 2) == C_NULL, "$pre alloc_real overflow: NULL"

  # wisdom
  w = F(pre, :export_wisdom_to_string)()
  @CHECK w != C_NULL && !isempty(unsafe_string(w)), "$pre export_wisdom_to_string"
  @CHECK F(pre, :import_wisdom_from_string)(convert(Cstring, w)) == 1, "$pre import_wisdom_from_string of the export"
  Libc.free(w)
  fn = tempname()
  @CHECK cs(F(pre, :export_wisdom_to_filename), fn) == 1 && isfile(fn), "$pre export_wisdom_to_filename"
  @CHECK cs(F(pre, :import_wisdom_from_filename), fn) == 1, "$pre import_wisdom_from_filename"
  rm(fn)
  @CHECK cs(F(pre, :import_wisdom_from_filename), fn) == 0, "$pre import_wisdom_from_filename of a missing file: 0"
  @CHECK cs(F(pre, :export_wisdom_to_filename), "/nonexistent-dir/wisdom") == 0, "$pre export_wisdom_to_filename unwritable: 0"
  @CHECK F(pre, :import_system_wisdom)() == 0, "$pre import_system_wisdom"
  F(pre, :forget_wisdom)()
  F(pre, :cleanup)()
  F(pre, :cleanup_threads)()
end

# Export through the callback: collect the characters into a Julia vector
const WCHARS = UInt8[]
wcollect(c::Cchar, ::Ptr{Cvoid}) = (push!(WCHARS, c % UInt8); nothing)

for T in (Float64, Float32)
  pre = pfx(T)
  for n in ([1], [2], [7], [8], [12], [17], [64], [97], [100], [5, 6], [8, 8], [7, 13], [1, 9], [16, 3],
            [4, 6, 5], [3, 7, 2], [8, 1, 9], [2, 3, 2, 3])
    test_c2c_basic(T, n)
  end
  for n in ([1], [2], [7], [8], [17], [100], [5, 6], [8, 9], [7, 1], [3, 4, 5], [4, 5, 6])
    test_real_basic(T, n)
  end
  for k in ALLKINDS, n in (2, 5, 8, 9, 16, 17)
    test_r2r_basic(T, [n], [k])
  end
  for n in ([5, 8], [9, 4], [1, 6], [4, 6, 5], [3, 5, 7]), _ in 1:3
    test_r2r_basic(T, n, [n[i] == 1 ? rand(setdiff(ALLKINDS, (FFTW.REDFT00,))) : rand(ALLKINDS) for i in eachindex(n)])
  end
  test_many(T)
  test_guru(T, WIodim, :guru)
  test_guru(T, WIodim64, :guru64)
  test_layouts(T, WIodim, :guru)
  test_layouts(T, WIodim64, :guru64)
  test_many_layouts(T)
  test_execute(T)
  test_misc(T)
  empty!(WCHARS)
  F(pre, :export_wisdom)(@cfunction(wcollect, Cvoid, (Cchar, Ptr{Cvoid})), C_NULL)
  @CHECK String(copy(WCHARS)) == W_WISDOM, "$pre export_wisdom through the callback"
end

println("fftw3_jltest: $ntests checks, $nfailed failed (Julia threads: $(Threads.nthreads()))")
nfailed == 0 || exit(1)
