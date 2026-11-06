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

# --- every NULL a planner returns leaves a message; sizes, threads, the plan registry ----------

# The message of the last failed call, reset before f runs: f must return NULL and set one.
function null_with_message(f, pre, name)
  set_error!("")
  p = f()
  @CHECK p == C_NULL, "$pre $name: NULL"
  @CHECK !isempty(lasterr()), "$pre $name: message"
  p == C_NULL || F(pre, :destroy_plan)(p)
end

function test_errors(T)
  pre = pfx(T)
  x = zeros(Complex{T}, 64); y = zeros(Complex{T}, 64); r = zeros(T, 64)
  I = WIodim64
  n8 = Cint[8]; n0 = Cint[0]; d8 = [I(8, 1, 1)]; d0 = [I(0, 1, 1)]; d0s = [I(0, 2, 2)]; vneg = [I(-1, 8, 8)]
  w8 = [WIodim(8, 1, 1)]; wneg = [WIodim(-4, 1, 1)]; kneg = Cint[-1]
  GC.@preserve x y r n8 n0 d8 d0 d0s vneg w8 wneg kneg begin
    for (name, f) in (
        ("n = 0", () -> F(pre, :plan_dft_1d)(Cint(0), vp(x), vp(y), Cint(-1), FFTW.ESTIMATE)),
        ("n = -1", () -> F(pre, :plan_dft_1d)(Cint(-1), vp(x), vp(y), Cint(-1), FFTW.ESTIMATE)),
        ("second size 0", () -> F(pre, :plan_dft_2d)(Cint(8), Cint(0), vp(x), vp(y), Cint(-1), FFTW.ESTIMATE)),
        ("r2c n = 0", () -> F(pre, :plan_dft_r2c_1d)(Cint(0), vp(r), vp(y), FFTW.ESTIMATE)),
        ("c2r 3d size -2", () -> F(pre, :plan_dft_c2r_3d)(Cint(4), Cint(-2), Cint(4), vp(y), vp(r), FFTW.ESTIMATE)),
        ("r2r 2d size 0", () -> F(pre, :plan_r2r_2d)(Cint(0), Cint(4), vp(r), vp(r), Cint(0), Cint(0), FFTW.ESTIMATE)),
        ("sign 0", () -> F(pre, :plan_dft_1d)(Cint(8), vp(x), vp(y), Cint(0), FFTW.ESTIMATE)),
        ("sign 2", () -> F(pre, :plan_dft_1d)(Cint(8), vp(x), vp(y), Cint(2), FFTW.ESTIMATE)),
        ("sign -2 (many)", () -> (F(pre, :plan_many_dft)(Cint(1), pointer(n8), Cint(2), vp(x), Ptr{Cint}(C_NULL),
                                   Cint(1), Cint(8), vp(y), Ptr{Cint}(C_NULL), Cint(1), Cint(8), Cint(-2), FFTW.ESTIMATE))),
        ("rank -1", () -> F(pre, :plan_dft)(Cint(-1), Ptr{Cint}(C_NULL), vp(x), vp(y), Cint(-1), FFTW.ESTIMATE)),
        ("r2r rank -1", () -> F(pre, :plan_r2r)(Cint(-1), Ptr{Cint}(C_NULL), vp(r), vp(r), Ptr{Cint}(C_NULL), FFTW.ESTIMATE)),
        ("many howmany -1", () -> F(pre, :plan_many_dft)(Cint(1), pointer(n8), Cint(-1), vp(x), Ptr{Cint}(C_NULL),
                                   Cint(1), Cint(8), vp(y), Ptr{Cint}(C_NULL), Cint(1), Cint(8), Cint(-1), FFTW.ESTIMATE)),
        ("many size 0", () -> F(pre, :plan_many_dft)(Cint(1), pointer(n0), Cint(2), vp(x), Ptr{Cint}(C_NULL),
                               Cint(1), Cint(8), vp(y), Ptr{Cint}(C_NULL), Cint(1), Cint(8), Cint(-1), FFTW.ESTIMATE)),
        ("guru howmany_rank -1", () -> F(pre, :plan_guru64_dft)(Cint(1), pointer(d8), Cint(-1), Ptr{I}(C_NULL),
                                        vp(x), vp(y), Cint(-1), FFTW.ESTIMATE)),
        ("guru vector size -1", () -> F(pre, :plan_guru64_dft)(Cint(1), pointer(d8), Cint(1), pointer(vneg),
                                       vp(x), vp(y), Cint(-1), FFTW.ESTIMATE)),
        ("guru transform size 0", () -> F(pre, :plan_guru64_dft)(Cint(1), pointer(d0), Cint(0), Ptr{I}(C_NULL),
                                         vp(x), vp(y), Cint(-1), FFTW.ESTIMATE)),
        ("split guru size 0", () -> F(pre, :plan_guru64_split_dft)(Cint(1), pointer(d0s), Cint(0), Ptr{I}(C_NULL),
                                     vp(r), vp(r) + sizeof(T), vp(x), vp(x) + sizeof(T), FFTW.ESTIMATE)),
        ("split r2c rank -1", () -> F(pre, :plan_guru_split_dft_r2c)(Cint(-1), Ptr{WIodim}(C_NULL), Cint(0), Ptr{WIodim}(C_NULL),
                                     vp(r), vp(x), vp(y), FFTW.ESTIMATE)),
        ("split c2r unknown size", () -> F(pre, :plan_guru_split_dft_c2r)(Cint(1), pointer(wneg), Cint(0), Ptr{WIodim}(C_NULL),
                                          vp(x), vp(y), vp(r), FFTW.ESTIMATE)),
        ("unknown r2r kind 11", () -> F(pre, :plan_r2r_1d)(Cint(8), vp(r), vp(r), Cint(11), FFTW.ESTIMATE)),
        ("unknown r2r kind -1 (guru)", () -> F(pre, :plan_guru_r2r)(Cint(1), pointer(w8), Cint(0), Ptr{WIodim}(C_NULL),
                                              vp(r), vp(r), pointer(kneg), FFTW.ESTIMATE)),
        ("DCT-I of length 1", () -> F(pre, :plan_r2r_1d)(Cint(1), vp(r), vp(r), Cint(FFTW.REDFT00), FFTW.ESTIMATE)),
        ("FFTW_WISDOM_ONLY", () -> F(pre, :plan_dft_r2c_2d)(Cint(4), Cint(6), vp(r), vp(y), FFTW.WISDOM_ONLY)))
      null_with_message(f, pre, name)
    end
    # copy_plan of no plan
    set_error!("")
    @CHECK F(pre, :copy_plan)(Ptr{Cvoid}(UInt(12345))) == C_NULL && !isempty(lasterr()), "$pre copy_plan of no plan: NULL and message"
    set_error!("")
    @CHECK F(pre, :sprint_plan)(Ptr{Cvoid}(C_NULL)) == C_NULL && !isempty(lasterr()), "$pre sprint_plan(NULL): NULL and message"
    set_error!("")
    @CHECK F(pre, :alloc_complex)(typemax(Csize_t) ÷ 4) == C_NULL && !isempty(lasterr()), "$pre alloc_complex overflow: NULL and message"
  end

  # a message survives later successful calls
  GC.@preserve x y begin
    F(pre, :plan_dft_1d)(Cint(0), vp(x), vp(y), Cint(-1), FFTW.ESTIMATE)
    m = lasterr()
    p = F(pre, :plan_dft_1d)(Cint(8), vp(x), vp(y), Cint(-1), FFTW.ESTIMATE)
    GC.@preserve x y F(pre, :execute)(p)
    F(pre, :destroy_plan)(p)
    @CHECK lasterr() == m && !isempty(m), "$pre message survives successful calls"
  end

  # pointers that are no plans: rejected with a message, nothing written
  for (what, q) in (("NULL", C_NULL), ("12345", Ptr{Cvoid}(UInt(12345))), ("-1", Ptr{Cvoid}(typemax(UInt))),
                    ("an uncreated index", Ptr{Cvoid}(UInt((Int64(1) << 32) | 99999) + 1)),
                    ("a bad precision", Ptr{Cvoid}(UInt(Int64(3) << 32) + 1)), ("a bad type", Ptr{Cvoid}(UInt(Int64(9) << 40 | Int64(1) << 32) + 1)))
    set_error!("")
    GC.@preserve x y F(pre, :execute_dft)(q, vp(x), vp(y))
    @CHECK occursin("invalid plan", lasterr()) && all(iszero, y), "$pre execute_dft on $what: rejected"
    set_error!("")
    F(pre, :execute)(q)
    @CHECK occursin("invalid plan", lasterr()), "$pre execute on $what: rejected"
    set_error!("")
    F(pre, :destroy_plan)(q)
    @CHECK q == C_NULL ? isempty(lasterr()) : occursin("invalid plan", lasterr()), "$pre destroy_plan on $what"
  end
  # destroying twice: the second is rejected
  let p = GC.@preserve x y F(pre, :plan_dft_1d)(Cint(64), vp(x), vp(y), Cint(-1), FFTW.ESTIMATE)
    F(pre, :destroy_plan)(p)
    set_error!("")
    F(pre, :destroy_plan)(p)
    @CHECK occursin("invalid plan", lasterr()), "$pre second destroy_plan rejected"
  end

  # many plans of both precisions alive at once, each run while the others exist
  let ps = Ptr{Cvoid}[], xs = [randn(Complex{i % 2 == 0 ? Float64 : Float32}, 16 + i) for i in 1:64]
    ys = [similar(v) for v in xs]
    GC.@preserve xs ys for i in 1:64
      push!(ps, F(pfx(real(eltype(xs[i]))), :plan_dft_1d)(Cint(16 + i), vp(xs[i]), vp(ys[i]), Cint(-1), FFTW.ESTIMATE))
    end
    @CHECK all(!=(C_NULL), ps), "$pre 64 plans of both precisions created"
    ok = 0
    GC.@preserve xs ys for i in 64:-1:1
      S = real(eltype(xs[i]))
      F(pfx(S), :execute)(ps[i])
      ok += close(ys[i], FFTW.fft(ComplexF64.(xs[i])), S)
      F(pfx(S), :destroy_plan)(ps[i])
    end
    @CHECK ok == 64, "$pre 64 plans executed: $ok correct"
  end
end

# Large transforms with 1 and 4 plan threads (the four-step path for one long line, lane
# blocks, Bluestein), against FFTW.jl; the two thread counts must agree.
function test_threads_sizes(T)
  pre = pfx(T)
  for (n, what) in (([131072], "1D 2^17 (four-step)"), ([65536], "1D 2^16"), ([1024, 33], "2D 1024 x 33"),
                    ([16, 16, 16], "3D 16^3"), ([9973], "1D prime 9973 (Bluestein)"), ([37, 5], "2D 37 x 5 (Bluestein)"))
    x = randn(Complex{T}, reverse(n)...)
    outs = map((1, 4)) do nt
      F(pre, :plan_with_nthreads)(Cint(nt))
      y = similar(x); nn = Cint.(n)
      p = GC.@preserve x y nn F(pre, :plan_dft)(Cint(length(n)), pointer(nn), vp(x), vp(y), Cint(-1), FFTW.ESTIMATE)
      GC.@preserve x y F(pre, :execute)(p)
      F(pre, :destroy_plan)(p)
      @CHECK close(y, FFTW.fft(ComplexF64.(x)), T), "$pre $what threads $nt: vs FFTW.jl"
      y
    end
    @CHECK relerr(ComplexF64.(outs[1]), ComplexF64.(outs[2])) < (T == Float64 ? 1e-13 : 1e-5), "$pre $what: threads 4 agrees with threads 1"
  end
  # 64 x 64 x 8, transformed along the two fastest dimensions, threads 4 (lane blocks)
  let x = randn(Complex{T}, 64, 64, 8), y = similar(x)
    F(pre, :plan_with_nthreads)(Cint(4))
    d = [WIodim(64, 64, 64), WIodim(64, 1, 1)]; v = [WIodim(8, 4096, 4096)]
    p = GC.@preserve x y d v F(pre, :plan_guru_dft)(Cint(2), pointer(d), Cint(1), pointer(v), vp(x), vp(y), Cint(-1), FFTW.ESTIMATE)
    GC.@preserve x y F(pre, :execute)(p)
    F(pre, :destroy_plan)(p)
    @CHECK close(y, FFTW.fft(ComplexF64.(x), 1:2), T), "$pre 64 x 64 x 8 along two dimensions, threads 4"
  end
  # real transforms: r2c/c2r with prime and mixed sizes, threads 1 and 4, c2r keeps its input
  for n in ([101], [7, 101], [12, 10], [9, 8, 7]), nt in (1, 4)
    F(pre, :plan_with_nthreads)(Cint(nt))
    jn = reverse(n); x = randn(T, jn...); h = jn[1] ÷ 2 + 1
    y = zeros(Complex{T}, h, jn[2:end]...); z = zeros(T, jn...); nn = Cint.(n)
    pf = GC.@preserve x y nn F(pre, :plan_dft_r2c)(Cint(length(n)), pointer(nn), vp(x), vp(y), FFTW.ESTIMATE)
    pb = GC.@preserve y z nn F(pre, :plan_dft_c2r)(Cint(length(n)), pointer(nn), vp(y), vp(z), FFTW.ESTIMATE)
    GC.@preserve x y F(pre, :execute)(pf)
    @CHECK close(y, FFTW.rfft(Float64.(x)), T), "$pre r2c $n threads $nt"
    y0 = copy(y)
    GC.@preserve y z F(pre, :execute)(pb)
    @CHECK y == y0, "$pre c2r $n threads $nt keeps its input"
    @CHECK close(z ./ prod(n), x, T), "$pre c2r $n threads $nt round trip"
    F(pre, :destroy_plan)(pf); F(pre, :destroy_plan)(pb)
  end
  # r2r: every kind at lengths 33 and 101; in place equals out of place bit for bit
  F(pre, :plan_with_nthreads)(Cint(1))
  for k in ALLKINDS, n in (33, 101)
    x = randn(T, n); y = zeros(T, n); z = copy(x)
    p = GC.@preserve x y F(pre, :plan_r2r_1d)(Cint(n), vp(x), vp(y), Cint(k), FFTW.ESTIMATE)
    q = GC.@preserve z F(pre, :plan_r2r_1d)(Cint(n), vp(z), vp(z), Cint(k), FFTW.ESTIMATE)
    GC.@preserve x y z (F(pre, :execute)(p); F(pre, :execute)(q))
    @CHECK close(y, r2r_ref(x, k, 1), T), "$pre r2r kind $k n $n"
    @CHECK y == z, "$pre r2r kind $k n $n: in place = out of place"
    F(pre, :destroy_plan)(p); F(pre, :destroy_plan)(q)
  end
  # three dimensions, one kind each, threads 4
  let x = randn(T, 9, 12, 5), y = similar(x), nn = Cint[5, 12, 9], kk = Cint[FFTW.RODFT11, FFTW.REDFT00, FFTW.HC2R]
    F(pre, :plan_with_nthreads)(Cint(4))
    p = GC.@preserve x y nn kk F(pre, :plan_r2r)(Cint(3), pointer(nn), vp(x), vp(y), pointer(kk), FFTW.ESTIMATE)
    GC.@preserve x y F(pre, :execute)(p)
    F(pre, :destroy_plan)(p)
    @CHECK close(y, r2r_ref(x, (FFTW.HC2R, FFTW.REDFT00, FFTW.RODFT11), 1:3), T), "$pre r2r 3D mixed kinds threads 4"
  end
  F(pre, :plan_with_nthreads)(Cint(0))
  @CHECK F(pre, :planner_nthreads)() == 1, "$pre plan_with_nthreads(0) gives 1"
  F(pre, :plan_with_nthreads)(Cint(-2))
  @CHECK F(pre, :planner_nthreads)() == 1, "$pre plan_with_nthreads(-2) gives 1"
end

# fftl_version and fftl_last_error, the two functions of fftl.h
function test_fftl_h()
  buf = zeros(UInt8, 64)
  n = fftl_version(pointer(buf), Csize_t(length(buf)))
  @CHECK n == 5 && String(buf[1:6]) == "0.1.0\0", "fftl_version 0.1.0"
  fill!(buf, 0)
  n = fftl_version(pointer(buf), Csize_t(3))
  @CHECK n == 5 && buf[1:3] == UInt8['0', '.', 0], "fftl_version truncates like snprintf"
  @CHECK fftl_version(Ptr{UInt8}(C_NULL), Csize_t(0)) == 5, "fftl_version(NULL, 0): the length"
  set_error!("a message of 22 bytes!")
  fill!(buf, 0xff)
  @CHECK fftl_last_error(Ptr{UInt8}(C_NULL), Csize_t(0)) == 22, "fftl_last_error(NULL, 0): the length"
  @CHECK fftl_last_error(pointer(buf), Csize_t(5)) == 22 && buf[1:5] == UInt8['a', ' ', 'm', 'e', 0], "fftl_last_error truncates like snprintf"
  # per thread: a message set on another thread is not seen here
  t = Threads.@spawn (set_error!("other thread"); Threads.threadid())
  tid = fetch(t)
  @CHECK tid == Threads.threadid() || lasterr() == "a message of 22 bytes!", "fftl_last_error is per thread"
end

# --- the legacy Fortran entry points (by reference, dimensions in Fortran order) ------------

function test_fortran_legacy(T)
  pre = T == Float64 ? "dfftw_" : "sfftw_"
  L(name, suffix = "_") = getfield(Main, Symbol(pre, name, suffix))
  ri(v) = Ref(Cint(v))
  for suffix in ("_", "__")
    plan = Ref(Int64(0)); n0 = ri(5); n1 = ri(6); sg = ri(-1); fl = ri(Cint(FFTW.ESTIMATE))
    x = randn(Complex{T}, 5, 6); y = similar(x)
    GC.@preserve x y plan n0 n1 sg fl begin
      L(:plan_dft_2d, suffix)(Base.unsafe_convert(Ptr{Int64}, plan), Base.unsafe_convert(Ptr{Cint}, n0),
          Base.unsafe_convert(Ptr{Cint}, n1), vp(x), vp(y), Base.unsafe_convert(Ptr{Cint}, sg),
          Base.unsafe_convert(Ptr{Cint}, fl))
    end
    @CHECK plan[] != 0, "$pre plan_dft_2d$suffix: plan"
    GC.@preserve x y plan L(:execute_dft, suffix)(Base.unsafe_convert(Ptr{Int64}, plan), vp(x), vp(y))
    @CHECK close(y, FFTW.fft(ComplexF64.(x)), T), "$pre plan_dft_2d$suffix: Fortran order vs FFTW.jl"
    GC.@preserve plan L(:destroy_plan, suffix)(Base.unsafe_convert(Ptr{Int64}, plan))
  end
  # r2r_2d: kind0 along the first Fortran dimension; guru_r2r: the kind array reversed
  let x = randn(T, 5, 6), y = similar(x), plan = Ref(Int64(0)), k0 = ri(FFTW.REDFT10), k1 = ri(FFTW.RODFT11)
    n0 = ri(5); n1 = ri(6); fl = ri(Cint(FFTW.ESTIMATE))
    GC.@preserve x y plan n0 n1 k0 k1 fl L(:plan_r2r_2d)(Base.unsafe_convert(Ptr{Int64}, plan), Base.unsafe_convert(Ptr{Cint}, n0),
        Base.unsafe_convert(Ptr{Cint}, n1), vp(x), vp(y), Base.unsafe_convert(Ptr{Cint}, k0), Base.unsafe_convert(Ptr{Cint}, k1),
        Base.unsafe_convert(Ptr{Cint}, fl))
    GC.@preserve x y plan L(:execute_r2r)(Base.unsafe_convert(Ptr{Int64}, plan), vp(x), vp(y))
    @CHECK close(y, r2r_ref(x, (FFTW.REDFT10, FFTW.RODFT11), 1:2), T), "$pre plan_r2r_2d: kinds in Fortran order"
    GC.@preserve plan L(:destroy_plan)(Base.unsafe_convert(Ptr{Int64}, plan))
    nn = Cint[6, 5]; is = Cint[5, 1]; ks = Cint[FFTW.DHT, FFTW.REDFT01]; rank = ri(2); h = ri(0)
    GC.@preserve x y plan nn is ks rank h fl L(:plan_guru_r2r)(Base.unsafe_convert(Ptr{Int64}, plan), Base.unsafe_convert(Ptr{Cint}, rank),
        pointer(nn), pointer(is), pointer(is), Base.unsafe_convert(Ptr{Cint}, h), pointer(nn), pointer(is), pointer(is),
        vp(x), vp(y), pointer(ks), Base.unsafe_convert(Ptr{Cint}, fl))
    GC.@preserve x y plan L(:execute_r2r)(Base.unsafe_convert(Ptr{Int64}, plan), vp(x), vp(y))
    @CHECK close(y, r2r_ref(x, (FFTW.DHT, FFTW.REDFT01), 1:2), T), "$pre plan_guru_r2r: kind array reversed"
    GC.@preserve plan L(:destroy_plan)(Base.unsafe_convert(Ptr{Int64}, plan))
  end
  # a failed plan is 0, with a message
  let x = zeros(Complex{T}, 4), plan = Ref(Int64(99)), n = ri(0), s = ri(-1), fl = ri(0)
    set_error!("")
    GC.@preserve x plan n s fl L(:plan_dft_1d)(Base.unsafe_convert(Ptr{Int64}, plan), Base.unsafe_convert(Ptr{Cint}, n), vp(x), vp(x),
        Base.unsafe_convert(Ptr{Cint}, s), Base.unsafe_convert(Ptr{Cint}, fl))
    @CHECK plan[] == 0 && !isempty(lasterr()), "$pre plan_dft_1d n = 0: plan 0 and message"
  end
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
  test_errors(T)
  test_threads_sizes(T)
  test_fortran_legacy(T)
  empty!(WCHARS)
  F(pre, :export_wisdom)(@cfunction(wcollect, Cvoid, (Cchar, Ptr{Cvoid})), C_NULL)
  @CHECK String(copy(WCHARS)) == W_WISDOM, "$pre export_wisdom through the callback"
end

test_fftl_h()

println("fftw3_jltest: $ntests checks, $nfailed failed (Julia threads: $(Threads.nthreads()))")
nfailed == 0 || exit(1)
