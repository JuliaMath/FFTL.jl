# Real-to-real transforms (unnormalised), along each dimension of a region with its own kind:
#   REDFT00  Y_k = X_0 + (-1)^k X_(n-1) + 2 Σ_(j=1)^(n-2) X_j cos(πjk/(n-1))      (DCT-I, n ≥ 2)
#   REDFT10  Y_k = 2 Σ X_j cos(π(j+½)k/n)                                          (DCT-II)
#   REDFT01  Y_k = X_0 + 2 Σ_(j≥1) X_j cos(πj(k+½)/n)                              (DCT-III)
#   REDFT11  Y_k = 2 Σ X_j cos(π(j+½)(k+½)/n)                                      (DCT-IV)
#   RODFT00  Y_k = 2 Σ X_j sin(π(j+1)(k+1)/(n+1))                                  (DST-I)
#   RODFT10  Y_k = 2 Σ X_j sin(π(j+½)(k+1)/n)                                      (DST-II)
#   RODFT01  Y_k = (-1)^k X_(n-1) + 2 Σ_(j≤n-2) X_j sin(π(j+1)(k+½)/n)              (DST-III)
#   RODFT11  Y_k = 2 Σ X_j sin(π(j+½)(k+½)/n)                                      (DST-IV)
#   DHT      Y_k = Σ X_j (cos(2πjk/n) + sin(2πjk/n))                               (Hartley)
#
# Each kind is computed from a real or complex DFT:
# - DCT-II: v = (x_0, x_2, x_4, …, x_5, x_3, x_1) (even indices forward, odd backward), V = DFT(v)
#   (r2c), U_k = e^(-iπk/(2n))·V_k; then Y_k = 2 Re U_k, Y_(n-k) = -2 Im U_k.
# - DCT-III: W_k = e^(iπk/(2n))·(X_k - i X_(n-k)) (X_n = 0), w = c2r(W) of length n, and
#   Y_(2j) = w_j, Y_(2j+1) = w_(n-1-j).
# - DCT-IV, n even: v_j = (x_(2j) - i x_(n-1-2j))·e^(iπj/n), V = backward DFT of length n/2,
#   W_m = e^(iπ(4m+1)/(4n))·V_m; Y_(2m) = 2 Re W_m, Y_(n-1-2m) = 2 Im W_m.
#   n odd: Y_k = 2 Re[e^(-iπ(2k+1)/(4n)) DFT_(2n)(x_j e^(-iπj/(2n)), zero padded)_k].
# - DST-II/IV: DCT-II/IV of ((-1)^j x_j), output reversed. DST-III: (-1)^k DCT-III(reversed x)_k.
# - DCT-I, N = n - 1 (points 0…N): with a_j = x_j + x_(N-j), b_j = x_j - x_(N-j),
#   y_j = a_j/2 - sin(πj/N)·b_j (j < N) and F = r2c(y) of length N:
#   Y_2k = 2 Re F_k, Y_1 = Σ_j b_j cos(πj/N), Y_(2k+1) = Y_(2k-1) - 2 Im F_k.
# - DST-I, N = n + 1 (points 1…N-1, x_0 = x_N = 0): y_j = sin(πj/N)·a_j + b_j/2, F = r2c(y):
#   Y_2k = -2 Im F_k, Y_1 = Re F_0, Y_(2k+1) = Y_(2k-1) + 2 Re F_k.
# - DHT: F = r2c(x); Y_k = Re F_k - Im F_k, Y_(n-k) = Re F_k + Im F_k.

@enum R2RKind REDFT00 REDFT10 REDFT01 REDFT11 RODFT00 RODFT10 RODFT01 RODFT11 DHT

"""
    R2RLine{T}

Kernel of one real-to-real kind on contiguous lines of length `n`.
"""
struct R2RLine{T,K}
  kind::R2RKind
  n::Int
  inner::K                       # RealLine or complex Kernel, see above
  tw::Vector{Complex{T}}         # pre/post twiddles of the kind
  tw2::Vector{Complex{T}}
end

function R2RLine{T}(kind::R2RKind, n::Int) where {T}
  n >= 1 || throw(ArgumentError("length must be positive"))
  kind == REDFT00 && n < 2 && throw(ArgumentError("REDFT00 needs length ≥ 2"))
  base = kind in (RODFT10,) ? REDFT10 : kind == RODFT01 ? REDFT01 : kind == RODFT11 ? REDFT11 : kind
  tw = Complex{T}[]; tw2 = Complex{T}[]
  if base == REDFT10
    inner = RealLine{T}(n, -1)
    tw = [Complex{T}(cispi(-k / (2n))) for k in 0:n÷2]
  elseif base == REDFT01
    inner = RealLine{T}(n, 1)
    tw = [Complex{T}(cispi(k / (2n))) for k in 0:n÷2]
  elseif base == REDFT11
    if iseven(n)
      inner = kernel(T, n ÷ 2, 1)
      tw = [Complex{T}(cispi(j / n)) for j in 0:n÷2-1]
      tw2 = [Complex{T}(cispi((4m + 1) / (4n))) for m in 0:n÷2-1]
    else
      inner = kernel(T, 2n, -1)
      tw = [Complex{T}(cispi(-j / (2n))) for j in 0:n-1]
      tw2 = [Complex{T}(cispi(-(2k + 1) / (4n))) for k in 0:n-1]
    end
  elseif base == REDFT00
    N = n - 1
    inner = RealLine{T}(N, -1)
    tw = [Complex{T}(cospi(j / N), sinpi(j / N)) for j in 0:N-1]
  elseif base == RODFT00
    N = n + 1
    inner = RealLine{T}(N, -1)
    tw = [Complex{T}(cospi(j / N), sinpi(j / N)) for j in 0:N-1]
  else # DHT
    inner = RealLine{T}(n, -1)
  end
  R2RLine{T,typeof(inner)}(kind, n, inner, tw, tw2)
end

"""
    R2RBuffers{T}

Scratch space of one task for lines of length up to `n`.
"""
struct R2RBuffers{T}
  r::Vector{T}
  c::Vector{Complex{T}}
  b1::Vector{Complex{T}}
  b2::Vector{Complex{T}}
  b3::Vector{Complex{T}}
end
R2RBuffers{T}(n) where {T} = R2RBuffers{T}(Vector{T}(undef, 2n + 4), Vector{Complex{T}}(undef, 2n + 4),
  Vector{Complex{T}}(undef, 2n + 4), Vector{Complex{T}}(undef, 2n + 4), Vector{Complex{T}}(undef, 2n + 4))

r2c!(c::Vector{Complex{T}}, r::Vector{T}, K::RealLine{T}, B) where {T} =
  GC.@preserve c r B r2c_line!(pointer(c), pointer(r), cptr(B.b1), cptr(B.b2), cptr(B.b3), K)
c2r!(r::Vector{T}, c::Vector{Complex{T}}, K::RealLine{T}, B) where {T} =
  GC.@preserve c r B c2r_line!(pointer(r), pointer(c), cptr(B.b1), cptr(B.b2), cptr(B.b3), K)

# y = transform of x along a line of length n (y may be the same line as x: each kind reads its
# whole input first). One function per family: closures in a shared function would capture
# variables assigned in several branches, which Julia boxes.
function r2r_line!(y::AbstractVector{T}, x::AbstractVector{T}, K::R2RLine{T}, B::R2RBuffers{T}) where {T}
  kind = K.kind
  if kind == REDFT10 || kind == RODFT10
    dct2!(y, x, K, B, kind == RODFT10)
  elseif kind == REDFT01 || kind == RODFT01
    dct3!(y, x, K, B, kind == RODFT01)
  elseif kind == REDFT11 || kind == RODFT11
    dct4!(y, x, K, B, kind == RODFT11)
  elseif kind == REDFT00
    dct1!(y, x, K, B)
  elseif kind == RODFT00
    dst1!(y, x, K, B)
  else
    dht!(y, x, K, B)
  end
  y
end

# DCT-II; with `dst` the DST-II as DCT-II of ((-1)^j x_j), output reversed
function dct2!(y, x, K::R2RLine{T}, B, dst::Bool) where {T}
  n = K.n; r = B.r; c = B.c
  @inbounds for j in 0:(n-1)÷2
    r[j+1] = x[2j+1]
  end
  @inbounds for j in 0:n÷2-1
    v = x[2j+2]; r[n-j] = dst ? -v : v
  end
  r2c!(c, r, K.inner, B)
  @inbounds for k in 0:n÷2
    u = K.tw[k+1] * c[k+1]
    y[(dst ? n - 1 - k : k) + 1] = 2real(u)
    if 0 < k < n - k
      y[(dst ? k - 1 : n - k) + 1] = -2imag(u)
    end
  end
  y
end

# DCT-III; with `dst` the DST-III as (-1)^k DCT-III(reversed x)_k
function dct3!(y, x, K::R2RLine{T}, B, dst::Bool) where {T}
  n = K.n; r = B.r; c = B.c
  @inbounds for k in 0:n÷2
    xk = dst ? x[n-k] : x[k+1]
    xm = k == 0 ? zero(T) : (dst ? x[k] : x[n-k+1])
    c[k+1] = K.tw[k+1] * Complex(xk, -xm)
  end
  c2r!(r, c, K.inner, B)
  @inbounds for j in 0:(n-1)÷2
    y[2j+1] = r[j+1]
  end
  @inbounds for j in 0:n÷2-1
    y[2j+2] = dst ? -r[n-j] : r[n-j]
  end
  y
end

# DCT-IV; with `dst` the DST-IV as DCT-IV of ((-1)^j x_j), output reversed
function dct4!(y, x, K::R2RLine{T}, B, dst::Bool) where {T}
  n = K.n; b = B.b1
  if iseven(n)
    M = n ÷ 2
    @inbounds for j in 0:M-1
      xe = x[2j+1]; xo = x[n-2j]                          # x_(2j) and x_(n-1-2j) (odd index)
      b[j+1] = Complex(xe, dst ? xo : -xo) * K.tw[j+1]
    end
    GC.@preserve b B run!(cptr(b), cptr(b), cptr(B.b2), cptr(B.b3), K.inner, 1)
    @inbounds for m in 0:M-1
      w = K.tw2[m+1] * b[m+1]
      k1 = 2m; k2 = n - 1 - 2m
      y[(dst ? n - 1 - k1 : k1) + 1] = 2real(w)
      y[(dst ? n - 1 - k2 : k2) + 1] = 2imag(w)
    end
  else
    @inbounds for j in 0:n-1
      v = x[j+1]
      b[j+1] = (dst && isodd(j) ? -v : v) * K.tw[j+1]
    end
    @inbounds for j in n:2n-1
      b[j+1] = zero(Complex{T})
    end
    GC.@preserve b B run!(cptr(b), cptr(b), cptr(B.b2), cptr(B.b3), K.inner, 1)
    @inbounds for k in 0:n-1
      y[(dst ? n - 1 - k : k) + 1] = 2real(K.tw2[k+1] * b[k+1])
    end
  end
  y
end

# The pair (j, N-j) gives both r_j and r_(N-j) from x_j, x_(N-j): with a = x_j + x_(N-j),
# b = x_j - x_(N-j) and s = sin(πj/N): r_j = a/2 - s·b, r_(N-j) = a/2 + s·b (DCT-I), and
# r_j = s·a + b/2, r_(N-j) = s·a - b/2 (DST-I).
function dct1!(y, x, K::R2RLine{T}, B) where {T}
  N = K.n - 1; r = B.r; c = B.c
  y1a = zero(T); y1b = zero(T)
  @inbounds begin
    r[1] = (x[1] + x[N+1]) / 2
    y1a = x[1] - x[N+1]
    for j in 1:(N-1)÷2
      xa = x[j+1]; xb = x[N-j+1]
      a = (xa + xb) / 2; bj = xa - xb; sj = imag(K.tw[j+1])
      r[j+1] = a - sj * bj
      r[N-j+1] = a + sj * bj
      y1b += 2 * bj * real(K.tw[j+1])      # b_(N-j)·cos(π(N-j)/N) = b_j·cos(πj/N)
    end
    iseven(N) && (r[N÷2+1] = x[N÷2+1])     # j = N/2: b = 0
  end
  r2c!(c, r, K.inner, B)
  @inbounds begin
    for k in 0:N÷2
      y[2k+1] = 2real(c[k+1])
    end
    acc = y1a + y1b
    y[2] = acc
    for k in 1:(N-1)÷2
      acc -= 2imag(c[k+1])
      y[2k+2] = acc
    end
  end
  y
end

function dst1!(y, x, K::R2RLine{T}, B) where {T}
  n = K.n; N = n + 1; r = B.r; c = B.c
  @inbounds begin
    r[1] = zero(T)                                        # points 0 and N are zero
    for j in 1:(N-1)÷2
      xa = x[j]; xb = x[N-j]                              # points j and N-j are x_(j-1), x_(N-j-1)
      sa = imag(K.tw[j+1]) * (xa + xb); hb = (xa - xb) / 2
      r[j+1] = sa + hb
      r[N-j+1] = sa - hb
    end
    iseven(N) && (r[N÷2+1] = 2 * x[N÷2])                 # j = N/2: sin = 1, b = 0
  end
  r2c!(c, r, K.inner, B)
  @inbounds begin
    acc = real(c[1])
    y[1] = acc
    for k in 1:N÷2
      2k <= n && (y[2k] = -2imag(c[k+1]))
      if 2k + 1 <= n
        acc += 2real(c[k+1])
        y[2k+1] = acc
      end
    end
  end
  y
end

function dht!(y, x, K::R2RLine{T}, B) where {T}
  n = K.n; r = B.r; c = B.c
  @inbounds for j in 1:n
    r[j] = x[j]
  end
  r2c!(c, r, K.inner, B)
  @inbounds for k in 0:n÷2
    f = c[k+1]
    y[k+1] = real(f) - imag(f)
    0 < k < n - k && (y[n-k+1] = real(f) + imag(f))
  end
  y
end

"""
    R2RPlan{T,N}

Plan for real-to-real transforms of arrays of size `sz`, one kind per dimension of `region`.
"""
mutable struct R2RPlan{T<:FFTLFloat,N,L<:Tuple,D} <: AbstractFFTs.Plan{T}
  sz::NTuple{N,Int}
  region::NTuple{D,Int}
  kinds::NTuple{D,R2RKind}
  lines::L
  nthreads::Int
  bufs::Vector{R2RBuffers{T}}
  pinv::AbstractFFTs.Plan
  R2RPlan{T,N,L,D}(sz, region, kinds, lines, nt, bufs) where {T,N,L,D} = new{T,N,L,D}(sz, region, kinds, lines, nt, bufs)
end

function R2RPlan{T}(sz::NTuple{N,Int}, kinds, region; threads::Integer = Threads.nthreads()) where {T<:FFTLFloat,N}
  reg = Tuple(collect(Int, region isa Integer ? (region,) : region))
  allunique(reg) && all(d -> 1 <= d <= N, reg) || throw(ArgumentError("region $region is not within 1:$N"))
  ks = kinds isa R2RKind ? ntuple(_ -> kinds, length(reg)) : Tuple(kinds)
  length(ks) == length(reg) || throw(ArgumentError("one kind per transformed dimension"))
  lines = Tuple(R2RLine{T}(k, sz[d]) for (k, d) in zip(ks, reg))
  nt = max(1, Int(threads))
  nmax = maximum((sz[d] for d in reg); init = 1)
  R2RPlan{T,N,typeof(lines),length(reg)}(sz, reg, ks, lines, nt, [R2RBuffers{T}(nmax) for _ in 1:nt])
end

Base.size(p::R2RPlan) = p.sz
Base.ndims(::R2RPlan{T,N}) where {T,N} = N
AbstractFFTs.fftdims(p::R2RPlan) = p.region

function Base.show(io::IO, p::R2RPlan{T}) where {T}
  print(io, "FFTL r2r plan for $(join(p.sz, "×")) $T, ", join(("dim $d: $k" for (d, k) in zip(p.region, p.kinds)), ", "))
end

"""
    StridedLine{T}

One line of an array as a vector: `n` values `stride` apart from `ptr` (the array must be kept
alive by the caller). Line kernels read and write through it without copies; every kind reads
its whole input before writing, so input and output may be the same line.
"""
struct StridedLine{T} <: AbstractVector{T}
  ptr::Ptr{T}
  stride::Int
  n::Int
end
Base.size(l::StridedLine) = (l.n,)
Base.@propagate_inbounds Base.getindex(l::StridedLine, i::Int) = unsafe_load(l.ptr, 1 + (i - 1) * l.stride)
Base.@propagate_inbounds Base.setindex!(l::StridedLine{T}, v, i::Int) where {T} = unsafe_store!(l.ptr, T(v), 1 + (i - 1) * l.stride)
Base.IndexStyle(::Type{<:StridedLine}) = IndexLinear()

# one dimension of an Array, each line read and written in place through a StridedLine
function r2r_dim!(y::Array{T}, x::Array{T}, d::Int, K::R2RLine{T}, p::R2RPlan{T}) where {T}
  S, outer = dim_extent(p.sz, d)
  n = p.sz[d]
  GC.@preserve x y begin
    px = pointer(x); py = pointer(y)
    each_line(S * outer, n, p.nthreads) do c, l
      local o, i
      o, i = divrem(l, S)
      local off = (i + S * n * o) * sizeof(T)
      r2r_line!(StridedLine{T}(py + off, S, n), StridedLine{T}(px + off, S, n), K, p.bufs[c])
    end
  end
  y
end

function apply_r2r!(y::Array{T}, x::Array{T}, p::R2RPlan{T}, ls::Tuple, ds::Tuple) where {T}
  isempty(ls) && return y
  r2r_dim!(y, x, first(ds), first(ls), p)
  apply_r2r!(y, y, p, Base.tail(ls), Base.tail(ds))
end

function LinearAlgebra.mul!(y::Array{T,N}, p::R2RPlan{T,N}, x::Array{T,N}) where {T,N}
  size(x) == p.sz && size(y) == p.sz || throw(DimensionMismatch("r2r plan for size $(p.sz)"))
  isempty(p.region) && return copyto!(y, x)
  apply_r2r!(y, x, p, p.lines, p.region)
end

Base.:*(p::R2RPlan{T,N}, x::AbstractArray{<:Real,N}) where {T,N} =
  mul!(Array{T,N}(undef, p.sz), p, convert(Array{T,N}, x))

# inverse kinds and the scaling of the unnormalised pair
inverse_kind(k::R2RKind) = k == REDFT10 ? REDFT01 : k == REDFT01 ? REDFT10 :
                           k == RODFT10 ? RODFT01 : k == RODFT01 ? RODFT10 : k
r2r_scale(k::R2RKind, n) = k == REDFT00 ? 2(n - 1) : k == RODFT00 ? 2(n + 1) : k == DHT ? n : 2n

function AbstractFFTs.plan_inv(p::R2RPlan{T}) where {T}
  s = prod(r2r_scale(k, p.sz[d]) for (k, d) in zip(p.kinds, p.region); init = 1)
  AbstractFFTs.ScaledPlan(R2RPlan{T}(p.sz, map(inverse_kind, p.kinds), p.region; threads = p.nthreads), inv(T(s)))
end

"""
    plan_r2r(x::AbstractArray{<:Real}, kinds, region = 1:ndims(x); threads)
    r2r(x, kinds, region = 1:ndims(x))

Unnormalised real-to-real transforms; `kinds` is one `R2RKind` (REDFT00 … RODFT11, DHT) for all
dimensions of `region` or one per dimension.
"""
plan_r2r(x::AbstractArray, kinds, region = 1:ndims(x); kw...) = R2RPlan{realtype(x)}(size(x), kinds, region; kw...)
r2r(x::AbstractArray, kinds, region = 1:ndims(x)) = plan_r2r(x, kinds, region) * x

"""
    dct(x, region = 1:ndims(x)), idct(x, region = 1:ndims(x))

Orthonormal DCT-II and its inverse (DCT-III) along `region`.
"""
function dct(x::AbstractArray, region = 1:ndims(x))
  y = r2r(x, REDFT10, region)
  for d in (region isa Integer ? (region,) : region)
    n = size(y, d)
    y .*= inv(sqrt(eltype(y)(2n)))
    selectdim(y, d, 1) ./= sqrt(eltype(y)(2))
  end
  y
end

function idct(x::AbstractArray, region = 1:ndims(x))
  y = convert(Array{realtype(x)}, copy(x))
  for d in (region isa Integer ? (region,) : region)
    n = size(y, d)
    selectdim(y, d, 1) .*= sqrt(eltype(y)(2))
    y .*= inv(sqrt(eltype(y)(2n)))
  end
  r2r(y, REDFT01, region)
end

"""
    R2RPassPlan{T}

Real-to-real transform of one kind along dimension `d` of all lines of arrays with a given size
and strides (any strides), like `PassPlan` for complex data.
"""
struct R2RPassPlan{T<:FFTLFloat,N,K}
  sz::NTuple{N,Int}
  st::NTuple{N,Int}
  d::Int
  line::K
  offs::Vector{Int}
  nthreads::Int
  bufs::Vector{R2RBuffers{T}}
end

"""
    plan_r2r_pass(v::StridedArray{<:Real}, d, kind; threads = 1)

Plan for the real-to-real transform `kind` (REDFT00 … RODFT11, DHT; unnormalised) along
dimension `d` of all lines of arrays with the size and strides of `v`. Apply in place with
`apply!(p, v)` or into an array of the same layout with `mul!(w, p, v)`.
"""
function plan_r2r_pass(v::StridedArray{T,N}, d::Integer, kind::R2RKind; threads::Integer = 1) where {T<:FFTLFloat,N}
  1 <= d <= N || throw(ArgumentError("dimension $d is not within 1:$N"))
  sz = size(v); st = strides(v)
  outer = Tuple(i for i in 1:N if i != d)
  offs = vec([sum(((I[j] - 1) * st[outer[j]] for j in eachindex(outer)); init = 0)
              for I in CartesianIndices(ntuple(j -> sz[outer[j]], length(outer)))])
  line = R2RLine{T}(kind, sz[d])
  nt = max(1, Int(threads))
  R2RPassPlan{T,N,typeof(line)}(sz, st, Int(d), line, offs, nt, [R2RBuffers{T}(sz[d]) for _ in 1:nt])
end

function Base.show(io::IO, p::R2RPassPlan{T}) where {T}
  print(io, "FFTL r2r pass plan, $(p.line.kind) along dim $(p.d) of $(join(p.sz, "×")) $T, strides $(p.st)")
end

function LinearAlgebra.mul!(w::StridedArray{T,N}, p::R2RPassPlan{T,N}, v::StridedArray{T,N}) where {T,N}
  (size(v) == p.sz && strides(v) == p.st && size(w) == p.sz && strides(w) == p.st) ||
    throw(DimensionMismatch("r2r pass plan for size $(p.sz) and strides $(p.st)"))
  n = p.sz[p.d]; sd = p.st[p.d]
  GC.@preserve v w p begin
    pv = pointer(v); pw = pointer(w)
    each_line(length(p.offs), n, p.nthreads) do c, l
      local off = p.offs[l+1] * sizeof(T)
      r2r_line!(StridedLine{T}(pw + off, sd, n), StridedLine{T}(pv + off, sd, n), p.line, p.bufs[c])
    end
  end
  w
end

apply!(p::R2RPassPlan, v) = mul!(v, p, v)
