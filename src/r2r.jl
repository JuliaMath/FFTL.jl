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
# - DCT-I: r2c of the even extension (x_0 … x_(n-1), x_(n-2) … x_1) of length 2(n-1).
# - DST-I: r2c of the odd extension (0, x_0 … x_(n-1), 0, -x_(n-1) … -x_0) of length 2(n+1),
#   Y_k = -Im of entry k+1.
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
    inner = RealLine{T}(2(n - 1), -1)
  elseif base == RODFT00
    inner = RealLine{T}(2(n + 1), -1)
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
  lin::Vector{T}
  lout::Vector{T}
end
R2RBuffers{T}(n) where {T} = R2RBuffers{T}(Vector{T}(undef, 2n + 4), Vector{Complex{T}}(undef, 2n + 4),
  Vector{Complex{T}}(undef, 2n + 4), Vector{Complex{T}}(undef, 2n + 4), Vector{Complex{T}}(undef, 2n + 4),
  Vector{T}(undef, n), Vector{T}(undef, n))

r2c!(c::Vector{Complex{T}}, r::Vector{T}, K::RealLine{T}, B) where {T} =
  GC.@preserve c r B r2c_line!(pointer(c), pointer(r), cptr(B.b1), cptr(B.b2), cptr(B.b3), K)
c2r!(r::Vector{T}, c::Vector{Complex{T}}, K::RealLine{T}, B) where {T} =
  GC.@preserve c r B c2r_line!(pointer(r), pointer(c), cptr(B.b1), cptr(B.b2), cptr(B.b3), K)

# y = transform of x, both contiguous of length n (y may alias x only through the buffers' copies)
function r2r_line!(y::AbstractVector{T}, x::AbstractVector{T}, K::R2RLine{T}, B::R2RBuffers{T}) where {T}
  n = K.n; kind = K.kind
  r = B.r; c = B.c
  if kind == REDFT10 || kind == RODFT10
    sgn = kind == RODFT10
    @inbounds for j in 0:(n-1)÷2
      r[j+1] = sgn && isodd(2j) ? -x[2j+1] : x[2j+1]
    end
    @inbounds for j in 0:n÷2-1
      v = x[2j+2]; r[n-j] = sgn ? -v : v                       # x_(2j+1) has odd index: sign flip
    end
    r2c!(c, r, K.inner, B)
    @inbounds for k in 0:n÷2
      u = K.tw[k+1] * c[k+1]
      kk = sgn ? n - 1 - k : k
      y[kk+1] = 2real(u)
      if 0 < k < n - k
        kk2 = sgn ? k - 1 : n - k
        y[kk2+1] = -2imag(u)
      end
    end
  elseif kind == REDFT01 || kind == RODFT01
    rev = kind == RODFT01
    xin(j) = rev ? x[n-j] : x[j+1]
    @inbounds for k in 0:n÷2
      xk = xin(k); xm = k == 0 ? zero(T) : xin(n - k)
      c[k+1] = K.tw[k+1] * Complex(xk, -xm)
    end
    c2r!(r, c, K.inner, B)
    @inbounds for j in 0:(n-1)÷2
      y[2j+1] = r[j+1]
    end
    @inbounds for j in 0:n÷2-1
      y[2j+2] = r[n-j]
    end
    if rev
      @inbounds for k in 1:2:n-1
        y[k+1] = -y[k+1]
      end
    end
  elseif kind == REDFT11 || kind == RODFT11
    sgn = kind == RODFT11
    xs(j) = sgn && isodd(j) ? -x[j+1] : x[j+1]
    out(k, v) = (y[(sgn ? n - 1 - k : k) + 1] = v)
    b = B.b1
    if iseven(n)
      M = n ÷ 2
      @inbounds for j in 0:M-1
        b[j+1] = Complex(xs(2j), -xs(n - 1 - 2j)) * K.tw[j+1]
      end
      GC.@preserve b B run!(cptr(b), cptr(b), cptr(B.b2), cptr(B.b3), K.inner, 1)
      @inbounds for m in 0:M-1
        w = K.tw2[m+1] * b[m+1]
        out(2m, 2real(w)); out(n - 1 - 2m, 2imag(w))
      end
    else
      @inbounds for j in 0:n-1
        b[j+1] = xs(j) * K.tw[j+1]
      end
      @inbounds for j in n:2n-1
        b[j+1] = zero(Complex{T})
      end
      GC.@preserve b B run!(cptr(b), cptr(b), cptr(B.b2), cptr(B.b3), K.inner, 1)
      @inbounds for k in 0:n-1
        out(k, 2real(K.tw2[k+1] * b[k+1]))
      end
    end
  elseif kind == REDFT00
    @inbounds for j in 0:n-1
      r[j+1] = x[j+1]
    end
    @inbounds for j in 1:n-2
      r[2(n-1)-j+1] = x[j+1]
    end
    r2c!(c, r, K.inner, B)
    @inbounds for k in 0:n-1
      y[k+1] = real(c[k+1])
    end
  elseif kind == RODFT00
    N2 = 2(n + 1)
    r[1] = zero(T); r[n+2] = zero(T)
    @inbounds for j in 0:n-1
      r[j+2] = x[j+1]; r[N2-j] = -x[j+1]
    end
    r2c!(c, r, K.inner, B)
    @inbounds for k in 0:n-1
      y[k+1] = -imag(c[k+2])
    end
  else # DHT
    @inbounds for j in 1:n
      r[j] = x[j]
    end
    r2c!(c, r, K.inner, B)
    @inbounds for k in 0:n÷2
      f = c[k+1]
      y[k+1] = real(f) - imag(f)
      0 < k < n - k && (y[n-k+1] = real(f) + imag(f))
    end
  end
  y
end

"""
    R2RPlan{T,N}

Plan for real-to-real transforms of arrays of size `sz`, one kind per dimension of `region`.
"""
mutable struct R2RPlan{T<:FFTLFloat,N,L<:Tuple} <: AbstractFFTs.Plan{T}
  sz::NTuple{N,Int}
  region::Tuple
  kinds::Tuple
  lines::L
  nthreads::Int
  bufs::Vector{R2RBuffers{T}}
  pinv::AbstractFFTs.Plan
  R2RPlan{T,N,L}(sz, region, kinds, lines, nt, bufs) where {T,N,L} = new{T,N,L}(sz, region, kinds, lines, nt, bufs)
end

function R2RPlan{T}(sz::NTuple{N,Int}, kinds, region; threads::Integer = Threads.nthreads()) where {T<:FFTLFloat,N}
  reg = Tuple(collect(Int, region isa Integer ? (region,) : region))
  allunique(reg) && all(d -> 1 <= d <= N, reg) || throw(ArgumentError("region $region is not within 1:$N"))
  ks = kinds isa R2RKind ? ntuple(_ -> kinds, length(reg)) : Tuple(kinds)
  length(ks) == length(reg) || throw(ArgumentError("one kind per transformed dimension"))
  lines = Tuple(R2RLine{T}(k, sz[d]) for (k, d) in zip(ks, reg))
  nt = max(1, Int(threads))
  nmax = maximum((sz[d] for d in reg); init = 1)
  R2RPlan{T,N,typeof(lines)}(sz, reg, ks, lines, nt, [R2RBuffers{T}(nmax) for _ in 1:nt])
end

Base.size(p::R2RPlan) = p.sz
Base.ndims(::R2RPlan{T,N}) where {T,N} = N
AbstractFFTs.fftdims(p::R2RPlan) = p.region

function Base.show(io::IO, p::R2RPlan{T}) where {T}
  print(io, "FFTL r2r plan for $(join(p.sz, "×")) $T, ", join(("dim $d: $k" for (d, k) in zip(p.region, p.kinds)), ", "))
end

# one dimension, lines gathered into the task's buffers (contiguous lines are copied as well)
function r2r_dim!(y::Array{T}, x::Array{T}, d::Int, K::R2RLine{T}, p::R2RPlan{T}) where {T}
  S, outer = dim_extent(p.sz, d)
  n = p.sz[d]
  nlines = S * outer
  each_line(nlines, n, p.nthreads) do c, l
    local B = p.bufs[c]
    local o, i
    o, i = divrem(l, S)
    local base = i + S * n * o
    for j in 0:n-1
      @inbounds B.lin[j+1] = x[base+S*j+1]
    end
    r2r_line!(view(B.lout, 1:n), view(B.lin, 1:n), K, B)
    for j in 0:n-1
      @inbounds y[base+S*j+1] = B.lout[j+1]
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
