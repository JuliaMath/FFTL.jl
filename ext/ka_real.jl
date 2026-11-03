# Real transforms on KernelAbstractions backends (part of FFTLKernelAbstractionsExt): r2c, c2r
# and the real-to-real kinds, built on the complex KAPlan plus small elementwise kernels.
#
# Lines along dimension d of an array of size sz: element j of line (i, o) at i + S(j + n o),
# S = prod(sz[1:d-1]), o over the dimensions after d.
# - r2c, n even (M = n/2): z_j = x_(2j) + i x_(2j+1), Z = DFT_M(z) along d, then
#   X_k = (Z_k + conj Z_(M-k))/2 + w^k (Z_k - conj Z_(M-k))/(2i), w = exp(-2πi/n), k = 0…M,
#   one thread per pair k, M - k.
#   n odd: the complex DFT of the line, first n÷2+1 values.
# - c2r, n even: Z_k = (X_k + conj X_(M-k)) + i w^k (X_k - conj X_(M-k)), w = exp(+2πi/n)
#   (real parts of X_0, X_M only), z = backward DFT_M(Z), x_(2j) = Re z_j, x_(2j+1) = Im z_j.
#   n odd: the Hermitian full line, backward DFT, real part.
# - r2r: a complex DFT (sign -1) of length L = 2n (2(n-1) for DCT-I, 2(n+1) for DST-I, n for DHT
#   and the halfcomplex pair) of a pre-twiddled, zero-padded line and a post-twiddle, as in
#   src/generic.jl (formulas there). Up to 2–4× the arithmetic of a dedicated kernel, any kind
#   along any dimension. DCT-II/III and DST-II/III of even length instead use one complex DFT of
#   half the length (ka_dct2_pre! and the kernels after it); the steps before and after it run
#   inside the DFT's first and last pass where the pass layout allows, and so do those of
#   DCT-IV/DST-IV of even length (complex DFT of n/2 between twiddles, see src/r2r.jl) and of
#   DCT-I (r2c of the even extension of length 2(n-1), a complex DFT of n - 1).
# The other dimensions of a real region are complex passes on the half spectrum.
# Along dimension 1 (S = 1, n even, n/2 without Bluestein) the line plan does it all: the real
# array is z itself (reinterpreted as complex), the r2c post-processing runs in its last pass,
# the c2r pre-processing in its first pass (pass_kernel's rhook). Elsewhere the elementwise
# kernels below run before and after the line plan.

@inline lineidx(t, S, n) = (i = t % S; r = t ÷ S; (i, r % n, r ÷ n))   # 0-based (i, j, o), Int32

@kernel function ka_pack!(z, @Const(x), S, M)
  t = (@index(Global, Linear) - 1) % Int32
  i, j, o = lineidx(t, S, M)
  b = i + S * (2j + 2M * o)
  @inbounds z[t+1] = complex(x[b+1], x[b+S+1])
end

# one thread per pair k, M - k (k = 0…M/2); u[k] = -i/2 · exp(-2πik/n): X_k = s/2 + u_k d,
# X_(M-k) = conj(s/2 - u_k d) with s = Z_k + conj Z_(M-k), d = Z_k - conj Z_(M-k)
@kernel function ka_r2c_post!(y, @Const(Z), @Const(u), S, M)
  t = (@index(Global, Linear) - 1) % Int32
  i, k, o = lineidx(t, S, M ÷ Int32(2) + Int32(1))
  zb = i + S * M * o; yb = i + S * (M + Int32(1)) * o
  T = real(eltype(y))
  @inbounds if k == 0
    z0 = Z[zb+1]
    y[yb+1] = complex(real(z0) + imag(z0), zero(T))
    y[yb+S*M+1] = complex(real(z0) - imag(z0), zero(T))
  else
    zk = Z[zb+S*k+1]; zm = conj(Z[zb+S*(M-k)+1])
    h = (zk + zm) * T(0.5); d = u[k+1] * (zk - zm)
    y[yb+S*k+1] = h + d
    y[yb+S*(M-k)+1] = conj(h - d)
  end
end

@kernel function ka_tocomplex!(z, @Const(x))
  t = @index(Global, Linear)
  @inbounds z[t] = complex(x[t])
end

@kernel function ka_half!(y, @Const(Z), S, n)
  t = (@index(Global, Linear) - 1) % Int32
  i, k, o = lineidx(t, S, n ÷ 2 + 1)
  @inbounds y[t+1] = Z[i+S*(k+n*o)+1]
end

# one thread per pair k, M - k; u[k] = i · exp(2πik/n): Z_k = s + u_k d, Z_(M-k) = conj(s - u_k d)
# with s = X_k + conj X_(M-k), d = X_k - conj X_(M-k); k = 0 from the real parts of X_0, X_M
@kernel function ka_c2r_pre!(Z, @Const(X), @Const(u), S, M)
  t = (@index(Global, Linear) - 1) % Int32
  i, k, o = lineidx(t, S, M ÷ Int32(2) + Int32(1))
  xb = i + S * (M + Int32(1)) * o; zb = i + S * M * o
  @inbounds if k == 0
    x0 = real(X[xb+1]); xM = real(X[xb+S*M+1])
    Z[zb+1] = complex(x0 + xM, x0 - xM)
  else
    xk = X[xb+S*k+1]; xm = conj(X[xb+S*(M-k)+1])
    sm = xk + xm; d = u[k+1] * (xk - xm)
    Z[zb+S*k+1] = sm + d
    Z[zb+S*(M-k)+1] = conj(sm - d)
  end
end

@kernel function ka_unpack!(y, @Const(z), S, M)
  t = (@index(Global, Linear) - 1) % Int32
  i, j, o = lineidx(t, S, M)
  b = i + S * (2j + 2M * o)
  @inbounds begin
    v = z[t+1]
    y[b+1] = real(v); y[b+S+1] = imag(v)
  end
end

@kernel function ka_hermitian!(F, @Const(X), S, n)
  t = (@index(Global, Linear) - 1) % Int32
  i, k, o = lineidx(t, S, n)
  h = n ÷ 2 + 1
  base = i + S * h * o
  @inbounds F[t+1] = k == 0 ? complex(real(X[base+1])) : k < h ? X[base+S*k+1] : conj(X[base+S*(n-k)+1])
end

@kernel function ka_real!(y, @Const(z))
  t = @index(Global, Linear)
  @inbounds y[t] = real(z[t])
end

# r2r kinds as integers (FFTL.R2RKind values): REDFT00 0, REDFT10 1, REDFT01 2, REDFT11 3,
# RODFT00 4, RODFT10 5, RODFT01 6, RODFT11 7, DHT 8, R2HC 9, HC2R 10
@kernel function ka_r2r_pre!(z, @Const(x), @Const(pre), S, n, L, kind)
  t = (@index(Global, Linear) - 1) % Int32
  i, m, o = lineidx(t, S, L)
  xb = i + S * n * o
  T = real(eltype(z))
  v = zero(eltype(z))
  @inbounds if kind == 0
    m < n && (v = complex((m == 0 || m == n - 1 ? T(1) : T(2)) * x[xb+S*m+1]))
  elseif kind == 2
    m < n && (v = (m == 0 ? T(1) : T(2)) * x[xb+S*m+1] * pre[m+1])
  elseif kind == 6
    1 <= m <= n && (v = (m == n ? T(1) : T(2)) * x[xb+S*(m-1)+1] * pre[m+1])
  elseif kind == 4
    1 <= m <= n && (v = complex(x[xb+S*(m-1)+1]))
  elseif kind == 10
    # conj of the halfcomplex spectrum: F_m = x_m + i x_(n-m) (0 < 2m < n), F_(n-m) = conj F_m
    v = m == 0 || 2m == n ? complex(x[xb+S*m+1]) :
        2m < n ? complex(x[xb+S*m+1], -x[xb+S*(n-m)+1]) : complex(x[xb+S*(n-m)+1], x[xb+S*m+1])
  elseif kind == 8 || kind == 9
    v = complex(x[xb+S*m+1])
  else  # 1, 3, 5, 7
    m < n && (v = x[xb+S*m+1] * pre[m+1])
  end
  @inbounds z[t+1] = v
end

@kernel function ka_r2r_post!(y, @Const(Z), @Const(post), S, n, L, kind)
  t = (@index(Global, Linear) - 1) % Int32
  i, k, o = lineidx(t, S, n)
  zb = i + S * L * o
  T = real(eltype(Z))
  @inbounds y[t+1] = if kind == 0 || kind == 2 || kind == 10
    real(Z[zb+S*k+1])
  elseif kind == 1 || kind == 3
    real(post[k+1] * Z[zb+S*k+1])
  elseif kind == 4
    -2imag(Z[zb+S*(k+1)+1])
  elseif kind == 5
    -imag(post[k+1] * Z[zb+S*(k+1)+1])
  elseif kind == 6
    -imag(Z[zb+S*k+1])
  elseif kind == 7
    -imag(post[k+1] * Z[zb+S*k+1])
  elseif kind == 8
    real(Z[zb+S*k+1]) - imag(Z[zb+S*k+1])
  else  # 9: R2HC
    2k <= n ? real(Z[zb+S*k+1]) : imag(Z[zb+S*(n-k)+1])
  end
end

# DCT-II / DST-II (dst = 1) of even length n = 2M along a dimension by one complex DFT of length
# M: z_m = r_2m + i r_2m+1 of the permuted line r_j = x_2j (j < M), r_j = ±x_2(n-1-j)+1 (j >= M,
# minus for DST-II, the DCT-II of ((-1)^j x_j)); then with C = DFT_M(z)
#   R_k = (C_k + conj C_(M-k))/2 - i/2 W_n^k (C_k - conj C_(M-k)),  u_k = exp(-iπk/(2n)) R_k,
#   y_k = 2 Re u_k, y_(n-k) = -2 Im u_k (k = 0…M); DST-II writes them to n-1-k and k-1.
# DCT-III / DST-III reverse the steps: R_k = conj(t_k) (y_k - i y_(n-k))/2 (y_n = 0),
#   C_k = 4 [(R_k + conj R_(M-k))/2 + i conj(W_n^k) (R_k - conj R_(M-k))/2], z = backward DFT_M(C),
#   x_2j = r_j, x_2j+1 = r_(n-1-j); DST-III reads the input reversed and signs the output (-1)^j.
# w[k] = W_n^k = exp(-2πik/n), t[k] = exp(-iπk/(2n)), k = 0…M.
@kernel function ka_dct2_pre!(z, @Const(x), S, n, M, dst)
  t = (@index(Global, Linear) - 1) % Int32
  i, m, o = lineidx(t, S, M)
  xb = i + S * n * o
  ja = Int32(2) * m; jb = ja + Int32(1)
  ra = ja < M ? Int32(2) * ja : Int32(2) * (n - Int32(1) - ja) + Int32(1)
  rb = jb < M ? Int32(2) * jb : Int32(2) * (n - Int32(1) - jb) + Int32(1)
  @inbounds va = x[xb+S*ra+1]; @inbounds vb = x[xb+S*rb+1]
  if dst != 0
    ja >= M && (va = -va)
    jb >= M && (vb = -vb)
  end
  @inbounds z[t+1] = complex(va, vb)
end

@kernel function ka_dct2_post!(y, @Const(Z), @Const(w), @Const(tw), S, n, M, dst)
  t = (@index(Global, Linear) - 1) % Int32
  i, k, o = lineidx(t, S, M + Int32(1))
  zb = i + S * M * o; yb = i + S * n * o
  T = real(eltype(y))
  k1 = k == M ? zero(Int32) : k
  k2 = k == zero(Int32) ? zero(Int32) : M - k
  @inbounds c1 = Z[zb+S*k1+1]
  @inbounds c2 = conj(Z[zb+S*k2+1])
  @inbounds u = tw[k+1] * ((c1 + c2) * T(0.5) + complex(zero(T), T(-0.5)) * w[k+1] * (c1 - c2))
  ka = dst != 0 ? n - Int32(1) - k : k
  @inbounds y[yb+S*ka+1] = 2real(u)
  if zero(Int32) < k < M
    kb = dst != 0 ? k - Int32(1) : n - k
    @inbounds y[yb+S*kb+1] = -2imag(u)
  end
end

@kernel function ka_dct3_pre!(Z, @Const(y), @Const(w), @Const(tw), S, n, M, dst)
  t = (@index(Global, Linear) - 1) % Int32
  i, k, o = lineidx(t, S, M)
  yb = i + S * n * o
  T = real(eltype(Z))
  # input element j (reversed for DST-III); y_n = 0
  @inline yv(j) = j >= n ? zero(T) : (@inbounds y[yb+S*(dst != 0 ? n - Int32(1) - j : j)+1])
  ka = k; kb = M - k
  Ra = conj(tw[ka+1]) * complex(yv(ka), -yv(n - ka)) * T(0.5)
  Rb = conj(tw[kb+1]) * complex(yv(kb), -yv(n - kb)) * T(0.5)
  Rb = conj(Rb)
  @inbounds Z[i+S*(k+M*o)+1] = T(2) * ((Ra + Rb) + complex(zero(T), one(T)) * conj(w[k+1]) * (Ra - Rb))
end

@kernel function ka_dct3_post!(x, @Const(z), S, n, M, dst)
  t = (@index(Global, Linear) - 1) % Int32
  i, j, o = lineidx(t, S, n)
  q = iseven(j) ? j ÷ Int32(2) : n - Int32(1) - (j - Int32(1)) ÷ Int32(2)
  @inbounds v = z[i+S*(q÷Int32(2)+M*o)+1]
  r = iseven(q) ? real(v) : imag(v)
  @inbounds x[t+1] = dst != 0 && isodd(j) ? -r : r
end

# DCT-I of lines of length N + 1 as the r2c of their even extension xe (length 2N, xe_i = x_i for
# i ≤ N, x_(2N-i) above) by a complex DFT of length N, for any N (Bluestein included):
# z_K = xe_2K + i xe_2K+1, Z = DFT_N(z), y_K = Re[(Z_K + conj Z_(N-K))/2 + u_K (Z_K - conj Z_(N-K))]
# with u_K = -i/2 exp(-iπK/N), K = 0…N (Z_N = Z_0).
@kernel function ka_dct1_pre!(z, @Const(x), S, N)
  t = (@index(Global, Linear) - 1) % Int32
  i, K, o = lineidx(t, S, N)
  xb = i + S * (N + Int32(1)) * o
  a = Int32(2) * K; b = a + Int32(1)
  a = a <= N ? a : Int32(2) * N - a; b = b <= N ? b : Int32(2) * N - b
  @inbounds z[t+1] = complex(x[xb+S*a+1], x[xb+S*b+1])
end

@kernel function ka_dct1_post!(y, @Const(Z), @Const(u), S, N)
  t = (@index(Global, Linear) - 1) % Int32
  i, K, o = lineidx(t, S, N + Int32(1))
  zb = i + S * N * o
  T = real(eltype(y))
  k1 = K == N ? zero(Int32) : K
  k2 = K == zero(Int32) ? zero(Int32) : N - K
  @inbounds c1 = Z[zb+S*k1+1]
  @inbounds c2 = conj(Z[zb+S*k2+1])
  @inbounds y[t+1] = real((c1 + c2) * T(0.5) + u[K+1] * (c1 - c2))
end

todevice(backend, v::Vector) = (d = KA.allocate(backend, eltype(v), length(v)); copyto!(d, v); d)
devzeros(backend, T, dims) = KA.zeros(backend, T, dims...)
setdim(sz, d, v) = ntuple(i -> i == d ? v : sz[i], length(sz))
# Int arguments as Int32: 64-bit division and remainder per element are slow on GPUs (sizes stay
# below 2^31 elements there)
i32(a) = a isa Int ? Int32(a) : a
run!(k, backend, n, args...) = k(backend)(map(i32, args)...; ndrange = n)

"""
    KARealPlan

r2c (`σ = -1`) or unnormalised c2r (`σ = +1`) plan on a KernelAbstractions backend; `sz` is the
size of the real array, the complex array has n ÷ 2 + 1 values along `region[1]`.
"""
mutable struct KARealPlan{T,N,B} <: AbstractFFTs.Plan{T}
  sz::NTuple{N,Int}
  region::Vector{Int}
  σ::Int
  line::KAPlan                       # along region[1]: length n/2 (even n) or n (odd n)
  rest::Union{Nothing,KAPlan}        # the other dimensions, on the half spectrum
  w::Any                             # u_k = -i/2 exp(-2πik/n) (r2c) or i exp(2πik/n) (c2r), k = 0…n/2
  bufs::Vector{Any}
  backend::B
  pinv::AbstractFFTs.Plan
  KARealPlan{T,N,B}(sz, region, σ, line, rest, w, bufs, backend) where {T,N,B} =
    new{T,N,B}(sz, region, σ, line, rest, w, bufs, backend)
end

function KARealPlan(backend, ::Type{T}, sz::NTuple{N,Int}, σ::Int; region = 1:N) where {T,N}
  reg = FFTL.normregion(region, N)
  isempty(reg) && throw(ArgumentError("a real transform needs at least one dimension"))
  d1 = reg[1]; n = sz[d1]; h = n ÷ 2 + 1
  hsz = setdim(sz, d1, h)
  even = iseven(n) && n > 2
  lsz = even ? setdim(sz, d1, n ÷ 2) : sz
  w = todevice(backend, [Complex{T}((σ < 0 ? complex(0, -0.5) : complex(0, 1)) * cispi(2σ * k / n)) for k in 0:h-1])
  fuse = even && d1 == 1 && FFTL.factorize(n ÷ 2)[2] == 1
  line = fuse ? KAPlan(backend, T, lsz, σ; region = d1:d1, rhook = σ, rtab = w) : KAPlan(backend, T, lsz, σ; region = d1:d1)
  rest = length(reg) > 1 ? KAPlan(backend, T, hsz, σ; region = reg[2:end]) : nothing
  bufs = Any[devzeros(backend, Complex{T}, lsz), devzeros(backend, Complex{T}, lsz), devzeros(backend, Complex{T}, hsz)]
  KARealPlan{T,N,typeof(backend)}(sz, reg, σ, line, rest, w, bufs, backend)
end

Base.size(p::KARealPlan) = p.σ < 0 ? p.sz : setdim(p.sz, p.region[1], p.sz[p.region[1]] ÷ 2 + 1)
AbstractFFTs.fftdims(p::KARealPlan) = Tuple(p.region)
AbstractFFTs.output_size(p::KARealPlan) = p.σ < 0 ? setdim(p.sz, p.region[1], p.sz[p.region[1]] ÷ 2 + 1) : p.sz
Base.show(io::IO, p::KARealPlan{T}) where {T} =
  print(io, "FFTL KernelAbstractions ", p.σ < 0 ? "r2c" : "c2r", " plan: ", T, " ", join(p.sz, "×"), ", dims ", Tuple(p.region))

function LinearAlgebra.mul!(y::AbstractArray, p::KARealPlan{T}, x::AbstractArray) where {T}
  size(x) == size(p) && size(y) == AbstractFFTs.output_size(p) || throw(DimensionMismatch("plan for $(size(p)) → $(AbstractFFTs.output_size(p))"))
  be = p.backend; d1 = p.region[1]; n = p.sz[d1]; h = n ÷ 2 + 1
  S = prod(p.sz[1:d1-1]; init = 1); outer = prod(p.sz[d1+1:end]; init = 1)
  even = iseven(n) && n > 2; M = n ÷ 2
  z, Z, hb = p.bufs
  # z as the real array itself (line plans with rhook)
  asz(a) = eltype(a) === T && S == 1 ? reinterpret(Complex{T}, a) : nothing
  if p.σ < 0
    out = p.rest === nothing ? y : hb
    if p.line.rhook != 0
      xz = asz(x)
      xz === nothing && (run!(ka_pack!, be, S * M * outer, z, x, S, M); xz = z)
      launch!(out, xz, p.line)
    elseif even
      run!(ka_pack!, be, S * M * outer, z, x, S, M)
      launch!(Z, z, p.line)
      run!(ka_r2c_post!, be, S * (M ÷ 2 + 1) * outer, out, Z, p.w, S, M)
    else
      run!(ka_tocomplex!, be, length(x), z, x)
      launch!(Z, z, p.line)
      run!(ka_half!, be, S * h * outer, out, Z, S, n)
    end
    p.rest === nothing || launch!(y, hb, p.rest)
  else
    src = x
    if p.rest !== nothing
      launch!(hb, x, p.rest); src = hb
    end
    if p.line.rhook != 0
      yz = asz(y)
      launch!(yz === nothing ? z : yz, src, p.line)
      yz === nothing && run!(ka_unpack!, be, S * M * outer, y, z, S, M)
    elseif even
      run!(ka_c2r_pre!, be, S * (M ÷ 2 + 1) * outer, Z, src, p.w, S, M)
      launch!(z, Z, p.line)
      run!(ka_unpack!, be, S * M * outer, y, z, S, M)
    else
      run!(ka_hermitian!, be, S * n * outer, Z, src, S, n)
      launch!(z, Z, p.line)
      run!(ka_real!, be, length(y), y, z)
    end
  end
  y
end

Base.:*(p::KARealPlan{T}, x::AbstractArray) where {T} =
  mul!(similar(x, p.σ < 0 ? Complex{T} : T, AbstractFFTs.output_size(p)), p, x)

AbstractFFTs.plan_inv(p::KARealPlan{T}) where {T} =
  AbstractFFTs.ScaledPlan(KARealPlan(p.backend, T, p.sz, -p.σ; region = p.region),
                          AbstractFFTs.normalization(T, p.sz, p.region))

"""
    KAR2RPlan

Real-to-real plan on a KernelAbstractions backend, one `FFTL.R2RKind` per dimension of `region`.
"""
mutable struct KAR2RPlan{T,N,B} <: AbstractFFTs.Plan{T}
  sz::NTuple{N,Int}
  region::Vector{Int}
  kinds::Vector{FFTL.R2RKind}
  dims::Vector{Any}                  # per dimension: (kind, L, KAPlan, pre, post, z, Z), or
                                     # (:half, kind, KAPlan, w, t, z, Z) for DCT/DST-II/III of even length,
                                     # (:fused, kind, KAPlan) for DCT-I and DCT/DST-II, -III, -IV in the DFT's passes
  backend::B
  pinv::AbstractFFTs.Plan
  KAR2RPlan{T,N,B}(sz, region, kinds, dims, backend) where {T,N,B} = new{T,N,B}(sz, region, kinds, dims, backend)
end

function r2r_twiddles(::Type{T}, kind::FFTL.R2RKind, n::Int) where {T}
  e(a, b, d) = [Complex{T}(cispi(-(a * j + b) / d)) for j in 0:n]     # exp(-iπ(a j + b)/d)
  one_ = ones(Complex{T}, n + 1)
  kind == FFTL.REDFT10 && return one_, 2 .* e(1, 0, 2n)
  kind == FFTL.RODFT10 && return one_, 2 .* e(1, 1, 2n)
  kind in (FFTL.REDFT01, FFTL.RODFT01) && return e(1, 0, 2n), one_
  kind in (FFTL.REDFT11, FFTL.RODFT11) && return e(1, 0, 2n), 2 .* e(2, 1, 4n)
  one_, one_
end

function KAR2RPlan(backend, ::Type{T}, sz::NTuple{N,Int}, kinds, region) where {T,N}
  reg = collect(Int, region isa Integer ? (region,) : region)
  allunique(reg) && all(d -> 1 <= d <= N, reg) || throw(ArgumentError("region $region is not within 1:$N"))
  ks = kinds isa FFTL.R2RKind ? fill(kinds, length(reg)) : collect(FFTL.R2RKind, kinds)
  length(ks) == length(reg) || throw(ArgumentError("one kind per transformed dimension"))
  dims = Any[]
  for (k, d) in zip(ks, reg)
    n = sz[d]
    k == FFTL.REDFT00 && n < 2 && throw(ArgumentError("REDFT00 needs length ≥ 2"))
    if k == FFTL.REDFT00 && n >= 3 && FFTL.factorize(n - 1)[2] == 1
      # DCT-I: r2c of the even extension (length 2(n-1)) by one complex DFT of length n - 1, the
      # extension built at the first pass's loads, real parts at the last pass (rhook 8)
      n1 = n - 1
      u = todevice(backend, [Complex{T}(complex(0, -0.5) * cispi(-j / n1)) for j in 0:n1])
      P = KAPlan(backend, T, setdim(sz, d, n1), -1; region = d:d, rhook = 8, rtab = (u, u))
      if P.rhook != 0
        push!(dims, (:fused, Int(k), P))
        continue
      end
    end
    if k == FFTL.REDFT00 && n >= 3
      # the same with the steps before and after the DFT as kernels of their own: n - 1 not
      # smooth (Bluestein), or the pass layout does not take the hooks
      n1 = n - 1; zsz = setdim(sz, d, n1)
      u = todevice(backend, [Complex{T}(complex(0, -0.5) * cispi(-j / n1)) for j in 0:n1])
      push!(dims, (:dct1, Int(k), KAPlan(backend, T, zsz, -1; region = d:d), u,
                   devzeros(backend, Complex{T}, zsz), devzeros(backend, Complex{T}, zsz)))
      continue
    end
    if k in (FFTL.REDFT11, FFTL.RODFT11) && iseven(n) && n >= 4 && FFTL.factorize(n ÷ 2)[2] == 1
      # DCT-IV/DST-IV of even length: complex backward DFT of n/2 between element-wise twiddles,
      # all in the DFT's passes (pass_kernel's rhook 6/7)
      M = n ÷ 2
      pre = todevice(backend, [Complex{T}(cispi(j / n)) for j in 0:M-1])
      post = todevice(backend, [Complex{T}(cispi((4j + 1) / (4n))) for j in 0:M-1])
      P = KAPlan(backend, T, setdim(sz, d, M), 1; region = d:d, rhook = k == FFTL.REDFT11 ? 6 : 7, rtab = (pre, post))
      if P.rhook != 0
        push!(dims, (:fused, Int(k), P))
        continue
      end
    end
    if k in (FFTL.REDFT10, FFTL.RODFT10, FFTL.REDFT01, FFTL.RODFT01) && iseven(n) && n >= 4
      M = n ÷ 2; hsz = setdim(sz, d, M)
      σ = k in (FFTL.REDFT10, FFTL.RODFT10) ? -1 : 1
      w = todevice(backend, [Complex{T}(cispi(-2j / n)) for j in 0:M])
      tw = todevice(backend, [Complex{T}(cispi(-j / (2n))) for j in 0:M])
      if σ > 0 && FFTL.factorize(M)[2] == 1
        # the whole DCT-III/DST-III in the line plan's passes (pass_kernel's rhook 4/5):
        # C_K = P_K (y_K - i y_(n-K)) + Q_K (y_(M-K) + i y_(M+K)) with a_k = conj(t_k)/2,
        # P_K = a_K (2 + 2i conj w_K), Q_K = conj(a_(M-K)) (2 - 2i conj w_K)
        tk(j) = cispi(-j / (2n)); wk(j) = cispi(-2j / n)
        Pt = todevice(backend, [Complex{T}(conj(tk(j)) / 2 * (2 + 2im * conj(wk(j)))) for j in 0:M-1])
        Qt = todevice(backend, [Complex{T}(tk(M - j) / 2 * (2 - 2im * conj(wk(j)))) for j in 0:M-1])
        P = KAPlan(backend, T, hsz, 1; region = d:d, rhook = k == FFTL.REDFT01 ? 4 : 5, rtab = (Pt, Qt))
        if P.rhook != 0
          push!(dims, (:fused, Int(k), P))
          continue
        end
      end
      if σ < 0 && FFTL.factorize(M)[2] == 1
        # the whole DCT-II/DST-II in the line plan's passes (pass_kernel's rhook 2/3)
        u = todevice(backend, [Complex{T}(complex(0, -0.5) * cispi(-2j / n)) for j in 0:M])
        P = KAPlan(backend, T, hsz, -1; region = d:d, rhook = k == FFTL.REDFT10 ? 2 : 3, rtab = (u, tw))
        if P.rhook != 0
          push!(dims, (:fused, Int(k), P))
          continue
        end
      end
      push!(dims, (:half, Int(k), KAPlan(backend, T, hsz, σ; region = d:d), w, tw,
                   devzeros(backend, Complex{T}, hsz), devzeros(backend, Complex{T}, hsz)))
      continue
    end
    L = k == FFTL.REDFT00 ? 2(n - 1) : k == FFTL.RODFT00 ? 2(n + 1) : k in (FFTL.DHT, FFTL.R2HC, FFTL.HC2R) ? n : 2n
    lsz = setdim(sz, d, L)
    pre, post = r2r_twiddles(T, k, n)
    push!(dims, (Int(k), L, KAPlan(backend, T, lsz, -1; region = d:d), todevice(backend, pre), todevice(backend, post),
                 devzeros(backend, Complex{T}, lsz), devzeros(backend, Complex{T}, lsz)))
  end
  KAR2RPlan{T,N,typeof(backend)}(sz, reg, ks, dims, backend)
end

Base.size(p::KAR2RPlan) = p.sz
AbstractFFTs.fftdims(p::KAR2RPlan) = Tuple(p.region)
Base.show(io::IO, p::KAR2RPlan{T}) where {T} =
  print(io, "FFTL KernelAbstractions r2r plan: ", T, " ", join(p.sz, "×"), ", ", join(("dim $d: $k" for (d, k) in zip(p.region, p.kinds)), ", "))

function LinearAlgebra.mul!(y::AbstractArray{T}, p::KAR2RPlan{T}, x::AbstractArray{T}) where {T}
  size(x) == p.sz && size(y) == p.sz || throw(DimensionMismatch("r2r plan for size $(p.sz)"))
  isempty(p.region) && return copyto!(y, x)
  be = p.backend; src = x
  for (d, dim) in zip(p.region, p.dims)
    n = p.sz[d]; S = prod(p.sz[1:d-1]; init = 1); outer = prod(p.sz[d+1:end]; init = 1)
    if dim[1] === :fused
      launch!(y, src, dim[3])
    elseif dim[1] === :dct1
      _, _, P, u, z, Z = dim
      run!(ka_dct1_pre!, be, S * (n - 1) * outer, z, src, S, n - 1)
      launch!(Z, z, P)
      run!(ka_dct1_post!, be, S * n * outer, y, Z, u, S, n - 1)
    elseif dim[1] === :half
      _, kind, P, w, tw, z, Z = dim
      M = n ÷ 2; dst = Int(kind == Int(FFTL.RODFT10) || kind == Int(FFTL.RODFT01))
      if kind == Int(FFTL.REDFT10) || kind == Int(FFTL.RODFT10)
        run!(ka_dct2_pre!, be, S * M * outer, z, src, S, n, M, dst)
        launch!(Z, z, P)
        run!(ka_dct2_post!, be, S * (M + 1) * outer, y, Z, w, tw, S, n, M, dst)
      else
        run!(ka_dct3_pre!, be, S * M * outer, Z, src, w, tw, S, n, M, dst)
        launch!(z, Z, P)
        run!(ka_dct3_post!, be, S * n * outer, y, z, S, n, M, dst)
      end
    else
      kind, L, P, pre, post, z, Z = dim
      run!(ka_r2r_pre!, be, S * L * outer, z, src, pre, S, n, L, kind)
      launch!(Z, z, P)
      run!(ka_r2r_post!, be, S * n * outer, y, Z, post, S, n, L, kind)
    end
    src = y
  end
  y
end

Base.:*(p::KAR2RPlan{T}, x::AbstractArray{<:Real}) where {T} = mul!(similar(x, T, p.sz), p, x)

function AbstractFFTs.plan_inv(p::KAR2RPlan{T}) where {T}
  s = prod(FFTL.r2r_scale(k, p.sz[d]) for (k, d) in zip(p.kinds, p.region); init = 1)
  AbstractFFTs.ScaledPlan(KAR2RPlan(p.backend, T, p.sz, map(FFTL.inverse_kind, p.kinds), p.region), inv(T(s)))
end

FFTL.plan_ka_rfft(backend, ::Type{T}, dims::Tuple; region = 1:length(dims)) where {T<:FFTL.FFTLFloat} =
  KARealPlan(backend, T, Tuple(Int.(dims)), -1; region)
FFTL.plan_ka_brfft(backend, ::Type{T}, dims::Tuple; region = 1:length(dims)) where {T<:FFTL.FFTLFloat} =
  KARealPlan(backend, T, Tuple(Int.(dims)), 1; region)
FFTL.plan_ka_r2r(backend, ::Type{T}, dims::Tuple, kinds; region = 1:length(dims)) where {T<:FFTL.FFTLFloat} =
  KAR2RPlan(backend, T, Tuple(Int.(dims)), kinds, region)
