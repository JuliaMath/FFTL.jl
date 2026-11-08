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
#   R2HC     the DFT F_k = Σ X_j e^(-2πijk/n) of real data in halfcomplex order:
#            Y_k = Re F_k (0 ≤ k ≤ n/2), Y_(n-k) = Im F_k (0 < k < n/2)
#   HC2R     the inverse: Y_j = Σ_k F_k e^(2πijk/n), F given in halfcomplex order   (unnormalised)
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
# - DCT-I, N = n - 1: F = r2c of the even extension (x_0, …, x_N, x_(N-1), …, x_1) of length 2N,
#   Y_k = Re F_k.
# - DST-I, N = n + 1: F = r2c of the odd extension (0, x_0, …, x_(n-1), 0, -x_(n-1), …, -x_0) of
#   length 2N, Y_k = -Im F_(k+1).
# - DHT: F = r2c(x); Y_k = Re F_k - Im F_k, Y_(n-k) = Re F_k + Im F_k.
# - R2HC: F = r2c(x) stored in halfcomplex order. HC2R: F_k = X_k + i X_(n-k) (imaginary part
#   zero for k = 0 and k = n/2), Y = c2r(F).

"""
    R2RKind

The real-to-real kinds: `REDFT00` … `REDFT11` (DCT-I … DCT-IV), `RODFT00` … `RODFT11` (DST-I … DST-IV),
`DHT` (Hartley), `R2HC` and `HC2R` (halfcomplex). The formulas are in the documentation
(Cosine, sine and Hartley transforms).
"""
@enum R2RKind REDFT00 REDFT10 REDFT01 REDFT11 RODFT00 RODFT10 RODFT01 RODFT11 DHT R2HC HC2R

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

# real-to-real regions keep their order (one kind per dimension), so duplicates are errors
function check_r2r_region(reg, N::Int)
  allunique(reg) || throw(ArgumentError("region contains a dimension twice"))
  all(d -> 1 <= d <= N, reg) || throw(ArgumentError("region is not within the array dimensions"))
  nothing
end

own_buffers(K::R2RLine) = (i = K.inner; i isa RealLine ? i.inner isa Union{Bluestein,Resample,Composite} : i isa Union{Bluestein,Resample,Composite})

# exp(iπ(a j + b)/d) for j = 0…m-1
function cis_table(::Type{T}, m::Int, a::Int, b::Int, d::Int) where {T}
  t = Vector{Complex{T}}(undef, m)
  for j in 0:m-1
    t[j+1] = Complex{T}(cispi((a * j + b) / d))
  end
  t
end

# Inner kernel of the line for run-time plans (C = RChain): one of these types
const RInner{T} = Union{RealLine{T,RKern{T}},RChain{T},RResample{T}}

function R2RLine{T}(kind::R2RKind, n::Int, ::Type{C} = TypedKernels) where {T,C}
  n >= 1 || throw(ArgumentError("length must be positive"))
  kind == REDFT00 && n < 2 && throw(ArgumentError("REDFT00 needs length ≥ 2"))
  base = kind in (RODFT10,) ? REDFT10 : kind == RODFT01 ? REDFT01 : kind == RODFT11 ? REDFT11 : kind
  tw = Complex{T}[]; tw2 = Complex{T}[]
  if base == REDFT10
    inner = RealLine{T}(n, -1, C)
    tw = cis_table(T, n ÷ 2 + 1, -1, 0, 2n)
  elseif base == REDFT01
    inner = RealLine{T}(n, 1, C)
    tw = cis_table(T, n ÷ 2 + 1, 1, 0, 2n)
  elseif base == REDFT11
    if iseven(n)
      inner = ckernel(T, n ÷ 2, 1, C)
      tw = cis_table(T, n ÷ 2, 1, 0, n)
      tw2 = cis_table(T, n ÷ 2, 4, 1, 4n)
    else
      inner = ckernel(T, 2n, -1, C)
      tw = cis_table(T, n, -1, 0, 2n)
      tw2 = cis_table(T, n, -2, -1, 4n)
    end
  elseif base == REDFT00
    inner = RealLine{T}(2(n - 1), -1, C)
  elseif base == RODFT00
    inner = RealLine{T}(2(n + 1), -1, C)
  elseif base == HC2R
    inner = RealLine{T}(n, 1, C)
  else # DHT, R2HC
    inner = RealLine{T}(n, -1, C)
  end
  R2RLine{T,C === RChain ? RInner{T} : typeof(inner)}(kind, n, inner, tw, tw2)
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
  blk::Vector{T}
end
R2RBuffers{T}(n) where {T} = R2RBuffers{T}(Vector{T}(undef, 2n + 4), Vector{Complex{T}}(undef, 2n + 4),
  Vector{Complex{T}}(undef, 2n + 4), Vector{Complex{T}}(undef, 2n + 4), Vector{Complex{T}}(undef, 2n + 4),
  Vector{T}(undef, r2r_block(T) * n))

# Lines per block along a strided dimension: one cache line of neighbouring values per row.
r2r_block(::Type{T}) where {T} = R2R_BLOCK_BYTES ÷ sizeof(T)
const R2R_BLOCK_BYTES = 64

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
  elseif kind == R2HC
    r2hc!(y, x, K, B)
  elseif kind == HC2R
    hc2r!(y, x, K, B)
  else
    dht!(y, x, K, B)
  end
  y
end

# DCT-II; with `dst` the DST-II as DCT-II of ((-1)^j x_j), output reversed
function dct2!(y, x, K::R2RLine{T}, B, dst::Bool) where {T}
  n = K.n; r = B.r; c = B.c
  # from 64 values on (below the extra calls cost more than they save: 16 values 40 → 60 ns)
  if x isa StridedLine && x.stride == 1 && iseven(n) && n >= 64
    GC.@preserve r dct2_gather!(pointer(r), x.ptr, n, dst)
    return dct2_pairs!(y, K, B, dst)
  end
  @inbounds for j in 0:(n-1)÷2
    r[j+1] = x[2j+1]
  end
  @inbounds for j in 0:n÷2-1
    v = x[2j+2]; r[n-j] = dst ? -v : v
  end
  iseven(n) && n >= 64 && return dct2_pairs!(y, K, B, dst)
  r2c!(c, r, K.inner::RealLine, B)
  @inbounds for k in 0:n÷2
    u = K.tw[k+1] * c[k+1]
    y[(dst ? n - 1 - k : k) + 1] = 2real(u)
    if 0 < k < n - k
      y[(dst ? k - 1 : n - k) + 1] = -2imag(u)
    end
  end
  y
end

# DCT-II/DST-II of even n = 2M from the permuted line in B.r: the complex DFT C of length M of
# it (read as M complex values), then per pair k, M-k the half-spectrum values
# R_k = (C_k + conj C_(M-k))/2 + u_k (C_k - conj C_(M-k)) and R_(M-k) = conj(...) (u_k from the
# real line, see real.jl) times the DCT twiddles, four outputs at once (no r2c post-processing
# pass in between).
function dct2_pairs!(y, K::R2RLine{T}, B, dst::Bool) where {T}
  n = K.n; M = n ÷ 2; c = B.c; r = B.r
  L = K.inner::RealLine
  GC.@preserve r c B run!(cptr(c), pointer(r), cptr(B.b1), cptr(B.b2), L.inner, 1)
  w = L.w; tw = K.tw
  idx(j) = (dst ? n - 1 - j : j) + 1
  @inbounds begin
    z0 = c[1]
    y[idx(0)] = 2 * (real(z0) + imag(z0))
    y[idx(M)] = 2 * real(tw[M+1]) * (real(z0) - imag(z0))
    if y isa StridedLine
      GC.@preserve c w tw begin
        if y.stride == 1 && T === Float32   # two complex values per 128-bit register; F64 keeps the one-complex pass
          dct2_pairs_wide!(y.ptr, cptr(c), cptr(w), cptr(tw), n, M, dst)
        else
          dct2_pairs_vec!(y.ptr, y.stride, cptr(c), cptr(w), cptr(tw), n, M, dst)
        end
      end
    else
      for k in 1:(M-1)÷2
        zk = c[k+1]; zm = conj(c[M-k+1])
        h = (zk + zm) / 2; d = w[k+1] * (zk - zm)
        uk = tw[k+1] * (h + d); um = tw[M-k+1] * conj(h - d)
        y[idx(k)] = 2real(uk); y[idx(n - k)] = -2imag(uk)
        y[idx(M - k)] = 2real(um); y[idx(M + k)] = -2imag(um)
      end
    end
    if iseven(M)
      k = M ÷ 2; zk = c[k+1]; zm = conj(zk)
      uk = tw[k+1] * ((zk + zm) / 2 + w[k+1] * (zk - zm))
      y[idx(k)] = 2real(uk); y[idx(n - k)] = -2imag(uk)
    end
  end
  y
end

# r_j = x_2j, r_(n-1-j) = ±x_2j+1 for a contiguous line of even length n: four values per load,
# the even ones stored forward, the odd ones backward
function dct2_gather!(pr::Ptr{T}, px::Ptr{T}, n::Int, dst::Bool) where {T}
  M = n ÷ 2; S = sizeof(T)
  sgn = Vec{2,T}(dst ? -one(T) : one(T))
  j = 0
  @inbounds while j + 1 < M
    a = vload(Vec{4,T}, px + 2j * S)
    vstore(shufflevector(a, Val((0, 2))), pr + j * S)
    vstore(shufflevector(a, Val((3, 1))) * sgn, pr + (n - 2 - j) * S)
    j += 2
  end
  @inbounds if j < M
    unsafe_store!(pr, unsafe_load(px, 2j + 1), j + 1)
    v = unsafe_load(px, 2j + 2); unsafe_store!(pr, dst ? -v : v, n - j)
  end
  nothing
end

# The pairs k = 1…(M-1)÷2 of dct2_pairs! with one complex value per vector; outputs `ys` apart
function dct2_pairs_vec!(py::Ptr{T}, ys::Int, pc::Ptr{T}, pw::Ptr{T}, ptw::Ptr{T}, n::Int, M::Int, dst::Bool) where {T}
  V = single_vec(T); CB = 2 * sizeof(T)
  half = V((T(0.5), T(0.5)))
  o(j) = (dst ? n - 1 - j : j) * ys + 1
  @inbounds for k in 1:(M-1)÷2
    zk = vload(V, pc + k * CB); czm = cconj(vload(V, pc + (M - k) * CB))
    h = (zk + czm) * half; d = ccmul(vload(V, pw + k * CB), zk - czm)
    uk = ccmul(vload(V, ptw + k * CB), h + d)
    um = ccmul(vload(V, ptw + (M - k) * CB), cconj(h - d))
    unsafe_store!(py, 2 * uk[1], o(k)); unsafe_store!(py, -2 * uk[2], o(n - k))
    unsafe_store!(py, 2 * um[1], o(M - k)); unsafe_store!(py, -2 * um[2], o(M + k))
  end
  nothing
end

# The pairs k = 1…(M-1)÷2 of dct2_pairs! with two consecutive k values per vector (two complex
# values per vector, the widest ladder step for Float32): the four output streams (re and imag of
# the values k and M - k) become adjacent pairs in memory, two values per store instead of four
# scalar ones. Bit-identical to dct2_pairs_vec!: the factor 2 of the outputs is folded into the
# twiddle product (halving and doubling are exact), so h = c_k + conj c_(M-k) and
# 2w_k(c_k - conj c_(M-k)) enter the product unscaled.
function dct2_pairs_wide!(py::Ptr{T}, pc::Ptr{T}, pw::Ptr{T}, ptw::Ptr{T}, n::Int, M::Int, dst::Bool) where {T}
  W = Vec{4,T}; V = Vec{2,T}; CB = 2 * sizeof(T); SB = sizeof(T)
  half = V((T(0.5), T(0.5))); mone = V((-one(T), -one(T)))
  rev = Val((2, 3, 0, 1))                # the mirror loads come out in (k+1, k) order
  kmax = (M - 1) ÷ 2
  @inbounds for k in 1:2:(kmax - 1)
    zc = vload(W, pc + k * CB)                       # c_k, c_(k+1)
    zm = cconj(shufflevector(vload(W, pc + (M - k - 1) * CB), rev))   # conj c_(M-k), conj c_(M-k-1)
    h = zc + zm
    d = ccmul(2 * vload(W, pw + k * CB), zc - zm)
    uk = ccmul(vload(W, ptw + k * CB), h + d)                         # lanes (k, k+1)
    um = ccmul(shufflevector(vload(W, ptw + (M - k - 1) * CB), rev), cconj(h - d))  # lanes (k, k+1)
    if dst
      vstore(shufflevector(uk, Val((2, 0))), py + (n - k - 2) * SB)         # re at n-1-k, n-2-k
      vstore(shufflevector(uk, Val((1, 3))) * mone, py + (k - 1) * SB)      # -im at k-1, k
      vstore(shufflevector(um, Val((0, 2))), py + (M + k - 1) * SB)         # re at M+k-1, M+k
      vstore(shufflevector(um, Val((3, 1))) * mone, py + (M - k - 2) * SB)  # -im at M-k-2, M-k-1
    else
      vstore(shufflevector(uk, Val((0, 2))), py + k * SB)                   # re at k, k+1
      vstore(shufflevector(uk, Val((3, 1))) * mone, py + (n - k - 1) * SB)  # -im at n-k-1, n-k
      vstore(shufflevector(um, Val((2, 0))), py + (M - k - 1) * SB)         # re at M-k-1, M-k
      vstore(shufflevector(um, Val((1, 3))) * mone, py + (M + k) * SB)      # -im at M+k, M+k+1
    end
  end
  # the last pair k, M-k on its own when their number is odd (scalar stores, as above)
  if isodd(kmax)
    k = kmax
    zk = vload(V, pc + k * CB); czm = cconj(vload(V, pc + (M - k) * CB))
    h = (zk + czm) * half; d = ccmul(vload(V, pw + k * CB), zk - czm)
    uk = ccmul(vload(V, ptw + k * CB), h + d)
    um = ccmul(vload(V, ptw + (M - k) * CB), cconj(h - d))
    idx(j) = (dst ? n - 1 - j : j) + 1
    @inbounds begin
      unsafe_store!(py, 2 * uk[1], idx(k))
      unsafe_store!(py, -2 * uk[2], idx(n - k))
      unsafe_store!(py, 2 * um[1], idx(M - k))
      unsafe_store!(py, -2 * um[2], idx(M + k))
    end
  end
  nothing
end

# DCT-III; with `dst` the DST-III as (-1)^k DCT-III(reversed x)_k
function dct3!(y, x, K::R2RLine{T}, B, dst::Bool) where {T}
  n = K.n; r = B.r; c = B.c
  @inbounds for k in 0:n÷2
    xk = dst ? x[n-k] : x[k+1]
    xm = k == 0 ? zero(T) : (dst ? x[k] : x[n-k+1])
    c[k+1] = K.tw[k+1] * Complex(xk, -xm)
  end
  c2r!(r, c, K.inner::RealLine, B)
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
    GC.@preserve b B run!(cptr(b), cptr(b), cptr(B.b2), cptr(B.b3), K.inner::Kernel, 1)
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
    GC.@preserve b B run!(cptr(b), cptr(b), cptr(B.b2), cptr(B.b3), K.inner::Kernel, 1)
    @inbounds for k in 0:n-1
      y[(dst ? n - 1 - k : k) + 1] = 2real(K.tw2[k+1] * b[k+1])
    end
  end
  y
end

# DCT-I and DST-I as the r2c of the even / odd extension of length 2N. The r2c of length N
# with the pairs (j, N-j) folded into one sequence would cost half as much, but recovers the odd
# outputs as a running sum over the spectrum, whose rounding errors accumulate with N.
function dct1!(y, x, K::R2RLine{T}, B) where {T}
  N = K.n - 1; r = B.r; c = B.c
  @inbounds begin
    for j in 0:N
      r[j+1] = x[j+1]
    end
    for j in 1:N-1
      r[2N-j+1] = x[j+1]
    end
  end
  r2c!(c, r, K.inner::RealLine, B)
  @inbounds for k in 0:N
    y[k+1] = real(c[k+1])
  end
  y
end

function dst1!(y, x, K::R2RLine{T}, B) where {T}
  n = K.n; N = n + 1; r = B.r; c = B.c
  @inbounds begin
    r[1] = zero(T); r[N+1] = zero(T)
    for j in 1:n
      v = x[j]
      r[j+1] = v
      r[2N-j+1] = -v
    end
  end
  r2c!(c, r, K.inner::RealLine, B)
  @inbounds for k in 1:n
    y[k] = -imag(c[k+1])
  end
  y
end

function r2hc!(y, x, K::R2RLine{T}, B) where {T}
  n = K.n; r = B.r; c = B.c
  @inbounds for j in 1:n
    r[j] = x[j]
  end
  r2c!(c, r, K.inner::RealLine, B)
  @inbounds for k in 0:n÷2
    f = c[k+1]
    y[k+1] = real(f)
    0 < k < n - k && (y[n-k+1] = imag(f))
  end
  y
end

function hc2r!(y, x, K::R2RLine{T}, B) where {T}
  n = K.n; r = B.r; c = B.c
  @inbounds for k in 0:n÷2
    c[k+1] = Complex{T}(x[k+1], 0 < k < n - k ? x[n-k+1] : zero(T))
  end
  c2r!(r, c, K.inner::RealLine, B)
  @inbounds for j in 1:n
    y[j] = r[j]
  end
  y
end

function dht!(y, x, K::R2RLine{T}, B) where {T}
  n = K.n; r = B.r; c = B.c
  @inbounds for j in 1:n
    r[j] = x[j]
  end
  r2c!(c, r, K.inner::RealLine, B)
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
mutable struct R2RPlan{T<:FFTLFloat,N,L<:Tuple,D,H<:Tuple} <: AbstractFFTs.Plan{T}
  sz::NTuple{N,Int}
  region::NTuple{D,Int}
  kinds::NTuple{D,R2RKind}
  lines::L
  halves::H                      # per dimension: HalfDFT or nothing
  nthreads::Int
  bufs::Vector{R2RBuffers{T}}
  pinv::AbstractFFTs.Plan
  R2RPlan{T,N,L,D,H}(sz, region, kinds, lines, halves, nt, bufs) where {T,N,L,D,H} =
    new{T,N,L,D,H}(sz, region, kinds, lines, halves, nt, bufs)
end

"""
    HalfDFT{T}

DCT-II or DST-II of even length n along a strided dimension (S > 1 values between neighbours of
a line) by one complex DFT of length n/2 along the same dimension: rows of the input are packed
into complex rows z_m = r_2m + i r_2m+1 of the permuted sequence r (r_j = x_2j, r_n-1-j = x_2j+1),
the DFT runs with the lane kernels of the complex plans, and the half-spectrum split and the DCT
twiddles combine whole rows. Every step reads and writes rows of S contiguous values. The DHT
takes the same path without the permutation (r = x) and with y_k = Re R_k - Im R_k,
y_(n-k) = Re R_k + Im R_k from the half spectrum R of the real DFT.
"""
struct HalfDFT{T,P}
  S::Int
  outer::Int
  n::Int
  dst::Bool
  dht::Bool                      # DHT: no permutation, y_k = Re R_k - Im R_k, y_(n-k) = Re R_k + Im R_k
  cplan::P
  z::Array{Complex{T},3}
  w::Vector{Complex{T}}          # -i/2 · exp(-2πik/n), k = 0…n/2
  t::Vector{Complex{T}}          # exp(-iπk/(2n)), k = 0…n/2
end

# M1, DCT-II along dim 2 of n×n against the blocked lines: 0.46–0.64 on one thread (n = 32…1024)
# and on 8 threads up to 128²; from 256² on 8 threads 0.97–1.7, as the half-length DFT needs three
# parallel steps where the blocked lines need one.
# A single even line on several threads has no other parallel loop: it takes the half-length DFT too,
# from 2^17 points on (M1, 8 threads: 32768 is 3× slower than one thread, 131072 is 1.4× faster).
use_half_dft(S, n, outer, nt) = iseven(n) && n >= 4 &&
  (S == 1 && outer == 1 ? nt > 1 && n >= 1 << 17 : S >= 4 && (nt == 1 || S * n * outer < 1 << 16))

function half_dft(::Type{T}, sz, d, kind, z::Vector{Complex{T}}, nt) where {T}
  S, outer = dim_extent(sz, d)
  n = sz[d]
  if S == 1 && outer == 1 && nt > 1 && n >= 1 << 17 && !iseven(n) && haskey(ODD_KINDS, kind)
    return half_odd(T, n, ODD_KINDS[kind], nt)
  end
  if (kind == REDFT11 || kind == RODFT11) && S == 1 && outer == 1 && nt > 1 && n >= 1 << 17 && iseven(n)
    return half_iv(T, n, kind == RODFT11, z, nt)
  end
  if (kind == REDFT01 || kind == RODFT01) && S == 1 && outer == 1 && nt > 1 && n >= 1 << 17 && iseven(n)
    return half_iii(T, n, kind == RODFT01, nt)
  end
  if (kind == REDFT00 || kind == RODFT00) && S == 1 && outer == 1 && nt > 1 && n >= 1 << 17
    return half_i(T, n, kind == RODFT00, nt)
  end
  (kind == REDFT10 || kind == RODFT10 || kind == DHT) && use_half_dft(S, n, outer, nt) || return nothing
  M = n ÷ 2
  zz = reshape(z, S, M, outer)
  cplan = FFTLPlan{T}((S, M, outer), (2,), -1; threads = nt)
  w = Complex{T}[Complex{T}(complex(0, -1 // 2) * cispi(-2k / n)) for k in 0:M]
  HalfDFT{T,typeof(cplan)}(S, outer, n, kind == RODFT10, kind == DHT, cplan, zz, w, cis_table(T, M + 1, -1, 0, 2n))
end

# One line kernel per task. A kernel with buffers of its own (Bluestein, composite) cannot run on
# two tasks at once, so each task gets a copy; the other kernels are shared.
task_lines(K::R2RLine{T}, threads::Integer) where {T} =
  own_buffers(K) && threads > 1 ? [K; [R2RLine{T}(K.kind, K.n) for _ in 2:threads]] : [K]

function R2RPlan{T}(sz::NTuple{N,Int}, kinds, region; threads::Integer = Threads.nthreads()) where {T<:FFTLFloat,N}
  reg = Tuple(collect(Int, region isa Integer ? (region,) : region))
  check_r2r_region(reg, N)
  ks = kinds isa R2RKind ? ntuple(_ -> kinds, length(reg)) : Tuple(kinds)
  length(ks) == length(reg) || throw(ArgumentError("one kind per transformed dimension"))
  lines = Tuple(task_lines(R2RLine{T}(k, sz[d]), threads) for (k, d) in zip(ks, reg))
  nt = max(1, Int(threads))
  nmax = maximum((sz[d] for d in reg); init = 1)
  z = Vector{Complex{T}}(undef, prod(sz) ÷ 2)
  halves = Tuple(half_dft(T, sz, d, k, z, nt) for (k, d) in zip(ks, reg))
  all(isnothing, halves) && (z = Complex{T}[])
  R2RPlan{T,N,typeof(lines),length(reg),typeof(halves)}(sz, reg, ks, lines, halves, nt, [R2RBuffers{T}(nmax) for _ in 1:nt])
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
r2r_dim!(y::Array{T}, x::Array{T}, d::Int, K::R2RLine{T}, p::R2RPlan{T}) where {T} =
  (GC.@preserve x y r2r_dim!(pointer(y), pointer(x), d, (K,), p); y)
r2r_dim!(y::Array{T}, x::Array{T}, d::Int, Ks::Vector{<:R2RLine{T}}, p::R2RPlan{T}) where {T} =
  (GC.@preserve x y r2r_dim!(pointer(y), pointer(x), d, Ks, p); y)

# p: a plan with sz, nthreads and bufs (R2RPlan, RR2RPlan)
# a tuple, not a vector: no allocation per call
r2r_dim!(py::Ptr{T}, px::Ptr{T}, d::Int, K::R2RLine{T}, p) where {T} = r2r_dim!(py, px, d, (K,), p)

# Ks: one kernel per task (see task_lines), or one shared kernel.
function r2r_dim!(py::Ptr{T}, px::Ptr{T}, d::Int, Ks, p) where {T}
  S, outer = dim_extent(p.sz, d)
  n = p.sz[d]
  S > 1 && return r2r_dim_blocked!(py, px, S, outer, n, Ks, p)
  each_line(S * outer, n, line_tasks_of(Ks, p.nthreads)) do c, l
    local o, i
    o, i = divrem(l, S)
    local off = (i + S * n * o) * sizeof(T)
    r2r_line!(StridedLine{T}(py + off, S, n), StridedLine{T}(px + off, S, n), Ks[task_kernel(Ks, c)], p.bufs[c])
  end
  nothing
end

line_tasks_of(Ks, nt) = length(Ks) > 1 ? length(Ks) : line_tasks(Ks[1], nt)
task_kernel(Ks, c) = length(Ks) > 1 ? c : 1

# Strided lines in blocks of L neighbours: gather the block into a contiguous buffer (one line
# after the other), transform each line there, scatter back. Reading the lines one by one would
# use one value per cache line and no SIMD.
function r2r_dim_blocked!(py::Ptr{T}, px::Ptr{T}, S::Int, outer::Int, n::Int, Ks, p) where {T}
  L = r2r_block(T)
  nb = cld(S, L)
  each_line(nb * outer, n * L, line_tasks_of(Ks, p.nthreads)) do c, l
    local K = Ks[task_kernel(Ks, c)]
    local o, b
    o, b = divrem(l, nb)
    local i0 = b * L
    local w = min(L, S - i0)
    local B = p.bufs[c]
    local blk = B.blk
    local base = i0 + S * n * o
    GC.@preserve blk begin
      local pb = pointer(blk)
      @inbounds for j in 0:n-1
        local src = px + (base + S * j) * sizeof(T)
        for q in 0:w-1
          unsafe_store!(pb, unsafe_load(src, q + 1), q * n + j + 1)
        end
      end
      for q in 0:w-1
        local line = StridedLine{T}(pb + q * n * sizeof(T), 1, n)
        r2r_line!(line, line, K, B)
      end
      @inbounds for j in 0:n-1
        local dst = py + (base + S * j) * sizeof(T)
        for q in 0:w-1
          unsafe_store!(dst, unsafe_load(pb, q * n + j + 1), q + 1)
        end
      end
    end
  end
  nothing
end

# DCT-IV and DST-IV of one even line on several threads, as a complex DFT of half the length. With
# z_m = (x_2m + i x_{n-1-2m}) e^{-iπm/n} and Z = DFT_{n/2}(z), the outputs are y_2k = 2 Re w_k and
# y_{n-1-2k} = -2 Im w_k, w_k = Z_k e^{-iπ(4k+1)/(4n)} (FFTW's REDFT11 scaling). DST-IV is the DCT-IV of the
# reversed input times (-1)^k. Checked against FFTW: relerr 2e-16 for n = 8…64.
struct HalfIV{T,F}
  n::Int
  dst::Bool
  z::Vector{Complex{T}}         # pre-twiddled input, length n ÷ 2
  zt::Vector{Complex{T}}        # its DFT
  pre::Vector{Complex{T}}       # e^{-iπm/n}
  post::Vector{Complex{T}}      # 2 e^{-iπ(4k+1)/(4n)}
  cplan::F
end

function half_iv(::Type{T}, n::Int, dst::Bool, z::Vector{Complex{T}}, nt::Int) where {T}
  M = n ÷ 2
  pre = Complex{T}[Complex{T}(cis(-π * m / n)) for m in 0:M-1]
  post = Complex{T}[2 * Complex{T}(cis(-π * (4k + 1) / (4n))) for k in 0:M-1]
  cplan = FFTLPlan{T}((M,), (1,), -1; threads = nt)
  HalfIV{T,typeof(cplan)}(n, dst, z, zeros(Complex{T}, M), pre, post, cplan)
end

half_dft!(py::Ptr{T}, px::Ptr{T}, H::HalfIV{T}, nt::Int) where {T} = half_iv!(py, px, H, nt)

function half_iv!(py::Ptr{T}, px::Ptr{T}, H::HalfIV{T}, nt::Int) where {T}
  n = H.n; M = n ÷ 2; dst = H.dst
  z = H.z; zt = H.zt; pre = H.pre; post = H.post
  GC.@preserve H px py begin
    each_line(M, 1, nt) do c, m
      @inbounds begin
        ir = dst ? n - 1 - 2m : 2m
        ii = dst ? 2m : n - 1 - 2m
        z[m+1] = Complex{T}(unsafe_load(px, ir + 1), unsafe_load(px, ii + 1)) * pre[m+1]
      end
    end
    mul!(zt, H.cplan, z)
    each_line(M, 1, nt) do c, k
      @inbounds begin
        w = zt[k+1] * post[k+1]
        # output 2k (even) and n-1-2k (odd): the DST-IV sign (-1)^k is (+1) and (-1) there
        unsafe_store!(py, real(w), 2k + 1)
        unsafe_store!(py, dst ? imag(w) : -imag(w), n - 2k)
      end
    end
  end
  nothing
end

# All kinds of one odd line on several threads: a complex DFT of length 2n (n for the DHT) of the
# zero-padded input, with a phase before (pre) and after (post) the transform. Every formula was
# checked against FFTW for n = 3 … 1001 (relerr 4e-16). DST-III and DST-IV are the DCT of the
# reversed input times (-1)^k.
const ODD_KINDS = Dict(REDFT10 => :dct2, RODFT10 => :dst2, REDFT11 => :dct4, RODFT11 => :dst4,
                       REDFT01 => :dct3, RODFT01 => :dst3, DHT => :dht)

struct HalfOdd{T,F}
  n::Int
  L::Int
  kind::Symbol
  oidx::Int                    # output index offset of the DFT (1 for the DST-II)
  rev::Bool                    # DST-III, DST-IV: reversed input
  alt::Bool                    # DST-III, DST-IV: output signs (-1)^k
  pre::Vector{Complex{T}}      # per input sample, length n
  post::Vector{Complex{T}}     # per output sample, length n
  w::Vector{Complex{T}}        # zero-padded input, length L
  out::Vector{Complex{T}}      # its DFT
  cplan::F
end

function half_odd(::Type{T}, n::Int, kind::Symbol, nt::Int) where {T}
  L = kind == :dht ? n : 2n
  σ = (kind == :dct3 || kind == :dst3) ? 1 : -1
  pre = Vector{Complex{T}}(undef, n)
  post = Vector{Complex{T}}(undef, n)
  for j in 0:n-1
    pre[j+1] = kind in (:dct4, :dst4) ? Complex{T}(cis(-π * j / (2n))) :
               kind in (:dct3, :dst3) ? (j == 0 ? one(Complex{T}) : 2 * Complex{T}(cis(π * j / (2n)))) : one(Complex{T})
    post[j+1] = kind == :dct2 ? Complex{T}(cis(-π * j / (2n))) :
                kind == :dst2 ? Complex{T}(cis(-π * (j + 1) / (2n))) :
                kind in (:dct4, :dst4) ? Complex{T}(cis(-π * (2j + 1) / (4n))) : one(Complex{T})
  end
  cplan = FFTLPlan{T}((L,), (1,), σ; threads = nt)
  HalfOdd{T,typeof(cplan)}(n, L, kind, kind == :dst2 ? 1 : 0, kind in (:dst3, :dst4), kind in (:dst3, :dst4),
                           pre, post, zeros(Complex{T}, L), zeros(Complex{T}, L), cplan)
end

half_dft!(py::Ptr{T}, px::Ptr{T}, H::HalfOdd{T}, nt::Int) where {T} = half_odd!(py, px, H, nt)

function half_odd!(py::Ptr{T}, px::Ptr{T}, H::HalfOdd{T}, nt::Int) where {T}
  n = H.n; L = H.L; kind = H.kind; w = H.w; out = H.out; pre = H.pre; post = H.post
  GC.@preserve H px py begin
    each_line(L, 1, nt) do c, j
      @inbounds begin
        if j < n
          src = H.rev ? n - 1 - j : j
          w[j+1] = unsafe_load(px, src + 1) * pre[j+1]
        else
          w[j+1] = zero(Complex{T})
        end
      end
    end
    mul!(out, H.cplan, w)
    each_line(n, 1, nt) do c, k
      @inbounds begin
        v = post[k+1] * out[k + H.oidx + 1]
        y = kind == :dst2 ? -2 * imag(v) : kind == :dht ? real(v) - imag(v) :
            kind in (:dct3, :dst3) ? real(v) : 2 * real(v)
        H.alt && isodd(k) && (y = -y)
        unsafe_store!(py, y, k + 1)
      end
    end
  end
  nothing
end

# DCT-III and DST-III of one even line on several threads, from two half-length complex DFTs:
# E_p = Re DFT(x_2q e^{-iπq/n})_p and O_p = Re(e^{-2πip/n} DFT(x_2q+1 e^{-iπ(2q+1)/(2n)})_p) give
# y_2p = 2(E_p + O_p) - x_0 and y_{n-1-2p} = 2(E_p - O_p) - x_0. DST-III is the DCT-III of the reversed
# input times (-1)^k. Checked against FFTW: relerr 3e-16 for n = 8…64.
struct HalfIII{T,F}
  n::Int
  dst::Bool
  a::Vector{Complex{T}}          # pre-twiddled even samples, then their DFT
  b::Vector{Complex{T}}          # pre-twiddled odd samples, then their DFT
  pre_a::Vector{Complex{T}}      # e^{-iπq/n}
  pre_b::Vector{Complex{T}}      # e^{-iπ(2q+1)/(2n)}
  post::Vector{Complex{T}}       # e^{-2πip/n}
  x0::Base.RefValue{T}           # the first input value (after reversal), kept for the output pass
  cplan::F
end

function half_iii(::Type{T}, n::Int, dst::Bool, nt::Int) where {T}
  M = n ÷ 2
  pre_a = Complex{T}[Complex{T}(cis(-π * q / n)) for q in 0:M-1]
  pre_b = Complex{T}[Complex{T}(cis(-π * (2q + 1) / (2n))) for q in 0:M-1]
  post = Complex{T}[Complex{T}(cis(-2π * p / n)) for p in 0:M-1]
  cplan = FFTLPlan{T}((M,), (1,), -1; threads = nt)
  HalfIII{T,typeof(cplan)}(n, dst, zeros(Complex{T}, M), zeros(Complex{T}, M), pre_a, pre_b, post,
                            Ref(zero(T)), cplan)
end

half_dft!(py::Ptr{T}, px::Ptr{T}, H::HalfIII{T}, nt::Int) where {T} = half_iii!(py, px, H, nt)

function half_iii!(py::Ptr{T}, px::Ptr{T}, H::HalfIII{T}, nt::Int) where {T}
  n = H.n; M = n ÷ 2; dst = H.dst
  a = H.a; b = H.b; pa = H.pre_a; pb = H.pre_b; post = H.post
  GC.@preserve H px py begin
    # input index of the (possibly reversed) sample j
    rd(j) = dst ? n - 1 - j : j
    each_line(M, 1, nt) do c, q
      @inbounds begin
        a[q+1] = unsafe_load(px, rd(2q) + 1) * pa[q+1]
        b[q+1] = unsafe_load(px, rd(2q + 1) + 1) * pb[q+1]
      end
    end
    H.x0[] = unsafe_load(px, rd(0) + 1)
    mul!(a, H.cplan, a)
    mul!(b, H.cplan, b)
    x0 = H.x0[]
    each_line(M, 1, nt) do c, p
      @inbounds begin
        E = real(a[p+1])
        O = real(post[p+1] * b[p+1])
        unsafe_store!(py, 2 * (E + O) - x0, 2p + 1)
        v = 2 * (E - O) - x0
        unsafe_store!(py, dst ? -v : v, n - 2p)     # output index n-1-2p is odd for even n
      end
    end
  end
  nothing
end

# DCT-I and DST-I of one line on several threads: the real even (DCT-I, length 2(n-1)) or odd (DST-I,
# length 2(n+1)) extension of the input, transformed by one complex DFT of that length. The real or the
# negated imaginary part of the DFT is the result. Checked against FFTW for n = 16 … 300000.
struct HalfI{T,F}
  n::Int
  dst::Bool
  w::Vector{Complex{T}}          # the extended input, length m
  out::Vector{Complex{T}}        # its DFT
  cplan::F
end

function half_i(::Type{T}, n::Int, dst::Bool, nt::Int) where {T}
  m = 2 * (dst ? n + 1 : n - 1)
  cplan = FFTLPlan{T}((m,), (1,), -1; threads = nt)
  HalfI{T,typeof(cplan)}(n, dst, zeros(Complex{T}, m), zeros(Complex{T}, m), cplan)
end

half_dft!(py::Ptr{T}, px::Ptr{T}, H::HalfI{T}, nt::Int) where {T} = half_i!(py, px, H, nt)

function half_i!(py::Ptr{T}, px::Ptr{T}, H::HalfI{T}, nt::Int) where {T}
  n = H.n; m = length(H.w); dst = H.dst; w = H.w
  GC.@preserve H px py begin
    if dst
      # odd extension about 0 and n+1: w_j = x_{j-1} for 1 ≤ j ≤ n, w_{m-j} = -w_j, zeros at 0 and n+1
      each_line(m, 1, nt) do c, j
        @inbounds begin
          if j == 0 || j == n + 1
            w[j+1] = 0
          elseif j <= n
            w[j+1] = unsafe_load(px, j) + 0im
          elseif j < m
            w[j+1] = -(unsafe_load(px, m - j - 1 + 1) + 0im)
          end
        end
      end
    else
      # even extension about 0 and n-1: w_j = x_j for j ≤ n-1, w_{m-j} = w_j
      each_line(m, 1, nt) do c, j
        @inbounds begin
          src = j < n ? j : m - j
          w[j+1] = unsafe_load(px, src + 1) + 0im
        end
      end
    end
    mul!(H.out, H.cplan, w)
    out = H.out
    each_line(n, 1, nt) do c, k
      @inbounds unsafe_store!(py, dst ? -imag(out[k+2]) : real(out[k+1]), k + 1)
    end
  end
  nothing
end

function half_dft!(py::Ptr{T}, px::Ptr{T}, H::HalfDFT{T}, nt::Int) where {T}
  S, n, outer = H.S, H.n, H.outer
  M = n ÷ 2
  sgn = H.dst ? -one(T) : one(T)
  z = H.z
  GC.@preserve H begin
    pz = cptr(z)
    # z_m = r_2m + i r_2m+1 with r_j = x_2j (j < M), r_j = ±x_2(n-1-j)+1 (j >= M, minus for DST-II);
    # DHT: r = x
    dht = H.dht
    each_line(M * outer, S, nt) do c, l
      local o, m
      o, m = divrem(l, M)
      local ja = 2m
      local jb = 2m + 1
      local ra = dht ? ja : ja < M ? 2ja : 2(n - 1 - ja) + 1
      local rb = dht ? jb : jb < M ? 2jb : 2(n - 1 - jb) + 1
      local sa = ja < M ? one(T) : sgn
      local sb = jb < M ? one(T) : sgn
      local xa = px + S * (ra + n * o) * sizeof(T)
      local xb = px + S * (rb + n * o) * sizeof(T)
      local zr = pz + S * (m + M * o) * 2sizeof(T)
      @inbounds @simd for i in 1:S
        unsafe_store!(Ptr{Complex{T}}(zr), Complex{T}(sa * unsafe_load(xa, i), sb * unsafe_load(xb, i)), i)
      end
    end
    mul!(z, H.cplan, z)
    # half-spectrum split R_k = (C_k + C*_(M-k))/2 - i/2 · W_n^k (C_k - C*_(M-k)), then
    # y_k = 2 Re(t_k R_k), y_(n-k) = -2 Im(t_k R_k); for DST-II both written to n-1-index
    each_line((M + 1) * outer, S, nt) do c, l
      local o, k
      o, k = divrem(l, M + 1)
      local w = H.w[k+1]
      local t = H.t[k+1]
      local c1 = Ptr{Complex{T}}(pz + S * (mod(k, M) + M * o) * 2sizeof(T))
      local c2 = Ptr{Complex{T}}(pz + S * (mod(M - k, M) + M * o) * 2sizeof(T))
      local ka = H.dst ? n - 1 - k : k
      local ya = py + S * (ka + n * o) * sizeof(T)
      if dht
        local yb = py + S * (n - k + n * o) * sizeof(T)
        if 0 < k < M
          @inbounds @simd for i in 1:S
            local a = unsafe_load(c1, i)
            local b = conj(unsafe_load(c2, i))
            local R = (a + b) / 2 + (a - b) * w
            unsafe_store!(ya, real(R) - imag(R), i)
            unsafe_store!(yb, real(R) + imag(R), i)
          end
        else
          @inbounds @simd for i in 1:S
            local a = unsafe_load(c1, i)
            local b = conj(unsafe_load(c2, i))
            local R = (a + b) / 2 + (a - b) * w
            unsafe_store!(ya, real(R) - imag(R), i)
          end
        end
      elseif 0 < k < M
        local kb = H.dst ? k - 1 : n - k
        local yb = py + S * (kb + n * o) * sizeof(T)
        @inbounds @simd for i in 1:S
          local a = unsafe_load(c1, i)
          local b = conj(unsafe_load(c2, i))
          local u = t * ((a + b) / 2 + (a - b) * w)
          unsafe_store!(ya, 2real(u), i)
          unsafe_store!(yb, -2imag(u), i)
        end
      else
        @inbounds @simd for i in 1:S
          local a = unsafe_load(c1, i)
          local b = conj(unsafe_load(c2, i))
          local u = t * ((a + b) / 2 + (a - b) * w)
          unsafe_store!(ya, 2real(u), i)
        end
      end
    end
  end
  nothing
end

function apply_r2r!(y::Array{T}, x::Array{T}, p::R2RPlan{T}, ls::Tuple, ds::Tuple, hs::Tuple) where {T}
  isempty(ls) && return y
  H = first(hs)
  if H === nothing
    r2r_dim!(y, x, first(ds), first(ls), p)
  else
    GC.@preserve x y half_dft!(pointer(y), pointer(x), H, p.nthreads)
  end
  apply_r2r!(y, y, p, Base.tail(ls), Base.tail(ds), Base.tail(hs))
end

function LinearAlgebra.mul!(y::Array{T,N}, p::R2RPlan{T,N}, x::Array{T,N}) where {T,N}
  size(x) == p.sz && size(y) == p.sz || throw(DimensionMismatch("r2r plan for size $(p.sz)"))
  isempty(p.region) && return copyto!(y, x)
  apply_r2r!(y, x, p, p.lines, p.region, p.halves)
end

Base.:*(p::R2RPlan{T,N}, x::AbstractArray{<:Real,N}) where {T,N} =
  mul!(Array{T,N}(undef, p.sz), p, convert(Array{T,N}, x))

# inverse kinds and the scaling of the unnormalised pair
inverse_kind(k::R2RKind) = k == REDFT10 ? REDFT01 : k == REDFT01 ? REDFT10 :
                           k == RODFT10 ? RODFT01 : k == RODFT01 ? RODFT10 :
                           k == R2HC ? HC2R : k == HC2R ? R2HC : k
r2r_scale(k::R2RKind, n) = k == REDFT00 ? 2(n - 1) : k == RODFT00 ? 2(n + 1) : k in (DHT, R2HC, HC2R) ? n : 2n

function AbstractFFTs.plan_inv(p::R2RPlan{T}) where {T}
  s = prod(r2r_scale(k, p.sz[d]) for (k, d) in zip(p.kinds, p.region); init = 1)
  AbstractFFTs.ScaledPlan(R2RPlan{T}(p.sz, map(inverse_kind, p.kinds), p.region; threads = p.nthreads), inv(T(s)))
end

"""
    plan_r2r(x::AbstractArray{<:Real}, kinds, region = 1:ndims(x); threads, num_threads, flags, timelimit)
    r2r(x, kinds, region = 1:ndims(x))

Unnormalised real-to-real transforms; `kinds` is one `R2RKind` (REDFT00 … RODFT11, DHT, R2HC, HC2R) for all
dimensions of `region` or one per dimension. Keywords as for [`plan_fft`](@ref).
"""
function plan_r2r(x::AbstractArray, kinds, region = 1:ndims(x); threads = nothing, num_threads = nothing,
                  flags = nothing, timelimit = nothing, kw...)
  generic_float(elreal(x)) && return GR2RPlan{elreal(x)}(size(x), kinds, region)
  R2RPlan{realtype(x)}(size(x), kinds, region; threads = planthreads(threads, num_threads), kw...)
end
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
  y = copyto!(similar(x, generic_float(elreal(x)) ? elreal(x) : realtype(x)), x)   # stays on its device
  for d in (region isa Integer ? (region,) : region)
    n = size(y, d)
    selectdim(y, d, 1) .*= sqrt(eltype(y)(2))
    y .*= inv(sqrt(eltype(y)(2n)))
  end
  r2r(y, REDFT01, region)
end

"""
    OrthoDCTPlan

Plan of the orthonormal DCT-II (`plan_dct`) or its inverse (`plan_idct`): an unnormalised
REDFT10 or REDFT01 plan and the scaling that makes it orthonormal. The scaling is not uniform
(the first element of each transformed dimension has its own factor), so a `ScaledPlan` cannot
express it.
"""
mutable struct OrthoDCTPlan{T,P<:AbstractFFTs.Plan{T}} <: AbstractFFTs.Plan{T}
  p::P
  region::Vector{Int}
  inverse::Bool
  pinv::AbstractFFTs.Plan{T}
  OrthoDCTPlan(p::AbstractFFTs.Plan{T}, region, inverse::Bool) where {T} =
    new{T,typeof(p)}(p, collect(Int, region isa Integer ? (region,) : region), inverse)
end

"""
    plan_dct(x, region = 1:ndims(x); …), plan_idct(x, region = 1:ndims(x); …)
    plan_dct!(x, region = 1:ndims(x); …), plan_idct!(x, region = 1:ndims(x); …)
    dct!(x, region = 1:ndims(x)), idct!(x, region = 1:ndims(x))

Plans of the orthonormal DCT-II and its inverse (the scaling of [`dct`](@ref)), and their
in-place forms. Keywords as for [`plan_fft`](@ref).
"""
plan_dct(x::AbstractArray, region = 1:ndims(x); kw...) = OrthoDCTPlan(plan_r2r(x, REDFT10, region; kw...), region, false)
plan_idct(x::AbstractArray, region = 1:ndims(x); kw...) = OrthoDCTPlan(plan_r2r(x, REDFT01, region; kw...), region, true)
plan_dct!(x::AbstractArray, region = 1:ndims(x); kw...) = InplacePlan(plan_dct(x, region; kw...))
plan_idct!(x::AbstractArray, region = 1:ndims(x); kw...) = InplacePlan(plan_idct(x, region; kw...))
dct!(x::AbstractArray, region = 1:ndims(x)) = plan_dct!(x, region) * x
idct!(x::AbstractArray, region = 1:ndims(x)) = plan_idct!(x, region) * x

Base.size(q::OrthoDCTPlan) = size(q.p)
Base.ndims(q::OrthoDCTPlan) = ndims(q.p)
AbstractFFTs.fftdims(q::OrthoDCTPlan) = q.region
AbstractFFTs.plan_inv(q::OrthoDCTPlan) =
  OrthoDCTPlan(plan_r2r(Array{eltype(q)}(undef, size(q)), q.inverse ? REDFT10 : REDFT01, q.region), q.region, !q.inverse)
Base.show(io::IO, q::OrthoDCTPlan) = print(io, q.inverse ? "orthonormal inverse DCT (DCT-III) plan, " :
                                           "orthonormal DCT-II plan, ", join(size(q), "×"), ", dims ", q.region)

function Base.:*(q::OrthoDCTPlan{T}, x::AbstractArray) where {T}
  if q.inverse
    y = copyto!(similar(x, T), x)
    for d in q.region
      selectdim(y, d, 1) .*= sqrt(T(2))
      y .*= inv(sqrt(T(2size(y, d))))
    end
    q.p * y
  else
    y = q.p * x
    for d in q.region
      y .*= inv(sqrt(T(2size(y, d))))
      selectdim(y, d, 1) ./= sqrt(T(2))
    end
    y
  end
end
LinearAlgebra.mul!(y::AbstractArray, q::OrthoDCTPlan, x::AbstractArray) = copyto!(y, q * x)

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

Plan for the real-to-real transform `kind` (REDFT00 … RODFT11, DHT, R2HC, HC2R; unnormalised) along
dimension `d` of all lines of arrays with the size and strides of `v`. Apply in place with
`apply!(p, v)` or into an array of the same layout with `mul!(w, p, v)`.
"""
function plan_r2r_pass(v::StridedArray{T,N}, d::Integer, kind::R2RKind; threads::Integer = 1) where {T<:FFTLFloat,N}
  1 <= d <= N || throw(ArgumentError("dimension $d is not within 1:$N"))
  sz = size(v); st = strides(v)
  outer = Tuple(i for i in 1:N if i != d)
  offs = line_offsets(sz, st, collect(Int, outer))
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
    each_line(length(p.offs), n, line_tasks(p.line, p.nthreads)) do c, l
      local off = p.offs[l+1] * sizeof(T)
      r2r_line!(StridedLine{T}(pw + off, sd, n), StridedLine{T}(pv + off, sd, n), p.line, p.bufs[c])
    end
  end
  w
end

apply!(p::R2RPassPlan, v) = mul!(v, p, v)
