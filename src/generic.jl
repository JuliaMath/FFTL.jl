# Transforms for element types beyond the hardware floats (BigFloat, Double64, Float128, …):
# plain Julia on arrays, no SIMD and no pointers (BigFloat is not an isbits type), single-threaded.
# Accuracy over speed: twiddles are computed in T, smooth lengths go through a recursive
# mixed-radix FFT, lengths with a prime factor above MAX_RADIX_PRIME through Bluestein's
# algorithm. Real transforms reduce to complex DFTs: r2c/c2r through the full complex line, the
# real-to-real kinds through a complex DFT of length 2n (2(n-1) for DCT-I, 2(n+1) for DST-I)
# with pre- and post-twiddles:
#   DCT-I    y_k = Re Z_k,  z_j = c_j x_j (c_0 = c_(n-1) = 1, else 2), length 2(n-1)
#   DCT-II   y_k = Re[2 e^(-iπk/2n) Z_k],  z_j = x_j
#   DCT-III  y_k = Re Z_k,  z_j = c_j x_j e^(-iπj/2n) (c_0 = 1, else 2)
#   DCT-IV   y_k = Re[2 e^(-iπ(k+½)/2n) Z_k],  z_j = x_j e^(-iπj/2n)
#   DST-I    y_k = -2 Im Z_(k+1),  z_(j+1) = x_j, length 2(n+1)
#   DST-II   y_k = -Im[2 e^(-iπ(k+1)/2n) Z_(k+1)],  z_j = x_j
#   DST-III  y_k = -Im Z_k,  z_m = c_m x_(m-1) e^(-iπm/2n), m = 1…n (c_n = 1, else 2)
#   DST-IV   y_k = -Im[2 e^(-iπ(k+½)/2n) Z_k],  z_j = x_j e^(-iπj/2n)
# with Z = DFT (sign -1) of z zero padded; DHT, R2HC and HC2R through the DFT of length n.

"""
    generic_float(T)

Whether FFTL transforms arrays of element type `T` (or `Complex{T}`) in `T` itself through the
generic path: every `AbstractFloat` except Float16 (converted to Float64) and the hardware
types Float32 and Float64 (the SIMD path).
"""
generic_float(::Type{T}) where {T} = T <: AbstractFloat && !(T <: Union{Float16,Float32,Float64})

# exp(σ 2πi e / n) for e = 0…n-1, from angles at most π
function generic_roots(::Type{T}, n::Int, σ::Int) where {T}
  w = Vector{Complex{T}}(undef, n)
  for e in 0:n-1
    m = min(e, n - e)
    s, c = sincos(T(2) * T(π) * m / n)
    v = Complex{T}(c, s)
    w[e+1] = (e <= n - e) == (σ > 0) ? v : conj(v)
  end
  w
end

"""
    GLine{T}

One-dimensional DFT of length `n`, sign `σ`, for a generic float type: mixed radix, or Bluestein
when a prime factor exceeds MAX_RADIX_PRIME.
"""
struct GLine{T}
  n::Int
  σ::Int
  rs::Vector{Int}                   # radices (empty for Bluestein)
  w::Vector{Complex{T}}             # exp(σ 2πi e / n)
  tmp::Vector{Complex{T}}           # one butterfly
  chirp::Vector{Complex{T}}         # Bluestein: c_j = exp(σ iπ j² / n)
  bhat::Vector{Complex{T}}          # Bluestein: FFT_M of the conjugate chirp, divided by M
  inner::Vector{GLine{T}}           # Bluestein: forward and backward DFT of length M
  buf::Vector{Complex{T}}           # Bluestein: work vectors of length M
  buf2::Vector{Complex{T}}
end

function GLine{T}(n::Int, σ::Int) where {T}
  n >= 1 || throw(ArgumentError("length must be positive"))
  f, rest = factorize(n)
  if rest == 1
    rs = Int[]
    for p in reverse!(sort!(collect(keys(f)))), _ in 1:f[p]
      push!(rs, p)
    end
    pmax = maximum(rs; init = 1)
    return GLine{T}(n, σ, rs, generic_roots(T, n, σ), Vector{Complex{T}}(undef, pmax),
                    Complex{T}[], Complex{T}[], GLine{T}[], Complex{T}[], Complex{T}[])
  end
  M = nextpow(2, 2n - 1)
  w2 = generic_roots(T, 2n, σ)
  c = [w2[mod(j * j, 2n)+1] for j in 0:n-1]
  fwd = GLine{T}(M, -1); bwd = GLine{T}(M, 1)
  b = zeros(Complex{T}, M)
  b[1] = conj(c[1])
  for j in 1:n-1
    b[j+1] = b[M-j+1] = conj(c[j+1])
  end
  bhat = similar(b)
  dft!(bhat, b, fwd)
  bhat ./= M
  GLine{T}(n, σ, Int[], Complex{T}[], Complex{T}[], c, bhat, [fwd, bwd], Vector{Complex{T}}(undef, M), Vector{Complex{T}}(undef, M))
end

# y[yo+k] = Σ_j x[xo + j xs] w_n^(σ j k), k = 0…n-1 (y and x distinct); n = L.n / step
function fft_rec!(y, yo::Int, x, xo::Int, xs::Int, n::Int, L::GLine, ri::Int)
  if n == 1
    y[yo] = x[xo]
    return
  end
  p = L.rs[ri]; m = n ÷ p
  for r in 0:p-1
    fft_rec!(y, yo + r * m, x, xo + r * xs, xs * p, m, L, ri + 1)
  end
  N = L.n; s = N ÷ n; t = L.tmp; w = L.w
  @inbounds for k in 0:m-1
    for r in 0:p-1
      t[r+1] = w[r*k*s+1] * y[yo+r*m+k]
    end
    for q in 0:p-1
      acc = t[1]
      for r in 1:p-1
        acc += t[r+1] * w[mod(r * q, p)*(N÷p)+1]
      end
      y[yo+q*m+k] = acc
    end
  end
  nothing
end

# y = DFT of x (vectors of length L.n, distinct)
function dft!(y::AbstractVector, x::AbstractVector, L::GLine{T}) where {T}
  if isempty(L.inner)
    fft_rec!(y, 1, x, 1, 1, L.n, L, 1)
  else
    n = L.n; b = L.buf; M = length(b); c = L.chirp
    z = L.buf2
    for j in 1:n
      b[j] = x[j] * c[j]
    end
    for j in n+1:M
      b[j] = zero(Complex{T})
    end
    dft!(z, b, L.inner[1])
    z .*= L.bhat
    dft!(b, z, L.inner[2])
    for k in 1:n
      y[k] = b[k] * c[k]
    end
  end
  y
end

# the lines of dimension d of A (column major), each transformed by f!(line)
function each_generic_line!(f!::F, A::AbstractArray, d::Int) where {F}
  sz = size(A); n = sz[d]
  S = prod(sz[1:d-1]; init = 1); outer = prod(sz[d+1:end]; init = 1)
  line = Vector{eltype(A)}(undef, n)
  v = vec(A)
  for o in 0:outer-1, i in 0:S-1
    base = i + S * n * o
    for j in 0:n-1
      line[j+1] = v[base+S*j+1]
    end
    f!(line)
    for j in 0:n-1
      v[base+S*j+1] = line[j+1]
    end
  end
  A
end

"""
    GPlan{T,N}

Complex DFT plan for element type `Complex{T}` with `generic_float(T)`, along `region`.
"""
mutable struct GPlan{T,N} <: AbstractFFTs.Plan{Complex{T}}
  sz::NTuple{N,Int}
  region::Vector{Int}
  σ::Int
  lines::Vector{GLine{T}}
  pinv::AbstractFFTs.Plan
  GPlan{T,N}(sz, region, σ, lines) where {T,N} = new{T,N}(sz, region, σ, lines)
end

function GPlan{T}(sz::NTuple{N,Int}, region, σ::Int) where {T,N}
  (σ == 1 || σ == -1) || throw(ArgumentError("sign must be 1 or -1"))
  reg = normregion(region, N)
  GPlan{T,N}(sz, reg, σ, [GLine{T}(sz[d], σ) for d in reg])
end

Base.size(p::GPlan) = p.sz
Base.ndims(::GPlan{T,N}) where {T,N} = N
AbstractFFTs.AdjointStyle(::GPlan) = AbstractFFTs.FFTAdjointStyle()
AbstractFFTs.fftdims(p::GPlan) = Tuple(p.region)
Base.show(io::IO, p::GPlan{T}) where {T} =
  print(io, "FFTL generic plan: ", p.σ < 0 ? "forward" : "backward", " Complex{", T, "} ", join(p.sz, "×"), ", dims ", Tuple(p.region))

function apply_generic!(A::Array{Complex{T}}, p::GPlan{T}) where {T}
  for (d, L) in zip(p.region, p.lines)
    out = Vector{Complex{T}}(undef, L.n)
    each_generic_line!(A, d) do line
      dft!(out, line, L)
      copyto!(line, out)
    end
  end
  A
end

function LinearAlgebra.mul!(y::AbstractArray{Complex{T},N}, p::GPlan{T,N}, x::AbstractArray{Complex{T},N}) where {T,N}
  size(x) == p.sz && size(y) == p.sz || throw(DimensionMismatch("plan for size $(p.sz)"))
  copyto!(y, apply_generic!(Array{Complex{T},N}(x), p))
end

function Base.:*(p::GPlan{T,N}, x::AbstractArray{<:Number,N}) where {T,N}
  size(x) == p.sz || throw(DimensionMismatch("plan for size $(p.sz)"))
  apply_generic!(Array{Complex{T},N}(x), p)
end

AbstractFFTs.plan_inv(p::GPlan{T}) where {T} =
  AbstractFFTs.ScaledPlan(GPlan{T}(p.sz, p.region, -p.σ), AbstractFFTs.normalization(T, p.sz, p.region))

# --- real input and output ----------------------------------------------------------

"""
    GRPlan{T,N}

r2c (`σ = -1`, real input of size `sz`) or unnormalised c2r (`σ = +1`, half spectrum of size
`sz`, real output of length `d` along the first dimension of the region) for a generic `T`.
"""
mutable struct GRPlan{T,N} <: AbstractFFTs.Plan{T}
  sz::NTuple{N,Int}
  d::Int
  region::Vector{Int}
  σ::Int
  line::GLine{T}                    # complex DFT of length d along region[1]
  rest::Union{Nothing,GPlan{T,N}}   # the other dimensions, on the half spectrum
  pinv::AbstractFFTs.Plan
  GRPlan{T,N}(sz, d, region, σ, line, rest) where {T,N} = new{T,N}(sz, d, region, σ, line, rest)
end

function GRPlan{T}(sz::NTuple{N,Int}, d::Int, region, σ::Int) where {T,N}
  reg = normregion(region, N)
  isempty(reg) && throw(ArgumentError("a real transform needs at least one dimension"))
  hsz = σ < 0 ? Base.setindex(sz, nhalf(sz[reg[1]]), reg[1]) : sz
  σ > 0 && nhalf(d) != sz[reg[1]] && throw(DimensionMismatch("half spectrum of length $(sz[reg[1]]) does not fit length $d"))
  rest = length(reg) > 1 ? GPlan{T}(hsz, reg[2:end], σ) : nothing
  GRPlan{T,N}(sz, d, reg, σ, GLine{T}(d, σ), rest)
end

Base.size(p::GRPlan) = p.sz
Base.ndims(::GRPlan{T,N}) where {T,N} = N
LinearAlgebra.mul!(y::AbstractArray, p::GRPlan, x::AbstractArray) = copyto!(y, p * x)
AbstractFFTs.fftdims(p::GRPlan) = Tuple(p.region)
AbstractFFTs.output_size(p::GRPlan) =
  Base.setindex(p.sz, p.σ < 0 ? nhalf(p.sz[p.region[1]]) : p.d, p.region[1])

function Base.:*(p::GRPlan{T,N}, x::AbstractArray{<:Number,N}) where {T,N}
  size(x) == p.sz || throw(DimensionMismatch("plan for size $(p.sz)"))
  d1 = p.region[1]; n = p.d; h = nhalf(n); L = p.line
  out = Vector{Complex{T}}(undef, n); full = Vector{Complex{T}}(undef, n)
  if p.σ < 0
    A = Array{Complex{T},N}(x)
    each_generic_line!(A, d1) do line
      dft!(out, line, L); copyto!(line, out)
    end
    Y = A[ntuple(i -> i == d1 ? (1:h) : (:), N)...]
    p.rest === nothing ? Y : apply_generic!(Y, p.rest)
  else
    Y = Array{Complex{T},N}(x)
    p.rest === nothing || apply_generic!(Y, p.rest)
    Rc = zeros(Complex{T}, AbstractFFTs.output_size(p))   # BigFloat arrays must not hold #undef
    Rc[ntuple(i -> i == d1 ? (1:h) : (:), N)...] = Y
    each_generic_line!(Rc, d1) do line
      # the full spectrum of a real line from its half; X_0 and X_(n/2) are real
      full[1] = Complex{T}(real(line[1]))
      for k in 1:n-1
        full[k+1] = k < h ? line[k+1] : conj(line[n-k+1])
      end
      iseven(n) && (full[n÷2+1] = Complex{T}(real(line[n÷2+1])))
      dft!(out, full, L); copyto!(line, out)
    end
    real.(Rc)
  end
end

AbstractFFTs.plan_inv(p::GRPlan{T}) where {T} = p.σ < 0 ?
  AbstractFFTs.ScaledPlan(GRPlan{T}(AbstractFFTs.output_size(p), p.d, p.region, 1), AbstractFFTs.normalization(T, p.sz, p.region)) :
  AbstractFFTs.ScaledPlan(GRPlan{T}(AbstractFFTs.output_size(p), p.d, p.region, -1),
                          AbstractFFTs.normalization(T, AbstractFFTs.output_size(p), p.region))

# --- real-to-real ---------------------------------------------------------------

struct GR2RLine{T}
  kind::R2RKind
  n::Int
  L::GLine{T}                       # forward DFT of the extended length
  pre::Vector{Complex{T}}
  post::Vector{Complex{T}}
  z::Vector{Complex{T}}
  Z::Vector{Complex{T}}
end

function GR2RLine{T}(kind::R2RKind, n::Int) where {T}
  n >= 1 || throw(ArgumentError("length must be positive"))
  kind == REDFT00 && n < 2 && throw(ArgumentError("REDFT00 needs length ≥ 2"))
  len = kind == REDFT00 ? 2(n - 1) : kind == RODFT00 ? 2(n + 1) : kind in (DHT, R2HC, HC2R) ? n : 2n
  w = generic_roots(T, 4n, -1)          # exp(-iπ e / 2n)
  pre = ones(Complex{T}, n + 1); post = ones(Complex{T}, n + 1)
  if kind in (REDFT10, RODFT10)
    for k in 0:n; post[k+1] = 2w[mod(kind == RODFT10 ? k + 1 : k, 4n)+1]; end
  elseif kind in (REDFT01, RODFT01)
    for j in 0:n; pre[j+1] = w[mod(j, 4n)+1]; end
  elseif kind in (REDFT11, RODFT11)
    w8 = generic_roots(T, 8n, -1)       # exp(-iπ e / 4n)
    for j in 0:n
      pre[j+1] = w[mod(j, 4n)+1]
      post[j+1] = 2w8[mod(2j + 1, 8n)+1]
    end
  end
  z = Vector{Complex{T}}(undef, len)
  GR2RLine{T}(kind, n, GLine{T}(len, -1), pre, post, z, similar(z))
end

function r2r_generic!(y::AbstractVector{T}, x::AbstractVector{T}, G::GR2RLine{T}) where {T}
  n = G.n; k = G.kind; z = G.z; Z = G.Z
  fill!(z, zero(Complex{T}))
  if k == REDFT00
    for j in 0:n-1; z[j+1] = (j == 0 || j == n - 1 ? 1 : 2) * x[j+1]; end
  elseif k == REDFT01
    for j in 0:n-1; z[j+1] = (j == 0 ? 1 : 2) * x[j+1] * G.pre[j+1]; end
  elseif k == RODFT01
    for m in 1:n; z[m+1] = (m == n ? 1 : 2) * x[m] * G.pre[m+1]; end
  elseif k == RODFT00
    for j in 0:n-1; z[j+2] = x[j+1]; end
  elseif k == HC2R
    h = nhalf(n)
    # inverse: Y_j = Σ F_m e^(2πijm/n) = conj DFT(conj F)
    for m in 0:n-1
      F = m == 0 || 2m == n ? Complex{T}(x[m+1]) : 2m < n ? Complex{T}(x[m+1], x[n-m+1]) : Complex{T}(x[n-m+1], -x[m+1])
      z[m+1] = conj(F)
    end
  else
    for j in 0:n-1; z[j+1] = x[j+1] * G.pre[j+1]; end
  end
  dft!(Z, z, G.L)
  for i in 0:n-1
    y[i+1] = if k == REDFT00 || k == REDFT01
      real(Z[i+1])
    elseif k == REDFT10 || k == REDFT11
      real(G.post[i+1] * Z[i+1])
    elseif k == RODFT00
      -2imag(Z[i+2])
    elseif k == RODFT10
      -imag(G.post[i+1] * Z[i+2])
    elseif k == RODFT01
      -imag(Z[i+1])
    elseif k == RODFT11
      -imag(G.post[i+1] * Z[i+1])
    elseif k == DHT
      real(Z[i+1]) - imag(Z[i+1])
    elseif k == R2HC
      2i <= n ? real(Z[i+1]) : imag(Z[n-i+1])
    else # HC2R
      real(Z[i+1])
    end
  end
  y
end

"""
    GR2RPlan{T,N}

Real-to-real plan for a generic float type `T`, one kind per dimension of `region`.
"""
mutable struct GR2RPlan{T,N} <: AbstractFFTs.Plan{T}
  sz::NTuple{N,Int}
  region::Vector{Int}
  kinds::Vector{R2RKind}
  lines::Vector{GR2RLine{T}}
  pinv::AbstractFFTs.Plan
  GR2RPlan{T,N}(sz, region, kinds, lines) where {T,N} = new{T,N}(sz, region, kinds, lines)
end

function GR2RPlan{T}(sz::NTuple{N,Int}, kinds, region) where {T,N}
  reg = collect(Int, region isa Integer ? (region,) : region)
  check_r2r_region(reg, N)
  ks = kinds isa R2RKind ? fill(kinds, length(reg)) : collect(R2RKind, kinds)
  length(ks) == length(reg) || throw(ArgumentError("one kind per transformed dimension"))
  GR2RPlan{T,N}(sz, reg, ks, [GR2RLine{T}(k, sz[d]) for (k, d) in zip(ks, reg)])
end

Base.size(p::GR2RPlan) = p.sz
Base.ndims(::GR2RPlan{T,N}) where {T,N} = N
LinearAlgebra.mul!(y::AbstractArray, p::GR2RPlan, x::AbstractArray) = copyto!(y, p * x)
AbstractFFTs.fftdims(p::GR2RPlan) = Tuple(p.region)

function Base.:*(p::GR2RPlan{T,N}, x::AbstractArray{<:Real,N}) where {T,N}
  size(x) == p.sz || throw(DimensionMismatch("r2r plan for size $(p.sz)"))
  A = Array{T,N}(x)
  for (d, G) in zip(p.region, p.lines)
    out = Vector{T}(undef, G.n)
    each_generic_line!(A, d) do line
      r2r_generic!(out, line, G); copyto!(line, out)
    end
  end
  A
end

function AbstractFFTs.plan_inv(p::GR2RPlan{T}) where {T}
  s = prod(r2r_scale(k, p.sz[d]) for (k, d) in zip(p.kinds, p.region); init = 1)
  AbstractFFTs.ScaledPlan(GR2RPlan{T}(p.sz, map(inverse_kind, p.kinds), p.region), inv(T(s)))
end
