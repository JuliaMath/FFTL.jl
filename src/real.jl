# Real-input (r2c) and real-output (c2r) transforms along the first dimension of a region; the
# other dimensions of the region are complex passes on the half spectrum.
#
# Even length n = 2M: a real line read as M complex values z_j = x_(2j) + i·x_(2j+1) (the memory
# layout as is) gets a complex DFT of length M, Z, and
#   r2c:  X_k = E_k + w^k·O_k,  E_k = (Z_k + conj Z_(M-k))/2,  O_k = (Z_k - conj Z_(M-k))/(2i),
#   c2r:  Z_k = (X_k + conj X_(M-k)) + i·w^k·(X_k - conj X_(M-k)),  then z = DFT_M(Z),
# with w = exp(σ·2πi/n) and Z_M = Z_0. Odd lengths use a complex DFT of length n.

"""
    RealLine{T}

Kernel for r2c (σ = -1) or c2r (σ = +1) transforms of contiguous real lines of length `n`
(half spectrum of `n ÷ 2 + 1` values).
"""
struct RealLine{T,K<:Kernel{T}}
  n::Int
  σ::Int
  inner::K                     # length n ÷ 2 (even n) or n (odd n)
  w::Vector{Complex{T}}        # w^k, k = 0…n÷2 (even n)
end

function RealLine{T}(n::Int, σ::Int) where {T}
  inner = kernel(T, iseven(n) ? n ÷ 2 : n, σ)
  w = iseven(n) ? [Complex{T}(cispi(2σ * k / n)) for k in 0:n÷2] : Complex{T}[]
  RealLine{T,typeof(inner)}(n, σ, inner, w)
end

nhalf(n) = n ÷ 2 + 1

# r2c of one line: x (n reals) → y (n÷2+1 complex); b1, b2, b3: complex buffers of n values.
function r2c_line!(py::Ptr{Complex{T}}, px::Ptr{T}, b1::Ptr{T}, b2::Ptr{T}, b3::Ptr{T}, K::RealLine{T}) where {T}
  n = K.n
  if iseven(n)
    M = n ÷ 2
    run!(Ptr{T}(py), px, b1, b2, K.inner, 1)
    @inbounds begin
      z0 = unsafe_load(py, 1)
      unsafe_store!(py, Complex(real(z0) + imag(z0), zero(T)), 1)
      unsafe_store!(py, Complex(real(z0) - imag(z0), zero(T)), M + 1)
      V = single_vec(T); CB = sizeof(Complex{T})
      half = V((T(0.5), T(0.5)))
      GC.@preserve K for k in 1:M÷2
        zk = vload(V, Ptr{T}(py + k * CB)); czm = cconj(vload(V, Ptr{T}(py + (M - k) * CB)))
        e = (zk + czm) * half
        ok = fcadd270(zero(V), (zk - czm) * half)                 # (Z_k - conj Z_(M-k))/(2i)
        wk = vload(V, Ptr{T}(pointer(K.w) + k * CB)); wm = vload(V, Ptr{T}(pointer(K.w) + (M - k) * CB))
        # the same pair gives E_(M-k) = conj(e), O_(M-k) = conj(ok)
        vstore(e + ccmul(wk, ok), Ptr{T}(py + k * CB))
        vstore(cconj(e) + ccmul(wm, cconj(ok)), Ptr{T}(py + (M - k) * CB))
      end
    end
  else
    cb1 = Ptr{Complex{T}}(b1)
    for j in 1:n
      unsafe_store!(cb1, Complex(unsafe_load(px, j), zero(T)), j)
    end
    run!(b1, b1, b2, b3, K.inner, 1)
    unsafe_copyto!(py, cb1, nhalf(n))
  end
  nothing
end

# c2r of one line: x (n÷2+1 complex) → y (n reals); the input is not modified. The imaginary
# parts of X_0 and (n even) X_(n/2) are ignored: a half spectrum of real data has none.
function c2r_line!(py::Ptr{T}, px::Ptr{Complex{T}}, b1::Ptr{T}, b2::Ptr{T}, b3::Ptr{T}, K::RealLine{T}) where {T}
  n = K.n
  cb1 = Ptr{Complex{T}}(b1)
  if iseven(n)
    M = n ÷ 2
    let x0 = real(unsafe_load(px, 1)), xM = real(unsafe_load(px, M + 1))
      unsafe_store!(cb1, Complex(x0 + xM, x0 - xM), 1)          # k = 0: real parts only
    end
    V = single_vec(T); CB = sizeof(Complex{T})
    GC.@preserve K for k in 1:M-1
      xk = vload(V, Ptr{T}(px + k * CB)); xm = cconj(vload(V, Ptr{T}(px + (M - k) * CB)))
      wk = vload(V, Ptr{T}(pointer(K.w) + k * CB))
      vstore((xk + xm) + ccmul(wk, fcadd90(zero(V), xk - xm)), Ptr{T}(cb1 + k * CB))   # + i·w^k·(xk - xm)
    end
    run!(py, b1, b2, b3, K.inner, 1)
  else
    h = nhalf(n)
    @inbounds for k in 1:h
      unsafe_store!(cb1, unsafe_load(px, k), k)
    end
    unsafe_store!(cb1, Complex(real(unsafe_load(px, 1)), zero(T)), 1)
    @inbounds for k in 1:n-h
      unsafe_store!(cb1, conj(unsafe_load(px, k + 1)), n - k + 1)
    end
    run!(b1, b1, b2, b3, K.inner, 1)
    @inbounds for j in 1:n
      unsafe_store!(py, real(unsafe_load(cb1, j)), j)
    end
  end
  nothing
end

# Lines along dimension d of an array of size `sz` (real side), as (lane stride S, line offsets
# in elements of the real side and of the complex side).
function line_layout(sz, d)
  S, outer = dim_extent(sz, d)
  n = sz[d]; h = nhalf(n)
  offs_r = Int[]; offs_c = Int[]
  for o in 0:outer-1, i in 0:S-1
    push!(offs_r, i + S * n * o); push!(offs_c, i + S * h * o)
  end
  S, offs_r, offs_c
end

"""
    RFFTLPlan{T,N}

Plan for the r2c DFT (forward) of real arrays of size `sz` along `region`: real along the first
dimension of the region (output length n ÷ 2 + 1 there), complex along the others.
"""
mutable struct RFFTLPlan{T<:FFTLFloat,N,K,C,D} <: AbstractFFTs.Plan{T}
  sz::NTuple{N,Int}
  region::NTuple{D,Int}
  line::K
  cplan::C                          # complex passes on the half spectrum, or nothing
  S::Int
  offs_r::Vector{Int}
  offs_c::Vector{Int}
  nthreads::Int
  bufs::Vector{Vector{Complex{T}}}  # three per thread, plus line buffers for strided lines
  pinv::AbstractFFTs.Plan
  RFFTLPlan{T,N,K,C,D}(sz, region, line, cplan, S, offs_r, offs_c, nt, bufs) where {T,N,K,C,D} =
    new{T,N,K,C,D}(sz, region, line, cplan, S, offs_r, offs_c, nt, bufs)
end

"""
    BRFFTLPlan{T,N}

Plan for the unnormalised c2r DFT (backward) from half spectra of size `sz` to real arrays with
length `d` along the first dimension of `region`.
"""
mutable struct BRFFTLPlan{T<:FFTLFloat,N,K,C,D} <: AbstractFFTs.Plan{Complex{T}}
  sz::NTuple{N,Int}
  d::Int
  region::NTuple{D,Int}
  line::K
  cplan::C
  S::Int
  offs_r::Vector{Int}
  offs_c::Vector{Int}
  nthreads::Int
  bufs::Vector{Vector{Complex{T}}}
  copy::Vector{Complex{T}}          # the input, transformed along the complex dimensions
  pinv::AbstractFFTs.Plan
  BRFFTLPlan{T,N,K,C,D}(sz, d, region, line, cplan, S, offs_r, offs_c, nt, bufs, copy) where {T,N,K,C,D} =
    new{T,N,K,C,D}(sz, d, region, line, cplan, S, offs_r, offs_c, nt, bufs, copy)
end

sortregion(region, N) = (reg = Tuple(sort(unique(collect(Int, region)))); all(d -> 1 <= d <= N, reg) ||
  throw(ArgumentError("region $region is not within 1:$N")); reg)

function RFFTLPlan{T}(sz::NTuple{N,Int}, region; threads::Integer = Threads.nthreads()) where {T<:FFTLFloat,N}
  reg = sortregion(region, N)
  isempty(reg) && throw(ArgumentError("rfft needs at least one dimension"))
  d1 = reg[1]; n = sz[d1]
  line = RealLine{T}(n, -1)
  csz = Base.setindex(sz, nhalf(n), d1)
  cplan = length(reg) > 1 ? FFTLPlan{T}(csz, reg[2:end], -1; threads) : nothing
  S, offs_r, offs_c = line_layout(sz, d1)
  nt = max(1, Int(threads))
  bufs = [Vector{Complex{T}}(undef, n) for _ in 1:5nt]
  RFFTLPlan{T,N,typeof(line),typeof(cplan),length(reg)}(sz, reg, line, cplan, S, offs_r, offs_c, nt, bufs)
end

function BRFFTLPlan{T}(sz::NTuple{N,Int}, d::Int, region; threads::Integer = Threads.nthreads()) where {T<:FFTLFloat,N}
  reg = sortregion(region, N)
  isempty(reg) && throw(ArgumentError("brfft needs at least one dimension"))
  d1 = reg[1]
  nhalf(d) == sz[d1] || throw(DimensionMismatch("half spectrum of length $(sz[d1]) does not fit length $d"))
  line = RealLine{T}(d, 1)
  cplan = length(reg) > 1 ? FFTLPlan{T}(sz, reg[2:end], 1; threads) : nothing
  rsz = Base.setindex(sz, d, d1)
  S, offs_r, offs_c = line_layout(rsz, d1)
  nt = max(1, Int(threads))
  bufs = [Vector{Complex{T}}(undef, d) for _ in 1:5nt]
  BRFFTLPlan{T,N,typeof(line),typeof(cplan),length(reg)}(sz, d, reg, line, cplan, S, offs_r, offs_c, nt, bufs,
                                              Vector{Complex{T}}(undef, prod(sz)))
end

Base.size(p::RFFTLPlan) = p.sz
Base.size(p::BRFFTLPlan) = p.sz
Base.ndims(::RFFTLPlan{T,N}) where {T,N} = N
Base.ndims(::BRFFTLPlan{T,N}) where {T,N} = N
AbstractFFTs.fftdims(p::Union{RFFTLPlan,BRFFTLPlan}) = p.region
AbstractFFTs.AdjointStyle(::RFFTLPlan) = AbstractFFTs.RFFTAdjointStyle()
AbstractFFTs.AdjointStyle(p::BRFFTLPlan) = AbstractFFTs.IRFFTAdjointStyle(p.d)
AbstractFFTs.output_size(p::RFFTLPlan) = Base.setindex(p.sz, nhalf(p.sz[p.region[1]]), p.region[1])
AbstractFFTs.output_size(p::BRFFTLPlan) = Base.setindex(p.sz, p.d, p.region[1])

function Base.show(io::IO, p::RFFTLPlan{T}) where {T}
  print(io, "FFTL r2c plan for $(join(p.sz, "×")) $T, dims $(p.region)")
end
function Base.show(io::IO, p::BRFFTLPlan{T}) where {T}
  print(io, "FFTL c2r plan for $(join(p.sz, "×")) Complex{$T} → length $(p.d), dims $(p.region)")
end

# Runs f(c, i) for the lines i = 0…nlines-1 on the plan's tasks.
function each_line(f, nlines, n, nt)
  ntasks = nt > 1 && nlines * n >= THREAD_MIN_WORK ? nt : 1
  ng = min(nlines, BLOCKS_PER_THREAD * ntasks)
  parallel_items(ng, ntasks) do c, g
    for i in chunk_range(g + 1, ng, nlines)
      f(c, i)
    end
  end
end

function LinearAlgebra.mul!(y::Array{Complex{T},N}, p::RFFTLPlan{T,N}, x::Array{T,N}) where {T,N}
  size(x) == p.sz || throw(DimensionMismatch("r2c plan for size $(p.sz) applied to size $(size(x))"))
  size(y) == AbstractFFTs.output_size(p) || throw(DimensionMismatch("output of size $(size(y)), expected $(AbstractFFTs.output_size(p))"))
  n = p.line.n; h = nhalf(n); S = p.S
  GC.@preserve x y p begin
    px = pointer(x); py = pointer(y)
    each_line(length(p.offs_r), n, p.nthreads) do c, i
      local b = p.bufs
      local pb1 = cptr(b[5c-4]); local pb2 = cptr(b[5c-3]); local pb3 = cptr(b[5c-2])
      if S == 1
        r2c_line!(py + p.offs_c[i+1] * sizeof(Complex{T}), px + p.offs_r[i+1] * sizeof(T), pb1, pb2, pb3, p.line)
      else
        # strided line: gather into a contiguous buffer, transform, scatter
        local lr = Ptr{T}(cptr(b[5c-1])); local lc = Ptr{Complex{T}}(cptr(b[5c]))
        for j in 0:n-1
          unsafe_store!(lr, unsafe_load(px, p.offs_r[i+1] + S * j + 1), j + 1)
        end
        r2c_line!(lc, lr, pb1, pb2, pb3, p.line)
        for k in 0:h-1
          unsafe_store!(py, unsafe_load(lc, k + 1), p.offs_c[i+1] + S * k + 1)
        end
      end
    end
  end
  p.cplan === nothing || mul!(y, p.cplan, y)
  y
end

function LinearAlgebra.mul!(y::Array{T,N}, p::BRFFTLPlan{T,N}, x::Array{Complex{T},N}) where {T,N}
  size(x) == p.sz || throw(DimensionMismatch("c2r plan for size $(p.sz) applied to size $(size(x))"))
  size(y) == AbstractFFTs.output_size(p) || throw(DimensionMismatch("output of size $(size(y)), expected $(AbstractFFTs.output_size(p))"))
  n = p.d; h = nhalf(n); S = p.S
  xc = x
  if p.cplan !== nothing
    xc = reshape(p.copy, p.sz)
    mul!(xc, p.cplan, x)
  end
  GC.@preserve xc y p begin
    px = pointer(xc); py = pointer(y)
    each_line(length(p.offs_r), n, p.nthreads) do c, i
      local b = p.bufs
      local pb1 = cptr(b[5c-4]); local pb2 = cptr(b[5c-3]); local pb3 = cptr(b[5c-2])
      if S == 1
        c2r_line!(py + p.offs_r[i+1] * sizeof(T), px + p.offs_c[i+1] * sizeof(Complex{T}), pb1, pb2, pb3, p.line)
      else
        local lr = Ptr{T}(cptr(b[5c-1])); local lc = Ptr{Complex{T}}(cptr(b[5c]))
        for k in 0:h-1
          unsafe_store!(lc, unsafe_load(px, p.offs_c[i+1] + S * k + 1), k + 1)
        end
        c2r_line!(lr, lc, pb1, pb2, pb3, p.line)
        for j in 0:n-1
          unsafe_store!(py, unsafe_load(lr, j + 1), p.offs_r[i+1] + S * j + 1)
        end
      end
    end
  end
  y
end

Base.:*(p::RFFTLPlan{T,N}, x::AbstractArray{<:Real,N}) where {T,N} =
  mul!(Array{Complex{T},N}(undef, AbstractFFTs.output_size(p)), p, convert(Array{T,N}, x))
Base.:*(p::BRFFTLPlan{T,N}, x::AbstractArray{<:Number,N}) where {T,N} =
  mul!(Array{T,N}(undef, AbstractFFTs.output_size(p)), p, convert(Array{Complex{T},N}, x))

AbstractFFTs.plan_inv(p::RFFTLPlan{T}) where {T} =
  AbstractFFTs.ScaledPlan(BRFFTLPlan{T}(AbstractFFTs.output_size(p), p.sz[p.region[1]], p.region; threads = p.nthreads),
                          AbstractFFTs.normalization(T, p.sz, p.region))
AbstractFFTs.plan_inv(p::BRFFTLPlan{T}) where {T} =
  AbstractFFTs.ScaledPlan(RFFTLPlan{T}(AbstractFFTs.output_size(p), p.region; threads = p.nthreads),
                          AbstractFFTs.normalization(T, AbstractFFTs.output_size(p), p.region))

realtype(::AbstractArray{T}) where {T<:FFTLFloat} = T
realtype(::AbstractArray{Complex{T}}) where {T<:FFTLFloat} = T
realtype(::AbstractArray) = Float64

"""
    plan_rfft(x::AbstractArray{<:Real}, region = 1:ndims(x); threads)
    plan_brfft(X::AbstractArray{<:Complex}, d, region = 1:ndims(X); threads)
    plan_irfft(X::AbstractArray{<:Complex}, d, region = 1:ndims(X); threads)

Real-input forward DFT (half spectrum along the first dimension of `region`), its unnormalised
inverse to real arrays with length `d` along that dimension, and the normalised inverse.
"""
plan_rfft(x::AbstractArray, region = 1:ndims(x); kw...) = RFFTLPlan{realtype(x)}(size(x), region; kw...)
plan_brfft(X::AbstractArray, d::Integer, region = 1:ndims(X); kw...) = BRFFTLPlan{realtype(X)}(size(X), Int(d), region; kw...)
function plan_irfft(X::AbstractArray, d::Integer, region = 1:ndims(X); kw...)
  p = plan_brfft(X, d, region; kw...)
  AbstractFFTs.ScaledPlan(p, AbstractFFTs.normalization(realtype(X), AbstractFFTs.output_size(p), p.region))
end

rfft(x::AbstractArray, region = 1:ndims(x)) = plan_rfft(x, region) * x
brfft(X::AbstractArray, d::Integer, region = 1:ndims(X)) = plan_brfft(X, d, region) * X
irfft(X::AbstractArray, d::Integer, region = 1:ndims(X)) = plan_irfft(X, d, region) * X
