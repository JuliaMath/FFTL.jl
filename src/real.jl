# Real-input (r2c) and real-output (c2r) transforms along the first dimension of a region; the
# other dimensions of the region are complex passes on the half spectrum.
#
# Even length n = 2M: a real line read as M complex values z_j = x_(2j) + i·x_(2j+1) (the memory
# layout as is) gets a complex DFT of length M, Z, and
#   r2c:  X_k = E_k + w^k·O_k,  E_k = (Z_k + conj Z_(M-k))/2,  O_k = (Z_k - conj Z_(M-k))/(2i),
#   c2r:  Z_k = (X_k + conj X_(M-k)) + i·w^k·(X_k - conj X_(M-k)),  then z = DFT_M(Z),
# with w = exp(σ·2πi/n) and Z_M = Z_0. Since w^(M-k) = -conj(w^k), the pair k, M-k shares its
# products: with s = Z_k + conj Z_(M-k), d = Z_k - conj Z_(M-k), t = u_k·d and u_k = -i·w^k/2
# (c2r: u_k = i·w^k, s and d of X) the outputs are
#   r2c:  X_k = s/2 + t,  X_(M-k) = conj(s/2 - t),    c2r:  Z_k = s + t,  Z_(M-k) = conj(s - t).
# Odd lengths use a complex DFT of length n.

"""
    RealLine{T}

Kernel for r2c (σ = -1) or c2r (σ = +1) transforms of contiguous real lines of length `n`
(half spectrum of `n ÷ 2 + 1` values).
"""
struct RealLine{T,K<:Kernel{T}}
  n::Int
  σ::Int
  inner::K                     # length n ÷ 2 (even n) or n (odd n)
  w::Vector{Complex{T}}        # u_k, k = 0…n÷2 (even n), see above
end

# Run-time kernels (RChain, or Bluestein over RChain) for the run-time plans (rreal.jl).
const RKern{T} = Union{RChain{T},RBluestein{T}}
rkernel(::Type{T}, n::Int, σ::Int) where {T} =
  (smooth(n) ? RChain{T}(n, σ, radices(n, 1; esize = sizeof(Complex{T}))) : Bluestein{T}(n, σ, RChain))::RKern{T}
ckernel(::Type{T}, n::Int, σ::Int, ::Type{Chain}) where {T} = kernel(T, n, σ)
ckernel(::Type{T}, n::Int, σ::Int, ::Type{RChain}) where {T} = rkernel(T, n, σ)

# C: Chain (kernel type in the line's type) or RChain (RKern{T}, for run-time plans)
function RealLine{T}(n::Int, σ::Int, ::Type{C} = Chain) where {T,C}
  inner = ckernel(T, iseven(n) ? n ÷ 2 : n, σ, C)
  u = σ == -1 ? ComplexF64(0, -0.5) : ComplexF64(0, 1)
  w = Complex{T}[]
  if iseven(n)
    W = unit_roots(σ, n)
    w = Vector{Complex{T}}(undef, n ÷ 2 + 1)
    for k in 0:n÷2
      w[k+1] = Complex{T}(u * W[k+1])
    end
  end
  RealLine{T,C === RChain ? RKern{T} : typeof(inner)}(n, σ, inner, w)
end

nhalf(n) = n ÷ 2 + 1

# r2c of one line: x (n reals) → y (n÷2+1 complex); b1, b2, b3: complex buffers of n values.
function r2c_line!(py::Ptr{Complex{T}}, px::Ptr{T}, b1::Ptr{T}, b2::Ptr{T}, b3::Ptr{T}, K::RealLine{T}) where {T}
  n = K.n
  if iseven(n)
    M = n ÷ 2
    run!(Ptr{T}(py), px, b1, b2, K.inner, 1)
    r2c_ends!(py, M)
    r2c_pairs!(py, K, M, 1, M ÷ 2)
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
    c2r_ends!(cb1, px, M)
    c2r_pairs!(cb1, px, K, M, 1, M ÷ 2)
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

# Even lengths: X_0 and X_M from Z_0 (r2c), Z_0 from the real parts of X_0, X_M (c2r), and the
# pairs k, M-k for k = k1…k2 (1 ≤ k ≤ M÷2); pairs are independent, so ranges of them can run on
# different tasks.
function r2c_ends!(py::Ptr{Complex{T}}, M::Int) where {T}
  z0 = unsafe_load(py, 1)
  unsafe_store!(py, Complex(real(z0) + imag(z0), zero(T)), 1)
  unsafe_store!(py, Complex(real(z0) - imag(z0), zero(T)), M + 1)
  nothing
end

function r2c_pairs!(py::Ptr{Complex{T}}, K::RealLine{T}, M::Int, k1::Int, k2::Int) where {T}
  V = single_vec(T); CB = sizeof(Complex{T})
  half = V((T(0.5), T(0.5)))
  GC.@preserve K @inbounds for k in k1:k2
    zk = vload(V, Ptr{T}(py + k * CB)); czm = cconj(vload(V, Ptr{T}(py + (M - k) * CB)))
    t = ccmul(vload(V, Ptr{T}(pointer(K.w) + k * CB)), zk - czm)
    h = (zk + czm) * half
    vstore(h + t, Ptr{T}(py + k * CB))
    vstore(cconj(h - t), Ptr{T}(py + (M - k) * CB))
  end
  nothing
end

function c2r_ends!(cb1::Ptr{Complex{T}}, px::Ptr{Complex{T}}, M::Int) where {T}
  x0 = real(unsafe_load(px, 1)); xM = real(unsafe_load(px, M + 1))
  unsafe_store!(cb1, Complex(x0 + xM, x0 - xM), 1)              # k = 0: real parts only
  nothing
end

function c2r_pairs!(cb1::Ptr{Complex{T}}, px::Ptr{Complex{T}}, K::RealLine{T}, M::Int, k1::Int, k2::Int) where {T}
  V = single_vec(T); CB = sizeof(Complex{T})
  GC.@preserve K @inbounds for k in k1:k2
    xk = vload(V, Ptr{T}(px + k * CB)); xm = cconj(vload(V, Ptr{T}(px + (M - k) * CB)))
    s = xk + xm
    t = ccmul(vload(V, Ptr{T}(pointer(K.w) + k * CB)), xk - xm)
    vstore(s + t, Ptr{T}(cb1 + k * CB))
    vstore(cconj(s - t), Ptr{T}(cb1 + (M - k) * CB))
  end
  nothing
end

# One long even line on several threads: the complex DFT of length n/2 by the four-step
# algorithm, the pairs in chunks on the tasks.
function r2c_line_fourstep!(py::Ptr{Complex{T}}, px::Ptr{T}, F::FourStep{T}, K::RealLine{T}, nt::Int) where {T}
  M = K.n ÷ 2
  run_fourstep!(Ptr{T}(py), px, F, nt)
  r2c_ends!(py, M)
  np = M ÷ 2; nc = min(np, BLOCKS_PER_THREAD * nt)
  parallel_items(nc, nt) do c, g
    local r = chunk_range(g + 1, nc, np)
    isempty(r) || r2c_pairs!(py, K, M, first(r) + 1, last(r) + 1)
  end
  nothing
end

function c2r_line_fourstep!(py::Ptr{T}, px::Ptr{Complex{T}}, cb1::Ptr{Complex{T}}, F::FourStep{T}, K::RealLine{T}, nt::Int) where {T}
  M = K.n ÷ 2
  c2r_ends!(cb1, px, M)
  np = M ÷ 2; nc = min(np, BLOCKS_PER_THREAD * nt)
  parallel_items(nc, nt) do c, g
    local r = chunk_range(g + 1, nc, np)
    isempty(r) || c2r_pairs!(cb1, px, K, M, first(r) + 1, last(r) + 1)
  end
  run_fourstep!(py, Ptr{T}(cb1), F, nt)
  nothing
end

# the four-step plan of a single long line, or nothing (also for the run-time plans)
line_fourstep(p) = nothing

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
mutable struct RFFTLPlan{T<:FFTLFloat,N,K,C,D,F} <: AbstractFFTs.Plan{T}
  sz::NTuple{N,Int}
  region::NTuple{D,Int}
  line::K
  lines::Vector{K}                  # one line kernel per task (copies of `line` if it has buffers of its own)
  four::F                           # one long even line on several threads: four-step, or nothing
  cplan::C                          # complex passes on the half spectrum, or nothing
  S::Int
  offs_r::Vector{Int}
  offs_c::Vector{Int}
  strided_r::Vector{Int}
  strided_c::Vector{Int}
  nthreads::Int
  bufs::Vector{Vector{Complex{T}}}  # three per thread, plus line buffers for strided lines
  pinv::AbstractFFTs.Plan
  RFFTLPlan{T,N,K,C,D,F}(sz, region, line, lines, four, cplan, S, offs_r, offs_c, nt, bufs) where {T,N,K,C,D,F} =
    new{T,N,K,C,D,F}(sz, region, line, lines, four, cplan, S, offs_r, offs_c, similar(offs_r), similar(offs_c), nt, bufs)
end

"""
    BRFFTLPlan{T,N}

Plan for the unnormalised c2r DFT (backward) from half spectra of size `sz` to real arrays with
length `d` along the first dimension of `region`.
"""
mutable struct BRFFTLPlan{T<:FFTLFloat,N,K,C,D,F} <: AbstractFFTs.Plan{Complex{T}}
  sz::NTuple{N,Int}
  d::Int
  region::NTuple{D,Int}
  line::K
  lines::Vector{K}                  # one line kernel per task, as in RFFTLPlan
  four::F
  cplan::C
  S::Int
  offs_r::Vector{Int}
  offs_c::Vector{Int}
  strided_r::Vector{Int}
  strided_c::Vector{Int}
  nthreads::Int
  bufs::Vector{Vector{Complex{T}}}
  copy::Array{Complex{T},N}        # the input, transformed along the complex dimensions
  pinv::AbstractFFTs.Plan
  BRFFTLPlan{T,N,K,C,D,F}(sz, d, region, line, lines, four, cplan, S, offs_r, offs_c, nt, bufs, copy) where {T,N,K,C,D,F} =
    new{T,N,K,C,D,F}(sz, d, region, line, lines, four, cplan, S, offs_r, offs_c, similar(offs_r), similar(offs_c), nt, bufs, copy)
end

sortregion(region, N) = Tuple(normregion(region, N))

# One real line of length n on several tasks: four-step for the half-length DFT. The extra
# parallel pass for the pairs pays off from 256 KB per line on (M1, 8 threads: Float64 32768 and
# Float32 65536 are 0.6-0.9 of one task, Float64 16384 and Float32 16384-32768 up to 3x slower).
use_real_fourstep(::Type{T}, n, nt) where {T} =
  iseven(n) && n * sizeof(T) >= TUNING[].real_fourstep_bytes && use_fourstep(n ÷ 2, nt)

function RFFTLPlan{T}(sz::NTuple{N,Int}, region; threads::Integer = Threads.nthreads()) where {T<:FFTLFloat,N}
  reg = sortregion(region, N)
  isempty(reg) && throw(ArgumentError("rfft needs at least one dimension"))
  all(>=(1), sz) || throw(ArgumentError("array size must be positive"))
  d1 = reg[1]; n = sz[d1]
  line = RealLine{T}(n, -1)
  csz = Base.setindex(sz, nhalf(n), d1)
  cplan = length(reg) > 1 ? FFTLPlan{T}(csz, reg[2:end], -1; threads) : nothing
  S, offs_r, offs_c = line_layout(sz, d1)
  nt = max(1, Int(threads))
  bufs = thread_buffers(T, n, 5nt)
  four = length(offs_r) == 1 && use_real_fourstep(T, n, nt) ? FourStep{T}(n ÷ 2, -1, nt) : nothing
  lines = task_reallines(line, n, -1, nt)
  RFFTLPlan{T,N,typeof(line),typeof(cplan),length(reg),typeof(four)}(sz, reg, line, lines, four, cplan, S, offs_r, offs_c, nt, bufs)
end

function BRFFTLPlan{T}(sz::NTuple{N,Int}, d::Int, region; threads::Integer = Threads.nthreads()) where {T<:FFTLFloat,N}
  reg = sortregion(region, N)
  isempty(reg) && throw(ArgumentError("brfft needs at least one dimension"))
  (d >= 1 && all(>=(1), sz)) || throw(ArgumentError("array size must be positive"))
  d1 = reg[1]
  nhalf(d) == sz[d1] || throw(DimensionMismatch("half spectrum of length $(sz[d1]) does not fit length $d"))
  line = RealLine{T}(d, 1)
  cplan = length(reg) > 1 ? FFTLPlan{T}(sz, reg[2:end], 1; threads) : nothing
  rsz = Base.setindex(sz, d, d1)
  S, offs_r, offs_c = line_layout(rsz, d1)
  nt = max(1, Int(threads))
  bufs = thread_buffers(T, d, 5nt)
  four = length(offs_r) == 1 && use_real_fourstep(T, d, nt) ? FourStep{T}(d ÷ 2, 1, nt) : nothing
  lines = task_reallines(line, d, 1, nt)
  BRFFTLPlan{T,N,typeof(line),typeof(cplan),length(reg),typeof(four)}(sz, d, reg, line, lines, four, cplan, S, offs_r, offs_c, nt, bufs,
                                              Array{Complex{T},N}(undef, sz))
end

line_fourstep(p::Union{RFFTLPlan,BRFFTLPlan}) = p.four

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

# Whether a line kernel uses work buffers of its own (a Bluestein kernel holds them in the kernel,
# not in the task's buffer set): such lines run on one task.
own_buffers(k) = k isa Union{Bluestein,Composite}
own_buffers(K::RealLine) = K.inner isa Union{Bluestein,Composite}
line_tasks(K, nt) = own_buffers(K) ? 1 : nt

# One line kernel per task. A kernel with buffers of its own (Bluestein, composite) cannot run on two
# tasks at once, so each task gets a copy; the other kernels are shared.
task_reallines(line::RealLine{T}, n::Int, σ::Int, nt::Int) where {T} =
  own_buffers(line) && nt > 1 ? [line; [RealLine{T}(n, σ) for _ in 2:nt]] : [line]
rline_tasks(L, nt) = length(L) > 1 ? length(L) : line_tasks(L[1], nt)
rline_task(L, c) = L[length(L) > 1 ? c : 1]

# Runs f(c, i) for the lines i = 0…nlines-1 on the plan's tasks.
function each_line(f::F, nlines, n, nt) where {F}
  ntasks = nt > 1 && nlines * n >= TUNING[].thread_min_work ? nt : 1
  if ntasks == 1
    for i in 0:nlines-1
      f(1, i)
    end
    return nothing
  end
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
  GC.@preserve x y p begin
    p.S == 1 ? r2c_contiguous!(pointer(y), pointer(x), p) : r2c_strided!(pointer(y), pointer(x), p)
  end
  p.cplan === nothing || mul!(y, p.cplan, y)
  y
end

# Keep the dense offsets intact: a plan may alternate between dense and strided operands.
function LinearAlgebra.mul!(y::StridedArray{Complex{T},N}, p::RFFTLPlan{T,N}, x::StridedArray{T,N}) where {T,N}
  size(x) == p.sz || throw(DimensionMismatch("r2c plan for size $(p.sz) applied to size $(size(x))"))
  size(y) == AbstractFFTs.output_size(p) || throw(DimensionMismatch("output of size $(size(y)), expected $(AbstractFFTs.output_size(p))"))
  if dense_layout(x) && dense_layout(y)
    GC.@preserve x y p begin
      p.S == 1 ? r2c_contiguous!(pointer(y), pointer(x), p) : r2c_strided!(pointer(y), pointer(x), p)
    end
  else
    d = p.region[1]
    pass_offsets!(p.strided_r, p.sz, strides(x), d)
    pass_offsets!(p.strided_c, size(y), strides(y), d)
    GC.@preserve x y p r2c_lines!(pointer(y), pointer(x), p.strided_c, p.strided_r,
                                 stride(y, d), stride(x, d), p.line, p.bufs, p.nthreads)
  end
  p.cplan === nothing || mul!(y, p.cplan, y)
  y
end

# Separate functions for contiguous and strided lines: a shared closure body is 30–70 ns slower
# on short lines.
function r2c_contiguous!(py::Ptr{Complex{T}}, px::Ptr{T}, p) where {T}
  b = p.bufs
  if length(p.offs_r) == 1
    F = line_fourstep(p)
    F === nothing || return r2c_line_fourstep!(py, px, F, p.line, p.nthreads)
    return r2c_line!(py, px, cptr(b[1]), cptr(b[2]), cptr(b[3]), p.line)
  end
  each_line(length(p.offs_r), p.line.n, rline_tasks(p.lines, p.nthreads)) do c, i
    local b = p.bufs
    r2c_line!(py + p.offs_c[i+1] * sizeof(Complex{T}), px + p.offs_r[i+1] * sizeof(T),
              cptr(b[5c-4]), cptr(b[5c-3]), cptr(b[5c-2]), rline_task(p.lines, c))
  end
end

# one kernel for all tasks
# a tuple, not a vector: no allocation per call
r2c_lines!(py, px, offs_y, offs_x, Sy, Sx, line::RealLine, bufs, nt::Int) =
  r2c_lines!(py, px, offs_y, offs_x, Sy, Sx, (line,), bufs, nt)
c2r_lines!(py, px, offs_y, offs_x, Sy, Sx, line::RealLine, bufs, nt::Int) =
  c2r_lines!(py, px, offs_y, offs_x, Sy, Sx, (line,), bufs, nt)

# strided line: gather into a contiguous buffer, transform, scatter
function r2c_lines!(py::Ptr{Complex{T}}, px::Ptr{T}, offs_y::Vector{Int}, offs_x::Vector{Int}, Sy::Int, Sx::Int,
                    lines::Union{Tuple,Vector}, bufs::Vector{Vector{Complex{T}}}, nt::Int) where {T}
  n = lines[1].n; h = nhalf(n)
  each_line(length(offs_x), n, rline_tasks(lines, nt)) do c, i
    local lr = Ptr{T}(cptr(bufs[5c-1])); local lc = Ptr{Complex{T}}(cptr(bufs[5c]))
    for j in 0:n-1
      unsafe_store!(lr, unsafe_load(px, offs_x[i+1] + Sx * j + 1), j + 1)
    end
    r2c_line!(lc, lr, cptr(bufs[5c-4]), cptr(bufs[5c-3]), cptr(bufs[5c-2]), rline_task(lines, c))
    for k in 0:h-1
      unsafe_store!(py, unsafe_load(lc, k + 1), offs_y[i+1] + Sy * k + 1)
    end
  end
  nothing
end

function LinearAlgebra.mul!(y::Array{T,N}, p::BRFFTLPlan{T,N}, x::Array{Complex{T},N}) where {T,N}
  size(x) == p.sz || throw(DimensionMismatch("c2r plan for size $(p.sz) applied to size $(size(x))"))
  size(y) == AbstractFFTs.output_size(p) || throw(DimensionMismatch("output of size $(size(y)), expected $(AbstractFFTs.output_size(p))"))
  xc = x
  if p.cplan !== nothing
    xc = p.copy
    mul!(xc, p.cplan, x)
  end
  GC.@preserve xc y p begin
    p.S == 1 ? c2r_contiguous!(pointer(y), pointer(xc), p) : c2r_strided!(pointer(y), pointer(xc), p)
  end
  y
end

function LinearAlgebra.mul!(y::StridedArray{T,N}, p::BRFFTLPlan{T,N}, x::StridedArray{Complex{T},N}) where {T,N}
  size(x) == p.sz || throw(DimensionMismatch("c2r plan for size $(p.sz) applied to size $(size(x))"))
  size(y) == AbstractFFTs.output_size(p) || throw(DimensionMismatch("output of size $(size(y)), expected $(AbstractFFTs.output_size(p))"))
  xc = x
  if p.cplan !== nothing
    xc = p.copy
    mul!(xc, p.cplan, x)
  end
  if dense_layout(xc) && dense_layout(y)
    GC.@preserve xc y p begin
      p.S == 1 ? c2r_contiguous!(pointer(y), pointer(xc), p) : c2r_strided!(pointer(y), pointer(xc), p)
    end
  else
    d = p.region[1]
    pass_offsets!(p.strided_r, size(y), strides(y), d)
    pass_offsets!(p.strided_c, p.sz, strides(xc), d)
    GC.@preserve xc y p c2r_lines!(pointer(y), pointer(xc), p.strided_r, p.strided_c,
                                  stride(y, d), stride(xc, d), p.line, p.bufs, p.nthreads)
  end
  y
end

# Preserve output identity through normalisation, as for complex strided plans.
function LinearAlgebra.mul!(y::StridedArray{T,N},
                            p::AbstractFFTs.ScaledPlan{<:Any,<:BRFFTLPlan{T,N}},
                            x::StridedArray{Complex{T},N}) where {T,N}
  mul!(y, p.p, x)
  LinearAlgebra.lmul!(p.scale, y)
  y
end
function LinearAlgebra.mul!(y::StridedArray{Complex{T},N},
                            p::AbstractFFTs.ScaledPlan{<:Any,<:RFFTLPlan{T,N}},
                            x::StridedArray{T,N}) where {T,N}
  mul!(y, p.p, x)
  LinearAlgebra.lmul!(p.scale, y)
  y
end

function c2r_contiguous!(py::Ptr{T}, px::Ptr{Complex{T}}, p) where {T}
  b = p.bufs
  if length(p.offs_r) == 1
    F = line_fourstep(p)
    F === nothing || return c2r_line_fourstep!(py, px, Ptr{Complex{T}}(cptr(b[1])), F, p.line, p.nthreads)
    return c2r_line!(py, px, cptr(b[1]), cptr(b[2]), cptr(b[3]), p.line)
  end
  each_line(length(p.offs_r), p.line.n, rline_tasks(p.lines, p.nthreads)) do c, i
    local b = p.bufs
    c2r_line!(py + p.offs_r[i+1] * sizeof(T), px + p.offs_c[i+1] * sizeof(Complex{T}),
              cptr(b[5c-4]), cptr(b[5c-3]), cptr(b[5c-2]), rline_task(p.lines, c))
  end
end

function c2r_lines!(py::Ptr{T}, px::Ptr{Complex{T}}, offs_y::Vector{Int}, offs_x::Vector{Int}, Sy::Int, Sx::Int,
                    lines::Union{Tuple,Vector}, bufs::Vector{Vector{Complex{T}}}, nt::Int) where {T}
  n = lines[1].n; h = nhalf(n)
  each_line(length(offs_x), n, rline_tasks(lines, nt)) do c, i
    local lr = Ptr{T}(cptr(bufs[5c-1])); local lc = Ptr{Complex{T}}(cptr(bufs[5c]))
    for k in 0:h-1
      unsafe_store!(lc, unsafe_load(px, offs_x[i+1] + Sx * k + 1), k + 1)
    end
    c2r_line!(lr, lc, cptr(bufs[5c-4]), cptr(bufs[5c-3]), cptr(bufs[5c-2]), rline_task(lines, c))
    for j in 0:n-1
      unsafe_store!(py, unsafe_load(lr, j + 1), offs_y[i+1] + Sy * j + 1)
    end
  end
  nothing
end

# the strided batches of RFFTLPlan and RRFFTPlan: input and output with the same stride S
r2c_strided!(py::Ptr{Complex{T}}, px::Ptr{T}, p) where {T} =
  r2c_lines!(py, px, p.offs_c, p.offs_r, p.S, p.S, p.lines, p.bufs, p.nthreads)
c2r_strided!(py::Ptr{T}, px::Ptr{Complex{T}}, p) where {T} =
  c2r_lines!(py, px, p.offs_r, p.offs_c, p.S, p.S, p.lines, p.bufs, p.nthreads)

Base.:*(p::RFFTLPlan{T,N}, x::StridedArray{T,N}) where {T,N} =
  mul!(Array{Complex{T},N}(undef, AbstractFFTs.output_size(p)), p, x)
Base.:*(p::BRFFTLPlan{T,N}, x::StridedArray{Complex{T},N}) where {T,N} =
  mul!(Array{T,N}(undef, AbstractFFTs.output_size(p)), p, x)

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
    plan_rfft(x::AbstractArray{<:Real}, region = 1:ndims(x); threads, num_threads, flags, timelimit)
    plan_brfft(X::AbstractArray{<:Complex}, d, region = 1:ndims(X); …)
    plan_irfft(X::AbstractArray{<:Complex}, d, region = 1:ndims(X); …)

Real-input forward DFT (half spectrum along the first dimension of `region`), its unnormalised
inverse to real arrays with length `d` along that dimension, and the normalised inverse.
Keywords as for [`plan_fft`](@ref).
"""
function plan_rfft(x::AbstractArray, region = 1:ndims(x); threads = nothing, num_threads = nothing,
                   flags = nothing, timelimit = nothing, kw...)
  generic_float(elreal(x)) && return GRPlan{elreal(x)}(size(x), size(x, minimum(region)), region, -1)
  RFFTLPlan{realtype(x)}(size(x), region; threads = planthreads(threads, num_threads), kw...)
end
function plan_brfft(X::AbstractArray, d::Integer, region = 1:ndims(X); threads = nothing, num_threads = nothing,
                    flags = nothing, timelimit = nothing, kw...)
  generic_float(elreal(X)) && return GRPlan{elreal(X)}(size(X), Int(d), region, 1)
  BRFFTLPlan{realtype(X)}(size(X), Int(d), region; threads = planthreads(threads, num_threads), kw...)
end
function plan_irfft(X::AbstractArray, d::Integer, region = 1:ndims(X); kw...)
  p = plan_brfft(X, d, region; kw...)
  T = generic_float(elreal(X)) ? elreal(X) : realtype(X)
  AbstractFFTs.ScaledPlan(p, AbstractFFTs.normalization(T, AbstractFFTs.output_size(p), p.region))
end

"""
    rfft(x, region = 1:ndims(x))

Forward DFT of real `x`, returning the half spectrum along the first dimension of `region`
(`size(x, d) ÷ 2 + 1` values there).
"""
rfft(x::AbstractArray, region = 1:ndims(x)) = plan_rfft(x, region) * x

"""
    brfft(X, d, region = 1:ndims(X))

Unnormalised inverse of the half spectrum `X` to a real array of length `d` along the first
dimension of `region`.
"""
brfft(X::AbstractArray, d::Integer, region = 1:ndims(X)) = plan_brfft(X, d, region) * X

"""
    irfft(X, d, region = 1:ndims(X))

Normalised inverse of the half spectrum `X` to a real array of length `d` along the first
dimension of `region`: `irfft(rfft(x), length(x)) ≈ x`.
"""
irfft(X::AbstractArray, d::Integer, region = 1:ndims(X)) = plan_irfft(X, d, region) * X
