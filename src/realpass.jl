# Real-to-complex and complex-to-real passes along one dimension d of a batch of lines. The real
# grid has n values along d, the half spectrum nhalf(n) = n÷2+1. Each line is gathered into a
# contiguous buffer, transformed by a RealLine kernel and scattered, so the other dimensions are
# plain batch lanes. Input and output may be any strided arrays (views of padded grids).

"""
    RealPassPlan{T,N,L}

Plan for the unnormalised r2c (`σ = -1`, the half spectrum as in `FFTL.rfft`) or c2r (`σ = +1`)
transform along dimension `d`. The plan fixes the size of the input and the output; the strides
are taken from the arrays given to `mul!`.
"""
struct RealPassPlan{T<:FFTLFloat,N,L<:RealLine{T}}
  σ::Int
  d::Int
  n::Int                              # real length along d
  sz_in::NTuple{N,Int}
  sz_out::NTuple{N,Int}
  line::L
  bufs::Vector{Vector{Complex{T}}}    # five per task
  nthreads::Int
end

# Plan for dimension d of an input of size sz_in; σ = -1 (r2c) reads n reals, σ = +1 (c2r) reads
# nhalf(n) complex values along d.
function realpass_plan(::Type{T}, σ::Int, sz_in::NTuple{N,Int}, d::Int, n::Int, nt::Int) where {T<:FFTLFloat,N}
  nin = σ == -1 ? n : nhalf(n)
  sz_in[d] == nin || throw(DimensionMismatch("input has length $(sz_in[d]) along dimension $d, expected $nin"))
  sz_out = Base.setindex(sz_in, σ == -1 ? nhalf(n) : n, d)
  line = RealLine{T}(n, σ)
  RealPassPlan{T,N,typeof(line)}(σ, d, n, sz_in, sz_out, line, thread_buffers(T, n, 5nt), nt)
end

"""
    plan_r2c_pass(v::StridedArray{<:Real}, d; threads = 1)

Plan for the batched r2c transform along dimension `d` of arrays with the size of `v`: the output
has the half spectrum `nhalf(size(v, d))` along `d`. Apply with `mul!(w, p, v)`, where `v` and `w`
may be strided views (dimension 1 of each line may be strided too). Same packing as `FFTL.rfft`
for each line.
"""
function plan_r2c_pass(v::StridedArray{T,N}, d::Integer; threads::Integer = 1) where {T<:FFTLFloat,N}
  1 <= d <= N || throw(ArgumentError("dimension $d is not within 1:$N"))
  all(>=(1), size(v)) || throw(ArgumentError("array size must be positive"))
  realpass_plan(T, -1, size(v), Int(d), size(v, d), max(1, Int(threads)))
end

"""
    plan_c2r_pass(v::StridedArray{<:Complex}, d; n = 2(size(v, d) - 1), threads = 1)

Plan for the batched unnormalised c2r transform along dimension `d` from the half spectrum `v`
(`size(v, d) == nhalf(n)`) to a real array with length `n` along `d`. Apply with `mul!(w, p, v)`
(`w` of size `size(v)` with `n` instead of `size(v, d)` along `d`, may be a strided view). `n`
must be given for odd lengths. The imaginary parts of the zero and Nyquist values are ignored, as
in `FFTL.irfft`.
"""
function plan_c2r_pass(v::StridedArray{Complex{T},N}, d::Integer; n::Integer = 2(size(v, d) - 1),
                       threads::Integer = 1) where {T<:FFTLFloat,N}
  1 <= d <= N || throw(ArgumentError("dimension $d is not within 1:$N"))
  all(>=(1), size(v)) || throw(ArgumentError("array size must be positive"))
  n >= 1 || throw(ArgumentError("real length must be positive"))
  nhalf(n) == size(v, d) || throw(DimensionMismatch("half spectrum of length $(size(v, d)) does not belong to length $n"))
  realpass_plan(T, 1, size(v), Int(d), Int(n), max(1, Int(threads)))
end

function Base.show(io::IO, p::RealPassPlan{T}) where {T}
  print(io, "FFTL ", p.σ == -1 ? "r2c" : "c2r", " pass plan, dim $(p.d) of $(join(p.sz_in, "×")) ",
        "$T -> $(join(p.sz_out, "×"))")
end

function check_realpass(p::RealPassPlan, v, w)
  size(v) == p.sz_in && size(w) == p.sz_out ||
    throw(DimensionMismatch("real pass plan for input size $(p.sz_in) and output size $(p.sz_out); " *
                            "applied to input size $(size(v)) and output size $(size(w))"))
end

function LinearAlgebra.mul!(w::StridedArray{Complex{T},N}, p::RealPassPlan{T,N}, v::StridedArray{T,N}) where {T,N}
  p.σ == -1 || throw(ArgumentError("c2r pass plan applied to a real input"))
  check_realpass(p, v, w)
  GC.@preserve v w p real_pass!(Ptr{Complex{T}}(pointer(w)), strides(w), Ptr{T}(pointer(v)), strides(v), p)
  w
end

function LinearAlgebra.mul!(w::StridedArray{T,N}, p::RealPassPlan{T,N}, v::StridedArray{Complex{T},N}) where {T,N}
  p.σ == 1 || throw(ArgumentError("r2c pass plan applied to a complex input"))
  check_realpass(p, v, w)
  GC.@preserve v w p real_pass!(Ptr{T}(pointer(w)), strides(w), Ptr{Complex{T}}(pointer(v)), strides(v), p)
  w
end

# Line offsets of an array of size sz and strides st, for all dimensions except d.
function pass_offsets(sz::NTuple{N,Int}, st::NTuple{N,Int}, d::Int) where {N}
  outer = Tuple(i for i in 1:N if i != d)
  line_offsets(sz, st, outer)
end

# r2c (σ = -1): py receives the half spectrum (strides sy), px the real input (strides sx).
# c2r (σ = +1): py receives the real output, px the half spectrum; the offsets are in elements.
function real_pass!(py, sy::NTuple{N,Int}, px, sx::NTuple{N,Int}, p::RealPassPlan{T,N}) where {T,N}
  n = p.n; d = p.d
  r2c = p.σ == -1
  offs_x = pass_offsets(p.sz_in, sx, d)
  offs_y = pass_offsets(p.sz_out, sy, d)
  Sx = sx[d]; Sy = sy[d]
  h = nhalf(n)
  each_line(length(offs_x), n, line_tasks(p.line, p.nthreads)) do c, i
    local b = p.bufs
    local lr = Ptr{T}(cptr(b[5c-1])); local lc = Ptr{Complex{T}}(cptr(b[5c]))
    local xo = offs_x[i+1]; local yo = offs_y[i+1]
    if r2c
      for j in 0:n-1
        unsafe_store!(lr, unsafe_load(px, xo + Sx * j + 1), j + 1)
      end
      r2c_line!(lc, lr, cptr(b[5c-4]), cptr(b[5c-3]), cptr(b[5c-2]), p.line)
      for k in 0:h-1
        unsafe_store!(py, unsafe_load(lc, k + 1), yo + Sy * k + 1)
      end
    else
      for k in 0:h-1
        unsafe_store!(lc, unsafe_load(px, xo + Sx * k + 1), k + 1)
      end
      c2r_line!(lr, lc, cptr(b[5c-4]), cptr(b[5c-3]), cptr(b[5c-2]), p.line)
      for j in 0:n-1
        unsafe_store!(py, unsafe_load(lr, j + 1), yo + Sy * j + 1)
      end
    end
  end
  nothing
end
