# libfftl: FFTL.jl as a trimmed C library. Interface in include/fftl.h: complex DFT plans by
# handle for arrays of any rank and any lengths, both precisions, both signs, single- or
# multi-threaded execution.
#
# The boundary maps the C calls onto the package's run-time plan type (src/rplan.jl):
# - fftl_plan_dft builds FFTL.RPlan{T}(dims, region, σ; threads); lengths may be anything
#   (Bluestein inside RPlan covers lengths with large prime factors), the region selects any
#   subset of dimensions. The plan (kernel tables, work buffers, thread count) is run-time data
#   there; the boundary keeps none of the phase-0 "keep in sync" copies or variant tables.
# - fftl_plan_execute calls FFTL.execute!(p, py, px), py == px in place; RPlan's execution is
#   plain code over the @generated stage codelets, the path designed for juliac --trim.
#
# Trim rules established in phase 0 (notes/libfftl.md, findings 2-5) that the boundary itself
# still follows: no closures capturing types (dispatch is a straight-line comparison on Ints),
# error messages are literal strings or the package's (set_last_error guards the message type,
# ArgumentError.msg is AbstractString), and every call from the @ccallable interface is
# monomorphic: per-precision registries with concrete element type and one typed function per
# operation and precision.
#
# The Julia runtime of a trimmed library starts with a fixed number of threads (baked at build
# time or set through JULIA_NUM_THREADS before load; see include/fftl.h). Plans may ask for
# more; execution then queues their tasks on fewer worker threads, which gives no speedup.

using FFTL
using FFTL: RPlan, execute!

# FFTL computes no BLAS or LAPACK routine: the library starts without loading OpenBLAS, and
# build.sh leaves it out of the bundle.
using LinearAlgebra
@eval LinearAlgebra.OpenBLAS_jll __init__() = nothing
@eval LinearAlgebra __init__() = nothing

# Plans can be created and destroyed from any thread; a plan must not run on two threads at
# once. The handle packs the registry index in the low 32 bits and the FFTL_PRECISION_* value
# (1 or 2) in bits 32 and above; negative values never occur as handles.
const REGS_F64 = Union{Nothing,RPlan{Float64}}[]
const REGS_F32 = Union{Nothing,RPlan{Float32}}[]
const REGISTRY_LOCK = ReentrantLock()

const LAST_ERROR = Ref("")
function set_last_error(e::Union{ArgumentError,DimensionMismatch,ErrorException})
  m = e.msg
  m isa String ? (LAST_ERROR[] = m) : (LAST_ERROR[] = "operation failed")
  nothing
end

# Writes the bytes of m as a zero-terminated string into buf (at most len bytes including the
# terminator) and returns the full byte length, like snprintf. buf == NULL returns the length.
# codeunits of a String is a CodeUnits{UInt8}, not a Vector, from both callers below.
function copy_cstr(m::AbstractVector{UInt8}, buf::Ptr{UInt8}, len::Int)::Csize_t
  if buf != C_NULL && len > 0
    nerr = min(length(m), len - 1)
    for i in 1:nerr
      unsafe_store!(buf, m[i], i)
    end
    unsafe_store!(buf, 0x00, nerr + 1)
  end
  Csize_t(length(m))
end

# --- plan creation ---------------------------------------------------------------

# The C arrays are copied with unsafe_load; every value is validated before or while it is
# read. Region errors the boundary itself rejects: strictly ascending 0-based indices below the
# rank (RPlan would silently merge duplicates; out-of-range indices it rejects on its own, as
# it does for sizes below one and signs other than ±1).
function read_dims(p::Ptr{Int64}, rank::Int)
  v = Vector{Int}(undef, rank)
  for i in 1:rank
    v[i] = Int(unsafe_load(p, i))
  end
  v
end

function read_region(p::Ptr{Int32}, trank::Int, rank::Int)
  v = Vector{Int}(undef, trank)
  ascending = true
  prev = -1
  for i in 1:trank
    r = Int(unsafe_load(p, i))
    ascending = ascending && (prev < r < rank)
    v[i] = r + 1                     # C is 0-based, RPlan takes 1-based
    prev = r
  end
  ascending || throw(ArgumentError("region must be 0-based, strictly ascending, and below rank"))
  v
end

function plan_dft(prec::Int, rank::Int, dims::Ptr{Int64}, trank::Int, region::Ptr{Int32},
                 sign::Int, threads::Int)::Int64
  (prec == 1 || prec == 2) || throw(ArgumentError("precision must be 1 or 2"))
  rank >= 1 || throw(ArgumentError("rank must be at least 1"))
  (0 <= trank <= rank) || throw(ArgumentError("transform rank must be between 0 and rank"))
  threads >= 1 || throw(ArgumentError("threads must be at least 1"))
  sz = read_dims(dims, rank)
  reg = read_region(region, trank, rank)     # nothing calls RPlan on invalid input above
  if prec == 1
    p = RPlan{Float64}(sz, reg, sign; threads = threads)
    idx = @lock REGISTRY_LOCK begin
      push!(REGS_F64, p)
      length(REGS_F64)
    end
    Int64(idx - 1) | (Int64(prec) << 32)
  else
    p = RPlan{Float32}(sz, reg, sign; threads = threads)
    idx = @lock REGISTRY_LOCK begin
      push!(REGS_F32, p)
      length(REGS_F32)
    end
    Int64(idx - 1) | (Int64(prec) << 32)
  end
end

Base.@ccallable function fftl_plan_dft(rank::Cint, dims::Ptr{Clonglong}, trank::Cint,
                                       region::Ptr{Cint}, precision::Cint, sign::Cint,
                                       threads::Cint)::Clonglong
  try
    plan_dft(Int(precision), Int(rank), Ptr{Int64}(dims), Int(trank), Ptr{Int32}(region),
             Int(sign), Int(threads))
  catch e
    if e isa ArgumentError || e isa DimensionMismatch || e isa ErrorException
      set_last_error(e)
    else
      LAST_ERROR[] = "plan creation failed"
    end
    Int64(-1)
  end
end

# --- plan execution and destruction ------------------------------------------------

function exec_f64(idx::Int, out::Ptr{Cvoid}, in_::Ptr{Cvoid})::Cint
  e = @lock REGISTRY_LOCK (0 <= idx <= length(REGS_F64) - 1 ? REGS_F64[idx + 1] : nothing)
  e === nothing && (LAST_ERROR[] = "invalid plan handle"; return Cint(-1))
  GC.@preserve e begin
    execute!(e, Ptr{Complex{Float64}}(out), Ptr{Complex{Float64}}(in_))
  end
  Cint(0)
end

function exec_f32(idx::Int, out::Ptr{Cvoid}, in_::Ptr{Cvoid})::Cint
  e = @lock REGISTRY_LOCK (0 <= idx <= length(REGS_F32) - 1 ? REGS_F32[idx + 1] : nothing)
  e === nothing && (LAST_ERROR[] = "invalid plan handle"; return Cint(-1))
  GC.@preserve e begin
    execute!(e, Ptr{Complex{Float32}}(out), Ptr{Complex{Float32}}(in_))
  end
  Cint(0)
end

function destroy_f64(idx::Int)::Cint
  ok = @lock REGISTRY_LOCK begin
    if 0 <= idx <= length(REGS_F64) - 1 && REGS_F64[idx + 1] !== nothing
      REGS_F64[idx + 1] = nothing
      true
    else
      false
    end
  end
  ok || (LAST_ERROR[] = "invalid plan handle")
  Cint(ok ? 0 : -1)
end

function destroy_f32(idx::Int)::Cint
  ok = @lock REGISTRY_LOCK begin
    if 0 <= idx <= length(REGS_F32) - 1 && REGS_F32[idx + 1] !== nothing
      REGS_F32[idx + 1] = nothing
      true
    else
      false
    end
  end
  ok || (LAST_ERROR[] = "invalid plan handle")
  Cint(ok ? 0 : -1)
end

function split_handle(h::Int64)
  (Int(h >> 32), Int(h & 0xffffffff))
end

Base.@ccallable function fftl_plan_execute(h::Clonglong, out::Ptr{Cvoid}, in_::Ptr{Cvoid})::Cint
  prec, idx = split_handle(Int64(h))
  if h < 0 || (prec != 1 && prec != 2)
    LAST_ERROR[] = "invalid plan handle"
    Cint(-1)
  elseif prec == 1
    exec_f64(idx, out, in_)
  else
    exec_f32(idx, out, in_)
  end
end

Base.@ccallable function fftl_plan_destroy(h::Clonglong)::Cint
  prec, idx = split_handle(Int64(h))
  if h < 0 || (prec != 1 && prec != 2)
    LAST_ERROR[] = "invalid plan handle"
    Cint(-1)
  elseif prec == 1
    destroy_f64(idx)
  else
    destroy_f32(idx)
  end
end

# --- errors and version -------------------------------------------------------------

Base.@ccallable function fftl_last_error(buf::Ptr{UInt8}, len::Csize_t)::Csize_t
  copy_cstr(codeunits(LAST_ERROR[]), buf, Int(len))
end

Base.@ccallable function fftl_version(buf::Ptr{UInt8}, len::Csize_t)::Csize_t
  copy_cstr(codeunits("0.1.0"), buf, Int(len))
end
