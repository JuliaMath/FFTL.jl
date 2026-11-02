# libfftl: FFTL.jl as a trimmed C library. Interface in include/fftl.h: complex DFT, real-input
# (r2c), real-output (c2r) and real-to-real plans by handle for arrays of any rank and any
# lengths, both precisions, single- or multi-threaded execution.
#
# The boundary maps the C calls onto the package's run-time plan types, whose kernels and
# sizes are data (src/rplan.jl, src/rreal.jl):
# - fftl_plan_dft builds FFTL.RPlan{T}(dims, region, σ; threads); lengths may be anything
#   (Bluestein covers lengths with large prime factors), the region selects any subset of
#   dimensions.
# - fftl_plan_dft_r2c / fftl_plan_dft_c2r build FFTL.RRFFTPlan{T} (σ = -1 / +1), real along
#   the first dimension of the region, complex along the others.
# - fftl_plan_r2r builds FFTL.RR2RPlan{T}, one R2RKind per transformed dimension.
# - fftl_plan_execute calls execute!, execute_r2c!, execute_c2r! or execute_r2r!.
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
using FFTL: RPlan, execute!, RRFFTPlan, RR2RPlan, execute_r2c!, execute_c2r!, execute_r2r!

# FFTL computes no BLAS or LAPACK routine: the library starts without loading OpenBLAS, and
# build.sh leaves it out of the bundle.
using LinearAlgebra
@eval LinearAlgebra.OpenBLAS_jll __init__() = nothing
@eval LinearAlgebra __init__() = nothing

# Plans can be created and destroyed from any thread; a plan must not run on two threads at
# once. The handle packs the registry index in the low 32 bits, the FFTL_PRECISION_* value (1
# or 2) in bits 32-39 and the transform type in bits 40 and above (DFT, R2C, C2R, R2R below);
# negative values never occur as handles. One registry per transform type and precision, each
# with a concrete element type.
const DFT, R2C, C2R, R2R = 0, 1, 2, 3
const REGS_F64 = Union{Nothing,RPlan{Float64}}[]
const REGS_F32 = Union{Nothing,RPlan{Float32}}[]
const RF_F64 = Union{Nothing,RRFFTPlan{Float64}}[]     # r2c and c2r
const RF_F32 = Union{Nothing,RRFFTPlan{Float32}}[]
const RR_F64 = Union{Nothing,RR2RPlan{Float64}}[]
const RR_F32 = Union{Nothing,RR2RPlan{Float32}}[]
const REGISTRY_LOCK = ReentrantLock()

# The message of the last failed call, per calling thread (fftl.h: "of this thread").
const LAST_ERRORS = Dict{Int,String}()
const ERROR_LOCK = ReentrantLock()
set_error!(m::String) = (@lock ERROR_LOCK LAST_ERRORS[Threads.threadid()] = m; nothing)
last_error() = @lock ERROR_LOCK get(LAST_ERRORS, Threads.threadid(), "")

function set_last_error(e::Union{ArgumentError,DimensionMismatch,ErrorException})
  m = e.msg
  m isa String ? set_error!(m) : set_error!("operation failed")
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

function register!(reg::Vector{Union{Nothing,P}}, p::P, typ::Int, prec::Int)::Int64 where {P}
  idx = @lock REGISTRY_LOCK begin
    push!(reg, p)
    length(reg)
  end
  Int64(idx - 1) | (Int64(prec) << 32) | (Int64(typ) << 40)
end

lookup(reg::Vector{Union{Nothing,P}}, idx::Int) where {P} =
  @lock REGISTRY_LOCK (0 <= idx <= length(reg) - 1 ? reg[idx + 1] : nothing)

function release!(reg::Vector{Union{Nothing,P}}, idx::Int)::Bool where {P}
  @lock REGISTRY_LOCK begin
    if 0 <= idx <= length(reg) - 1 && reg[idx + 1] !== nothing
      reg[idx + 1] = nothing
      true
    else
      false
    end
  end
end

# --- plan creation ---------------------------------------------------------------

# The C arrays are copied with unsafe_load; every value is validated before or while it is
# read. Region errors the boundary itself rejects: strictly ascending 0-based indices below the
# rank (the plans would silently merge duplicates; out-of-range indices they reject on their
# own, as they do for sizes below one and signs other than ±1).
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
    v[i] = r + 1                     # C is 0-based, the plans take 1-based
    prev = r
  end
  ascending || throw(ArgumentError("region must be 0-based, strictly ascending, and below rank"))
  v
end

const KINDS = Tuple(instances(FFTL.R2RKind))     # FFTL_REDFT00 = 0 … FFTL_HC2R = 10

function read_kinds(p::Ptr{Int32}, trank::Int)
  v = Vector{FFTL.R2RKind}(undef, trank)
  for i in 1:trank
    k = Int(unsafe_load(p, i))
    0 <= k < length(KINDS) || throw(ArgumentError("unknown real-to-real kind"))
    v[i] = KINDS[k + 1]
  end
  v
end

function check_common(prec::Int, rank::Int, trank::Int, threads::Int, mintrank::Int)
  (prec == 1 || prec == 2) || throw(ArgumentError("precision must be 1 or 2"))
  rank >= 1 || throw(ArgumentError("rank must be at least 1"))
  (mintrank <= trank <= rank) || throw(ArgumentError("transform rank must be between 1 (0 for complex DFTs) and rank"))
  threads >= 1 || throw(ArgumentError("threads must be at least 1"))
  nothing
end

function plan_dft(prec::Int, rank::Int, dims::Ptr{Int64}, trank::Int, region::Ptr{Int32},
                  sign::Int, threads::Int)::Int64
  check_common(prec, rank, trank, threads, 0)
  sz = read_dims(dims, rank)
  reg = read_region(region, trank, rank)     # nothing calls a plan on invalid input above
  prec == 1 ? register!(REGS_F64, RPlan{Float64}(sz, reg, sign; threads = threads), DFT, prec) :
              register!(REGS_F32, RPlan{Float32}(sz, reg, sign; threads = threads), DFT, prec)
end

function plan_real(typ::Int, prec::Int, rank::Int, dims::Ptr{Int64}, trank::Int, region::Ptr{Int32},
                   threads::Int)::Int64
  check_common(prec, rank, trank, threads, 1)
  sz = read_dims(dims, rank)
  reg = read_region(region, trank, rank)
  σ = typ == R2C ? -1 : 1
  prec == 1 ? register!(RF_F64, RRFFTPlan{Float64}(sz, reg, σ; threads = threads), typ, prec) :
              register!(RF_F32, RRFFTPlan{Float32}(sz, reg, σ; threads = threads), typ, prec)
end

function plan_r2r(prec::Int, rank::Int, dims::Ptr{Int64}, trank::Int, region::Ptr{Int32},
                  kinds::Ptr{Int32}, threads::Int)::Int64
  check_common(prec, rank, trank, threads, 0)
  sz = read_dims(dims, rank)
  reg = read_region(region, trank, rank)
  ks = read_kinds(kinds, trank)
  prec == 1 ? register!(RR_F64, RR2RPlan{Float64}(sz, ks, reg; threads = threads), R2R, prec) :
              register!(RR_F32, RR2RPlan{Float32}(sz, ks, reg; threads = threads), R2R, prec)
end

# The error message of a failed plan creation from the caught exception `e`, -1. Expanded in the
# catch block itself: a call with the exception as an argument (type Any) cannot be compiled
# ahead of time; each branch below has a concrete type.
macro plan_failed(e, what)
  e = esc(e)
  quote
    if $e isa ArgumentError
      set_last_error($e)
    elseif $e isa DimensionMismatch
      set_last_error($e)
    elseif $e isa ErrorException
      set_last_error($e)
    else
      set_error!($what)
    end
    Int64(-1)
  end
end

Base.@ccallable function fftl_plan_dft(rank::Cint, dims::Ptr{Clonglong}, trank::Cint,
                                       region::Ptr{Cint}, precision::Cint, sign::Cint,
                                       threads::Cint)::Clonglong
  try
    plan_dft(Int(precision), Int(rank), Ptr{Int64}(dims), Int(trank), Ptr{Int32}(region),
             Int(sign), Int(threads))
  catch e
    @plan_failed(e, "plan creation failed")
  end
end

Base.@ccallable function fftl_plan_dft_r2c(rank::Cint, dims::Ptr{Clonglong}, trank::Cint,
                                           region::Ptr{Cint}, precision::Cint,
                                           threads::Cint)::Clonglong
  try
    plan_real(R2C, Int(precision), Int(rank), Ptr{Int64}(dims), Int(trank), Ptr{Int32}(region),
              Int(threads))
  catch e
    @plan_failed(e, "plan creation failed")
  end
end

Base.@ccallable function fftl_plan_dft_c2r(rank::Cint, dims::Ptr{Clonglong}, trank::Cint,
                                           region::Ptr{Cint}, precision::Cint,
                                           threads::Cint)::Clonglong
  try
    plan_real(C2R, Int(precision), Int(rank), Ptr{Int64}(dims), Int(trank), Ptr{Int32}(region),
              Int(threads))
  catch e
    @plan_failed(e, "plan creation failed")
  end
end

Base.@ccallable function fftl_plan_r2r(rank::Cint, dims::Ptr{Clonglong}, trank::Cint,
                                       region::Ptr{Cint}, kinds::Ptr{Cint}, precision::Cint,
                                       threads::Cint)::Clonglong
  try
    plan_r2r(Int(precision), Int(rank), Ptr{Int64}(dims), Int(trank), Ptr{Int32}(region),
             Ptr{Int32}(kinds), Int(threads))
  catch e
    @plan_failed(e, "plan creation failed")
  end
end

# --- plan execution and destruction ------------------------------------------------

invalid_handle() = (set_error!("invalid plan handle"); Cint(-1))

function exec_dft(reg::Vector{Union{Nothing,RPlan{T}}}, idx::Int, out::Ptr{Cvoid}, in_::Ptr{Cvoid})::Cint where {T}
  e = lookup(reg, idx)
  e === nothing && return invalid_handle()
  GC.@preserve e execute!(e, Ptr{Complex{T}}(out), Ptr{Complex{T}}(in_))
  Cint(0)
end

function exec_real(reg::Vector{Union{Nothing,RRFFTPlan{T}}}, idx::Int, typ::Int, out::Ptr{Cvoid}, in_::Ptr{Cvoid})::Cint where {T}
  e = lookup(reg, idx)
  (e === nothing || (typ == R2C) != (e.σ == -1)) && return invalid_handle()
  GC.@preserve e begin
    typ == R2C ? execute_r2c!(e, Ptr{Complex{T}}(out), Ptr{T}(in_)) : execute_c2r!(e, Ptr{T}(out), Ptr{Complex{T}}(in_))
  end
  Cint(0)
end

function exec_r2r(reg::Vector{Union{Nothing,RR2RPlan{T}}}, idx::Int, out::Ptr{Cvoid}, in_::Ptr{Cvoid})::Cint where {T}
  e = lookup(reg, idx)
  e === nothing && return invalid_handle()
  GC.@preserve e execute_r2r!(e, Ptr{T}(out), Ptr{T}(in_))
  Cint(0)
end

# (type, precision, index) of a handle; type -1 for anything that is not a handle
function split_handle(h::Int64)
  h < 0 && return (-1, 0, 0)
  typ = Int(h >> 40); prec = Int((h >> 32) & 0xff)
  (0 <= typ <= R2R && (prec == 1 || prec == 2)) || return (-1, 0, 0)
  (typ, prec, Int(h & 0xffffffff))
end

Base.@ccallable function fftl_plan_execute(h::Clonglong, out::Ptr{Cvoid}, in_::Ptr{Cvoid})::Cint
  try
    execute_handle(Int64(h), out, in_)
  catch
    set_error!("execution failed")
    Cint(-1)
  end
end

function execute_handle(h::Int64, out::Ptr{Cvoid}, in_::Ptr{Cvoid})::Cint
  typ, prec, idx = split_handle(h)
  if typ == DFT
    prec == 1 ? exec_dft(REGS_F64, idx, out, in_) : exec_dft(REGS_F32, idx, out, in_)
  elseif typ == R2C || typ == C2R
    prec == 1 ? exec_real(RF_F64, idx, typ, out, in_) : exec_real(RF_F32, idx, typ, out, in_)
  elseif typ == R2R
    prec == 1 ? exec_r2r(RR_F64, idx, out, in_) : exec_r2r(RR_F32, idx, out, in_)
  else
    invalid_handle()
  end
end

Base.@ccallable function fftl_plan_destroy(h::Clonglong)::Cint
  typ, prec, idx = split_handle(Int64(h))
  ok = if typ == DFT
    prec == 1 ? release!(REGS_F64, idx) : release!(REGS_F32, idx)
  elseif typ == R2C || typ == C2R
    prec == 1 ? release!(RF_F64, idx) : release!(RF_F32, idx)
  elseif typ == R2R
    prec == 1 ? release!(RR_F64, idx) : release!(RR_F32, idx)
  else
    false
  end
  ok ? Cint(0) : invalid_handle()
end

# --- errors and version -------------------------------------------------------------

Base.@ccallable function fftl_last_error(buf::Ptr{UInt8}, len::Csize_t)::Csize_t
  copy_cstr(codeunits(last_error()), buf, Int(len))
end

Base.@ccallable function fftl_version(buf::Ptr{UInt8}, len::Csize_t)::Csize_t
  copy_cstr(codeunits("0.1.0"), buf, Int(len))
end
