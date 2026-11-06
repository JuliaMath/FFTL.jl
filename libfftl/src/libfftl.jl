# libfftl: FFTL.jl as a trimmed C library. The transforms are FFTW3's C interface
# (include/fftw3.h, implemented in fftw3api.jl) over the package's run-time plan types, whose
# kernels and sizes are data (src/rplan.jl, src/rreal.jl). include/fftl.h declares what FFTW
# has no counterpart for: the message of the last error and the library version.
#
# Rules for code compiled ahead of time that the boundary follows: no closures capturing types
# (dispatch is a straight-line comparison on Ints), error messages are literal strings or the
# package's (set_last_error guards the message type, ArgumentError.msg is AbstractString), and
# every call from the @ccallable interface is monomorphic: per-precision registries with
# concrete element type and one typed function per operation and precision.
#
# The Julia runtime of the library has a fixed number of threads, baked in at build time
# (include/fftl.h, Threading). Plans may ask for more; their tasks then queue on fewer worker
# threads, which gives no speedup.

using FFTL
using FFTL: RPlan, execute!, RRFFTPlan, RR2RPlan, execute_r2c!, execute_c2r!, execute_r2r!

# FFTL computes no BLAS or LAPACK routine: the library starts without loading OpenBLAS, and
# build.sh leaves it out of the bundle.
using LinearAlgebra
@eval LinearAlgebra.OpenBLAS_jll __init__() = nothing
@eval LinearAlgebra __init__() = nothing

# Plans are created and destroyed under this lock (the registries in fftw3api.jl).
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

# --- errors and version -------------------------------------------------------------

Base.@ccallable function fftl_last_error(buf::Ptr{UInt8}, len::Csize_t)::Csize_t
  copy_cstr(codeunits(last_error()), buf, Int(len))
end

Base.@ccallable function fftl_version(buf::Ptr{UInt8}, len::Csize_t)::Csize_t
  copy_cstr(codeunits("0.1.0"), buf, Int(len))
end

# --- the FFTW3 interface (include/fftw3.h) --------------------------------------------

include("fftw3api.jl")
