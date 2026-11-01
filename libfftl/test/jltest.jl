# Boundary tests in Julia, mirroring test/test.c (same matrix) but without a juliac build: the
# @ccallable entry points of src/libfftl.jl are plain methods, so this runs in any Julia with
# the libfftl project (julia --project=. -t 4 test/jltest.jl from libfftl/). In-place and
# out-of-place, both precisions, partial regions, empty regions, threads, error paths.
#
# The reference is a direct DFT in double precision, independent of FFTL.

using FFTL
include(joinpath(@__DIR__, "..", "src", "libfftl.jl"))

ntests = 0
nfailed = 0

macro CHECK(ex)
  Meta.isexpr(ex, :tuple) && length(ex.args) == 2 ||
    error("@CHECK takes the form: @CHECK condition, name")
  esc(quote
    global ntests, nfailed
    local cond, nm = $ex
    ntests += 1
    cond || (nfailed += 1; println("FAIL ", nm))
  end)
end

# Direct DFT of one dimension (0-based d) of a column-major array, double precision.
function dft_pass!(a::Vector{ComplexF64}, dims::Vector{Int}, d::Int, σ::Int)
  S = prod(dims[1:d]; init = 1)
  n = dims[d + 1]
  outer = prod(dims[d + 2:end]; init = 1)
  for o in 0:outer - 1, l in 0:S - 1
    base = o * S * n + l
    x = [a[base + S * j + 1] for j in 0:n - 1]
    for k in 0:n - 1
      a[base + S * k + 1] = sum(x[j + 1] * cispi(2σ * mod(j * k, n) / n) for j in 0:n - 1)
    end
  end
  a
end

function naive(x::Vector{ComplexF64}, dims::Vector{Int}, region::Vector{Int}, σ::Int)
  a = copy(x)
  for d in region
    dft_pass!(a, dims, d, σ)
  end
  a
end

relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps())

# One library transform against the naive reference, out of place and in place. region0 is
# 0-based like the C interface; tol is relative, against the double reference.
function check_region(name::String, dims::Vector{Int}, region0::Vector{Int}, sign::Int,
                      precision::Integer, threads::Int, tol::Real)
  nval = prod(dims)
  dims64 = Int64.(dims)
  reg32 = Int32.(region0)
  x0 = [ComplexF64(randn(), randn()) for _ in 1:nval]
  T = precision == FFTL_PRECISION_DOUBLE ? Float64 : Float32
  xin = Complex{T}.(x0)
  ref = naive(x0, dims, region0, sign)

  p = GC.@preserve dims64 reg32 fftl_plan_dft(Cint(length(dims)), pointer(dims64),
                                              Cint(length(region0)), pointer(reg32),
                                              Cint(precision), Cint(sign), Cint(threads))
  @CHECK p >= 0, name * ": plan creation"
  p < 0 && return
  y = Vector{Complex{T}}(undef, nval)
  z = copy(xin)
  GC.@preserve xin y z begin
    @CHECK fftl_plan_execute(p, Ptr{Cvoid}(pointer(y)), Ptr{Cvoid}(pointer(xin))) == 0,
      name * ": out-of-place execute"
    @CHECK relerr(ComplexF64.(y), ref) < tol, name * ": out of place vs naive"
    @CHECK fftl_plan_execute(p, Ptr{Cvoid}(pointer(z)), Ptr{Cvoid}(pointer(z))) == 0,
      name * ": in-place execute"
    @CHECK relerr(ComplexF64.(z), ref) < tol, name * ": in place vs naive"
  end
  @CHECK fftl_plan_destroy(p) == 0, name * ": destroy"
end

# Forward-backward round trip for sizes beyond the reach of the direct DFT.
function check_roundtrip(name::String, dims::Vector{Int}, region0::Vector{Int}, threads::Int,
                         tol::Real)
  nval = prod(dims)
  dims64 = Int64.(dims)
  reg32 = Int32.(region0)
  scale = prod(dims[d + 1] for d in region0)
  x = [ComplexF64(randn(), randn()) for _ in 1:nval]
  orig = copy(x)
  y = similar(x)
  GC.@preserve dims64 reg32 x y begin
    pf = fftl_plan_dft(Cint(length(dims)), pointer(dims64), Cint(length(region0)),
                       pointer(reg32), Cint(FFTL_PRECISION_DOUBLE), Cint(-1), Cint(threads))
    pb = fftl_plan_dft(Cint(length(dims)), pointer(dims64), Cint(length(region0)),
                       pointer(reg32), Cint(FFTL_PRECISION_DOUBLE), Cint(1), Cint(threads))
    @CHECK pf >= 0 && pb >= 0, name * ": plan creations"
    if pf < 0 || pb < 0
      return
    end
    @CHECK fftl_plan_execute(pf, Ptr{Cvoid}(pointer(y)), Ptr{Cvoid}(pointer(x))) == 0,
      name * ": forward"
    @CHECK fftl_plan_execute(pb, Ptr{Cvoid}(pointer(x)), Ptr{Cvoid}(pointer(y))) == 0,
      name * ": backward"
    @CHECK relerr(x ./ scale, orig) < tol, name * ": round trip"
    @CHECK fftl_plan_destroy(pf) == 0 && fftl_plan_destroy(pb) == 0, name * ": destroys"
  end
end

# --- version and fftl_last_error -----------------------------------------------------

const FFTL_PRECISION_DOUBLE = Int32(1)
const FFTL_PRECISION_SINGLE = Int32(2)

begin
  buf = zeros(UInt8, 64)
  n = fftl_version(pointer(buf), Csize_t(length(buf)))
  @CHECK n == 5 && String(buf[1:6]) == "0.1.0\0", "fftl_version 0.1.0"
  fill!(buf, 0)
  n = fftl_version(pointer(buf), Csize_t(3))
  @CHECK n == 5 && buf[1] == UInt8('0') && buf[2] == UInt8('.') && buf[3] == 0,
    "fftl_version truncates like snprintf"
  fill!(buf, 0)

  dims = Int64[256]
  reg = Int32[0]
  bad = GC.@preserve dims reg fftl_plan_dft(Cint(1), pointer(dims), Cint(1), pointer(reg),
                                            Cint(99), Cint(-1), Cint(1))
  @CHECK bad == -1, "precision 99 rejected"
  @CHECK fftl_last_error(pointer(buf), Csize_t(length(buf))) > 0 && buf[1] != 0,
    "error message present"

  # a successful call does not clear the message
  pver = GC.@preserve dims reg fftl_plan_dft(Cint(1), pointer(dims), Cint(1), pointer(reg),
                                             Cint(FFTL_PRECISION_DOUBLE), Cint(-1), Cint(1))
  @CHECK pver >= 0, "valid plan after an error"
  fftl_last_error(pointer(buf), Csize_t(length(buf)))
  @CHECK buf[1] != 0, "error survives a successful call"
  @CHECK fftl_plan_destroy(pver) == 0, "destroy valid"
end

# --- 1D, both precisions, smooth and prime lengths ------------------------------------

for n in (1, 2, 256, 1000, 9973)
  check_region("1d f64 n=$n forward", [n], [0], -1, FFTL_PRECISION_DOUBLE, 1, 1e-13)
end
check_region("1d f64 n=101 forward (Bluestein)", [101], [0], -1, FFTL_PRECISION_DOUBLE, 1, 1e-13)
check_region("1d f64 n=256 backward", [256], [0], 1, FFTL_PRECISION_DOUBLE, 1, 1e-13)
check_region("1d f64 n=101 backward", [101], [0], 1, FFTL_PRECISION_DOUBLE, 1, 1e-13)
check_region("1d f32 n=256 forward", [256], [0], -1, FFTL_PRECISION_SINGLE, 1, 2e-6)
check_region("1d f32 n=1000 forward", [1000], [0], -1, FFTL_PRECISION_SINGLE, 1, 2e-6)
check_region("1d f32 n=101 forward (Bluestein)", [101], [0], -1, FFTL_PRECISION_SINGLE, 1, 5e-5)
check_region("1d f32 n=9973 forward (Bluestein)", [9973], [0], -1, FFTL_PRECISION_SINGLE, 1, 2e-4)

# --- 2D and 3D, partial regions, Bluestein inside higher ranks ------------------------

check_region("2d f64 (8,12) region {0,1}", [8, 12], [0, 1], -1, FFTL_PRECISION_DOUBLE, 1, 1e-12)
check_region("2d f64 (5,37) region {0,1} (Bluestein)", [5, 37], [0, 1], -1, FFTL_PRECISION_DOUBLE,
             1, 1e-12)
check_region("2d f64 (3,8) region {1}", [3, 8], [1], -1, FFTL_PRECISION_DOUBLE, 1, 1e-12)
check_region("2d f64 (3,8) region {1} backward", [3, 8], [1], 1, FFTL_PRECISION_DOUBLE, 1, 1e-12)
check_region("2d f64 (1024,7) region {1} (lanes)", [1024, 7], [1], 1, FFTL_PRECISION_DOUBLE, 1,
             1e-12)
check_region("2d f32 (8,12) region {0,1}", [8, 12], [0, 1], -1, FFTL_PRECISION_SINGLE, 1, 1e-5)
check_region("3d f64 (4,6,10) region {0,1,2}", [4, 6, 10], [0, 1, 2], -1, FFTL_PRECISION_DOUBLE, 1,
             1e-12)
check_region("3d f64 (4,6,10) region {0,2}", [4, 6, 10], [0, 2], -1, FFTL_PRECISION_DOUBLE, 1,
             1e-12)
check_region("3d f64 (4,6,10) region {1}", [4, 6, 10], [1], 1, FFTL_PRECISION_DOUBLE, 1, 1e-12)
check_region("3d f32 (4,6,10) region {0,1,2}", [4, 6, 10], [0, 1, 2], -1, FFTL_PRECISION_SINGLE,
             1, 1e-5)
check_region("3d f64 (2,3,37) region {0,2} (Bluestein)", [2, 3, 37], [0, 2], -1,
             FFTL_PRECISION_DOUBLE, 1, 1e-12)

# --- empty region: copy out of place, no-op in place ----------------------------------

begin
  dims = Int64[4, 5]
  p = GC.@preserve dims fftl_plan_dft(Cint(2), pointer(dims), Cint(0), Ptr{Cint}(0),
                                       Cint(FFTL_PRECISION_DOUBLE), Cint(-1), Cint(1))
  @CHECK p >= 0, "empty region plan"
  if p >= 0
    x = [ComplexF64(randn(), randn()) for _ in 1:20]
    y = fill(ComplexF64(1e30), 20)
    x0 = copy(x)
    GC.@preserve x y begin
      @CHECK fftl_plan_execute(p, Ptr{Cvoid}(pointer(y)), Ptr{Cvoid}(pointer(x))) == 0,
        "empty region executes"
      @CHECK y == x, "empty region copies out of place"
      @CHECK fftl_plan_execute(p, Ptr{Cvoid}(pointer(x)), Ptr{Cvoid}(pointer(x))) == 0,
        "empty region executes in place"
      @CHECK x == x0, "empty region is a no-op in place"
    end
    @CHECK fftl_plan_destroy(p) == 0, "empty region destroy"
  end
end

# --- threads 1 and 4 must agree; large round trips ------------------------------------

begin
  dims = Int64[33, 1024]
  reg = Int32[0, 1]
  nval = 33 * 1024
  x = [ComplexF64(randn(), randn()) for _ in 1:nval]
  y1 = similar(x)
  y4 = similar(x)
  GC.@preserve dims reg x y1 y4 begin
    p1 = fftl_plan_dft(Cint(2), pointer(dims), Cint(2), pointer(reg), Cint(FFTL_PRECISION_DOUBLE),
                       Cint(-1), Cint(1))
    p4 = fftl_plan_dft(Cint(2), pointer(dims), Cint(2), pointer(reg), Cint(FFTL_PRECISION_DOUBLE),
                       Cint(-1), Cint(4))
    @CHECK p1 >= 0 && p4 >= 0, "threads 1 and 4 plans"
    @CHECK fftl_plan_execute(p1, Ptr{Cvoid}(pointer(y1)), Ptr{Cvoid}(pointer(x))) == 0,
      "2D threads=1 execute"
    @CHECK fftl_plan_execute(p4, Ptr{Cvoid}(pointer(y4)), Ptr{Cvoid}(pointer(x))) == 0,
      "2D threads=4 execute"
    @CHECK relerr(y4, y1) < 1e-13, "2D threads=4 matches threads=1"
    @CHECK fftl_plan_destroy(p1) == 0 && fftl_plan_destroy(p4) == 0, "thread plan destroys"
  end
end

check_roundtrip("1d f64 131072 threads=4 (four-step)", [131072], [0], 4, 1e-13)
check_roundtrip("1d f64 131072 threads=1", [131072], [0], 1, 1e-13)
check_roundtrip("3d f64 (16,16,16) threads=4", [16, 16, 16], [0, 1, 2], 4, 1e-13)
check_roundtrip("1d f64 65536 threads=1", [65536], [0], 1, 1e-13)

# threads=4 against the naive reference in single precision (lane-block threading)
check_region("3d f32 (64,64,8) region {0,1} threads=4", [64, 64, 8], [0, 1], -1,
             FFTL_PRECISION_SINGLE, 4, 1e-5)

# --- error paths ----------------------------------------------------------------------

begin
  for (name, rank, dm, tr, reg, prec, sig, thr) in (
      ("precision 0", 1, [64], 1, [0], 0, -1, 1),
      ("precision 3", 1, [64], 1, [0], 3, -1, 1),
      ("sign 0", 1, [64], 1, [0], 1, 0, 1),
      ("sign 2", 1, [64], 1, [0], 1, 2, 1),
      ("rank 0", 0, Int[], 0, Int[], 1, -1, 1),
      ("trank -1", 1, [64], -1, [0], 1, -1, 1),
      ("trank above rank", 1, [64], 2, [0, 1], 1, -1, 1),
      ("size 0", 1, [0], 1, [0], 1, -1, 1),
      ("size -1", 1, [-1], 1, [0], 1, -1, 1),
      ("second size 0", 2, [8, 0], 1, [0], 1, -1, 1),
      ("region -1", 2, [8, 4], 1, [-1], 1, -1, 1),
      ("region == rank", 2, [8, 4], 1, [2], 1, -1, 1),
      ("region above rank", 2, [8, 4], 1, [3], 1, -1, 1),
      ("region duplicate", 3, [8, 4, 2], 2, [0, 0], 1, -1, 1),
      ("region descending", 3, [8, 4, 2], 2, [1, 0], 1, -1, 1),
      ("region unsorted", 3, [8, 4, 2], 3, [0, 2, 1], 1, -1, 1),
      ("threads 0", 1, [64], 1, [0], 1, -1, 0),
      ("threads -2", 1, [64], 1, [0], 1, -1, -2))
    dims64 = Int64.(dm)
    reg32 = Int32.(reg)
    perr = GC.@preserve dims64 reg32 fftl_plan_dft(Cint(rank), pointer(dims64), Cint(tr),
                                                    pointer(reg32), Cint(prec), Cint(sig),
                                                    Cint(thr))
    @CHECK perr == -1, name * ": rejected"
    @CHECK fftl_last_error(Ptr{UInt8}(0), Csize_t(0)) > 0, name * ": message length"
  end

  # invalid handles
  @CHECK fftl_plan_execute(Clonglong(-1), Ptr{Cvoid}(0), Ptr{Cvoid}(0)) == -1,
    "execute handle -1 rejected"
  @CHECK fftl_plan_execute(Clonglong(0), Ptr{Cvoid}(0), Ptr{Cvoid}(0)) == -1,
    "execute handle 0 rejected (precision 0)"
  @CHECK fftl_plan_execute(Clonglong((Int64(1) << 32) | 99), Ptr{Cvoid}(0), Ptr{Cvoid}(0)) == -1,
    "execute uncreated registry index rejected"
  @CHECK fftl_plan_execute(Clonglong(Int64(3) << 32), Ptr{Cvoid}(0), Ptr{Cvoid}(0)) == -1,
    "execute invalid precision handle rejected"
  @CHECK fftl_plan_destroy(Clonglong(-1)) == -1, "destroy handle -1 rejected"
  @CHECK fftl_plan_destroy(Clonglong(0)) == -1, "destroy handle 0 rejected"

  # a double destroy fails
  dims = Int64[64]
  reg = Int32[0]
  p = GC.@preserve dims reg fftl_plan_dft(Cint(1), pointer(dims), Cint(1), pointer(reg),
                                           Cint(FFTL_PRECISION_DOUBLE), Cint(-1), Cint(1))
  @CHECK p >= 0, "double-destroy target plan"
  @CHECK fftl_plan_destroy(p) == 0, "first destroy"
  @CHECK fftl_plan_destroy(p) == -1, "second destroy rejected"
  # many plans at once, both precisions
  pl = Vector{Clonglong}(undef, 64)
  GC.@preserve dims reg begin
    for i in 0:63
      pl[i + 1] = fftl_plan_dft(Cint(1), pointer(dims), Cint(1), pointer(reg),
                                Cint(i % 2 + 1), Cint(-1), Cint(1))
    end
  end
  @CHECK all(>=(0), pl), "64 plans created"
  @CHECK all(h -> fftl_plan_destroy(h) == 0, pl), "64 plans destroyed"
end

println("jltest: $ntests checks, $nfailed failed")
nfailed == 0 || exit(1)
