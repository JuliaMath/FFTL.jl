# CPU overview over all transform kinds: c2c (forward), r2c, c2r (unnormalised), r2r with
# DCT-II (REDFT10) as the main kind plus REDFT00, REDFT01, REDFT11, RODFT10 and DHT, out of place.
# Libraries: FFTL, FFTW MEASURE (wisdom forgotten before every plan), ducc0 (c2c, r2c, c2r, DHT
# in 1D as its genuine Hartley transform), Apple vDSP at one thread (c2c 1D, DCT-II F32 1D of
# lengths f·2^k; vDSP's DCT-II is half of FFTW's REDFT10 and is compared scaled by 2).
#
#   julia --project=bench -t 8 bench/overview/run_kinds.jl NTHREADS out.csv
#
# One process per thread count; appends to out.csv and writes out.meta (commit, date, load).
# OVERVIEW_RESUME continues an interrupted run (see main).
# Columns: lib,kind,T,dims,shape,n,threads,seconds,plan_seconds,relerr,gflops. Time: minimum
# over samples of (time of r calls)/r with r so that a sample takes ≥ 50 µs, at least 5 samples
# and 0.2 s per case (as bench/baseline/run.jl). relerr: max-norm against FFTW in Float64.
# gflops: 5 n log2 n / t for c2c, 2.5 n log2 n / t for the real kinds (nominal).

using LinearAlgebra, Random, Printf
import FFTW, FFTL
include(joinpath(@__DIR__, "..", "ducc-ffi", "ducc_ffi.jl"))   # ducc0 through libducc_ffi, not Ducc0.jl

const C2C_SHAPES = vcat(
  tuple.(vcat([2^k for k in 4:22],
              [48, 80, 96, 120, 240, 384, 480, 1000, 1536, 2000, 3000, 6144, 10000, 12288, 30720, 46080,
               100000, 120000, 360000, 786432, 1000000],
              [56, 448, 1001, 3584, 9801, 50421], [17, 101, 1009, 10007, 100003, 1000003])),
  [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096),
   (100, 100), (120, 120), (480, 480), (1000, 1000), (1296, 1296), (1920, 1080)],
  [(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256), (48, 48, 48),
   (100, 100, 100), (120, 120, 120), (160, 160, 160)])
const REAL_SHAPES = vcat(
  tuple.(vcat([2^k for k in 4:22], [48, 100, 120, 1000, 1001, 1009, 3000, 10000, 100000, 1000000])),
  [(64, 64), (256, 256), (1024, 1024), (2048, 2048), (1000, 1000), (1920, 1080)],
  [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256), (100, 100, 100)])
const DCT_SHAPES = vcat(tuple.(vcat([2^k for k in 4:2:20], [1000, 1001, 100000])),
                        [(256, 256), (1024, 1024), (2048, 2048), (1000, 1000)], [(64, 64, 64), (128, 128, 128)])
const R2R_OTHER = (:REDFT00, :REDFT01, :REDFT11, :RODFT10, :DHT)
const OTHER_SHAPES = [(1024,), (1000,), (256, 256), (64, 64, 64)]

function repeat_run(run, r)
  for _ in 1:r
    run()
  end
end
function tmin(run)
  run()
  r = 1
  while (t = @elapsed repeat_run(run, r)) < 50e-6
    r *= 2
  end
  best = t / r; n = 1; t0 = time()
  while n < 5 || time() - t0 < 0.2
    best = min(best, (@elapsed repeat_run(run, r)) / r)
    n += 1
  end
  best
end

# One function per library and kind: a variable captured by the returned closure must have one
# type, otherwise every call pays a dynamic dispatch.
# OVERVIEW_KEEP_WISDOM=1: do not forget the wisdom between cases. On a host where FFTW.forget_wisdom
# crashes with threaded plans (Zen 2, Ubuntu 20.04: segfault in libfftw3f in 20–70 % of loops, with the
# native build and with FFTW_jll), a plan then reuses sub-plans measured for earlier cases, which can
# only favour FFTW.
const KEEP_WISDOM = get(ENV, "OVERVIEW_KEEP_WISDOM", "") == "1"
forget_wisdom() = KEEP_WISDOM || FFTW.forget_wisdom()
fftw_prep(nt) = (FFTW.set_num_threads(nt); forget_wisdom())
function c2c_fftl(x, y, nt); t = @elapsed p = FFTL.plan_fft(x; threads = nt); (() -> mul!(y, p, x)), t; end
function c2c_fftw(x, y, nt); fftw_prep(nt); t = @elapsed p = FFTW.plan_fft(copy(x); flags = FFTW.MEASURE); (() -> mul!(y, p, x)), t; end
c2c_ducc(x, y, nt) = ducc_c2c_setup(x, y, nt)
function r2c_fftl(x, y, nt); t = @elapsed p = FFTL.plan_rfft(x; threads = nt); (() -> mul!(y, p, x)), t; end
function r2c_fftw(x, y, nt); fftw_prep(nt); t = @elapsed p = FFTW.plan_rfft(copy(x); flags = FFTW.MEASURE); (() -> mul!(y, p, x)), t; end
r2c_ducc(x, y, nt) = ducc_r2c_setup(x, y, nt)
function c2r_fftl(x, y, nt); t = @elapsed p = FFTL.plan_brfft(x, size(y, 1); threads = nt); (() -> mul!(y, p, x)), t; end
# FFTW's c2r overwrites its input: each call gets a fresh copy (the copy is part of the time)
function c2r_fftw(x, y, nt)
  fftw_prep(nt); xc = copy(x)
  t = @elapsed p = FFTW.plan_brfft(copy(x), size(y, 1); flags = FFTW.MEASURE)
  (() -> (copyto!(xc, x); mul!(y, p, xc))), t
end
c2r_ducc(x, y, nt) = ducc_c2r_setup(x, y, nt)
function r2r_fftl(x, y, nt, k); t = @elapsed p = FFTL.plan_r2r(x, getfield(FFTL, k); threads = nt); (() -> mul!(y, p, x)), t; end
function r2r_fftw(x, y, nt, k); fftw_prep(nt); t = @elapsed p = FFTW.plan_r2r(copy(x), getfield(FFTW, k); flags = FFTW.MEASURE); (() -> mul!(y, p, x)), t; end
dht_ducc(x, y, nt) = ducc_r2r_setup(x, y, nt, 2)

const ACCELERATE = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
# vDSP takes lengths f·2^k with f in {1, 3, 5, 15}; other lengths make the library crash in Execute.
vdsp_length(n) = (while iseven(n); n ÷= 2; end; n in (1, 3, 5, 15))
function c2c_vdsp(x::Vector{ComplexF64}, y)
  vdsp_length(length(x)) || return nothing
  t = @elapsed s = @ccall ACCELERATE.vDSP_DFT_Interleaved_CreateSetupD(C_NULL::Ptr{Cvoid}, length(x)::Csize_t, Cint(1)::Cint, 0::Cint)::Ptr{Cvoid}
  s == C_NULL ? nothing : ((() -> @ccall ACCELERATE.vDSP_DFT_Interleaved_ExecuteD(s::Ptr{Cvoid}, x::Ptr{ComplexF64}, y::Ptr{ComplexF64})::Cvoid), t)
end
function c2c_vdsp(x::Vector{ComplexF32}, y)
  vdsp_length(length(x)) || return nothing
  t = @elapsed s = @ccall ACCELERATE.vDSP_DFT_Interleaved_CreateSetup(C_NULL::Ptr{Cvoid}, length(x)::Csize_t, Cint(1)::Cint, 0::Cint)::Ptr{Cvoid}
  s == C_NULL ? nothing : ((() -> @ccall ACCELERATE.vDSP_DFT_Interleaved_Execute(s::Ptr{Cvoid}, x::Ptr{ComplexF32}, y::Ptr{ComplexF32})::Cvoid), t)
end
c2c_vdsp(x, y) = nothing
const VDSP_DCT_II = Cint(2)
function dct_vdsp(x::Vector{Float32}, y)
  vdsp_length(length(x)) || return nothing
  t = @elapsed s = @ccall ACCELERATE.vDSP_DCT_CreateSetup(C_NULL::Ptr{Cvoid}, length(x)::Csize_t, VDSP_DCT_II::Cint)::Ptr{Cvoid}
  s == C_NULL ? nothing : ((() -> @ccall ACCELERATE.vDSP_DCT_Execute(s::Ptr{Cvoid}, x::Ptr{Float32}, y::Ptr{Float32})::Cvoid), t)
end
dct_vdsp(x, y) = nothing

function setup(lib, kind, x, y, nt)
  if kind == "c2c"
    lib == :fftl && return c2c_fftl(x, y, nt)
    lib == :fftw_measure && return c2c_fftw(x, y, nt)
    lib == :ducc && return c2c_ducc(x, y, nt)
    lib == :vdsp && return nt == 1 && Sys.isapple() ? c2c_vdsp(x, y) : nothing
  elseif kind == "r2c"
    lib == :fftl && return r2c_fftl(x, y, nt)
    lib == :fftw_measure && return r2c_fftw(x, y, nt)
    lib == :ducc && return r2c_ducc(x, y, nt)
  elseif kind == "c2r"
    lib == :fftl && return c2r_fftl(x, y, nt)
    lib == :fftw_measure && return c2r_fftw(x, y, nt)
    lib == :ducc && return c2r_ducc(x, y, nt)
  else
    k = Symbol(kind[5:end])
    lib == :fftl && return r2r_fftl(x, y, nt, k)
    lib == :fftw_measure && return r2r_fftw(x, y, nt, k)
    # ducc0's genuine Hartley transform equals FFTW's separable DHT only in 1D
    lib == :ducc && k == :DHT && ndims(x) == 1 && return dht_ducc(x, y, nt)
    lib == :vdsp && k == :REDFT10 && nt == 1 && Sys.isapple() && return dct_vdsp(x, y)
  end
  nothing
end

function reference(kind, x, n1)
  x64 = kind == "c2c" ? ComplexF64.(x) : kind == "c2r" ? ComplexF64.(x) : Float64.(x)
  kind == "c2c" && return FFTW.fft(x64)
  kind == "r2c" && return FFTW.rfft(x64)
  kind == "c2r" && return FFTW.brfft(x64, n1)
  FFTW.r2r(x64, getfield(FFTW, Symbol(kind[5:end])))
end

function main(nt, out)
  # OVERVIEW_LIBS=fftw_measure,ducc: only these libraries (default all). FFTW runs in its own C threads, so a
  # process with one Julia thread can time it at nt threads (FFTW.jl with several Julia threads
  # crashes in libfftw3f with threaded Float32 3D plans on the Zen 2 host).
  libnames = Symbol.(split(get(ENV, "OVERVIEW_LIBS", "fftl,fftw_measure,ducc,vdsp"), ","))
  (:fftl in libnames || :ducc in libnames) && Threads.nthreads() < nt && error("start Julia with at least $nt threads")
  meta = replace(out, r"\.csv$" => ".meta")
  d = dirname(dirname(pathof(FFTL)))
  open(meta, "a") do io
    println(io, "threads: $nt; FFTL ", readchomp(`git -C $d rev-parse --short HEAD`), "; julia ", VERSION, "; ", gethostname())
    println(io, "start: ", Libc.strftime("%Y-%m-%dT%H:%M:%S", time()), "; load: ", readchomp(`uptime`))
  end
  isfile(out) || open(io -> println(io, "lib,kind,T,dims,shape,n,threads,seconds,plan_seconds,relerr,gflops"), out, "w")
  cases = vcat([("c2c", s) for s in C2C_SHAPES], [("r2c", s) for s in REAL_SHAPES], [("c2r", s) for s in REAL_SHAPES],
               [("r2r_REDFT10", s) for s in DCT_SHAPES], [("r2r_$k", s) for k in R2R_OTHER for s in OTHER_SHAPES])
  # OVERVIEW_QUICK=1: a few cases of every kind (smoke test)
  haskey(ENV, "OVERVIEW_QUICK") && (cases = [c for (i, c) in enumerate(cases) if i % 15 == 1])
  libs = Tuple(l for l in (:fftl, :fftw_measure, :ducc, :vdsp) if l in libnames)
  # OVERVIEW_RESUME=kind,shape,T (e.g. c2c,120x120x120,Float32): skip the cases before this one
  # (continue an interrupted run; remove that case's partial rows from out.csv first)
  resume = get(ENV, "OVERVIEW_RESUME", "")
  # OVERVIEW_SKIP=lib:kind,shape,T;... : cases a library crashes on (recorded in the .meta file)
  skip = Set(split(get(ENV, "OVERVIEW_SKIP", ""), ";"; keepempty = false))
  isempty(skip) || open(io -> println(io, "skipped: ", join(skip, "; ")), meta, "a")
  KEEP_WISDOM && open(io -> println(io, "FFTW wisdom kept between cases (OVERVIEW_KEEP_WISDOM)"), meta, "a")
  open(io -> println(io, "libraries: ", join(libnames, ",")), meta, "a")
  for (kind, shape) in cases, T in (Float64, Float32)
    if !isempty(resume)
      join((kind, join(shape, "x"), T), ",") == resume || continue
      resume = ""
    end
    n = prod(shape)
    xr = randn(MersenneTwister(1), T, shape...)
    x = kind == "c2c" ? randn(MersenneTwister(1), Complex{T}, shape...) :
        kind == "c2r" ? Complex{T}.(FFTW.rfft(Float64.(xr))) : xr
    FFTW.set_num_threads(nt)
    ref = reference(kind, x, shape[1])
    ysz = kind == "r2c" ? (shape[1] ÷ 2 + 1, shape[2:end]...) : shape
    yT = kind in ("c2c", "r2c") ? Complex{T} : T
    flop = (kind == "c2c" ? 5 : 2.5) * n * log2(max(n, 2))
    for lib in libs
      "$lib:" * join((kind, join(shape, "x"), T), ",") in skip && continue
      y = Array{yT}(undef, ysz)
      s = try setup(lib, kind, x, y, nt) catch e; @warn "$lib $kind $shape $T" e; nothing end
      s === nothing && continue
      run, tplan = s
      run()
      scale = lib == :vdsp && kind == "r2r_REDFT10" ? 2 : 1
      err = maximum(abs.(scale .* ComplexF64.(y) .- ref)) / maximum(abs.(ref))
      t = tmin(run)
      line = join((lib, kind, T, length(shape), join(shape, "x"), n, nt, @sprintf("%.4e", t),
                   @sprintf("%.3e", tplan), @sprintf("%.2e", err), @sprintf("%.2f", flop / t / 1e9)), ",")
      println(line); flush(stdout)
      open(io -> println(io, line), out, "a")
      lib == :fftw_measure && forget_wisdom()
      GC.gc(false)
    end
  end
  open(io -> println(io, "end: ", Libc.strftime("%Y-%m-%dT%H:%M:%S", time()), "; load: ", readchomp(`uptime`)), meta, "a")
end

# Included by harness/measure.jl for its library setups and shape lists: run main only as a script.
abspath(PROGRAM_FILE) == (@__FILE__) && main(parse(Int, ARGS[1]), ARGS[2])
