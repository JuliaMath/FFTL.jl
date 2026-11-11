# GPU part of the harness, used by measure.jl when the machine profile has device = "metal" or
# "cuda". Same raw format and the same cases, rounds and A/A pairing as the CPU measurement.
#
# Libraries:
#   fftl      FFTL through its KernelAbstractions extension (all kinds of the catalog)
#   mpsgraph  Apple's FFT through Metal.jl (AbstractFFTs plans; c2c, r2c, c2r; Metal has Float32 only)
#   cufft     native cuFFT (bench/cufft, one process per call; c2c, r2c, c2r)
#   vkfft     VkFFT (bench/vkfft, one process per call; the kinds in VKKIND)
#
# Timing. Julia-side libraries: a round is the median over seven samples of `reps` transforms back to
# back with one synchronisation, per transform. The separate-process libraries report their own
# timings: one call of their harness is one round, and that harness takes its best of seven samples.
# plan_seconds is the first plan, with kernel generation and run-time compilation.
#
# Not here: vkfft_batch (Metal; all transforms of a sample in one command buffer), which is not a
# per-call time.

using FFTL, FFTW, LinearAlgebra, Random, Statistics, Printf
import AbstractFFTs
include(joinpath(@__DIR__, "..", "vkfft", "vkfft.jl"))
include(joinpath(@__DIR__, "..", "cufft", "cufft.jl"))

const VKKIND = Dict("r2r_REDFT00" => :dct1, "r2r_REDFT10" => :dct2, "r2r_REDFT01" => :dct3,
                    "r2r_REDFT11" => :dct4, "r2r_RODFT10" => :dst2, "c2c" => :c2c, "r2c" => :r2c, "c2r" => :c2r)
const VENDOR_KINDS = ("c2c", "r2c", "c2r")

if DEVICE == "metal"
  @eval using KernelAbstractions, Metal
  const AT = Metal.MtlArray; const SYNC = Metal.synchronize
  const FLOATS = (Float32,)
else
  @eval using KernelAbstractions, CUDA
  const AT = CUDA.CuArray; const SYNC = CUDA.synchronize
  const FLOATS = (Float32, Float64)
end

# Whether a library has the kind.
function available(lib, kind)
  kind == "r2r_DHT" && return false          # no GPU DHT in this harness (not in the GPU panels either)
  lib == "fftl" && return true
  lib in ("mpsgraph", "cufft") && return kind in VENDOR_KINDS
  lib == "vkfft" && return haskey(VKKIND, kind)
  false
end

# Plan of a Julia-side library: (planning seconds, plan).
function julia_plan(lib, kind, xd, n1)
  t0 = time_ns()
  p = if lib == "fftl"
    kind == "c2c" ? FFTL.plan_fft(xd) :
    kind == "r2c" ? FFTL.plan_rfft(xd) :
    kind == "c2r" ? FFTL.plan_brfft(xd, n1) :
    FFTL.plan_r2r(xd, getfield(FFTL, Symbol(kind[5:end])))
  else  # mpsgraph
    kind == "c2c" ? AbstractFFTs.plan_fft(xd) :
    kind == "r2c" ? AbstractFFTs.plan_rfft(xd) :
    AbstractFFTs.plan_brfft(xd, n1)
  end
  (time_ns() - t0) / 1e9, p
end

# Median over seven samples of `reps` back-to-back transforms, per transform.
function per_transform(f, reps)
  f(); SYNC()
  ts = Float64[]
  for _ in 1:7
    t = time_ns()
    for _ in 1:reps
      f()
    end
    SYNC()
    push!(ts, (time_ns() - t) / 1e9 / reps)
  end
  median(ts)
end

function gpu_reference(kind, x, n1)
  kind == "c2c" && return FFTW.fft(ComplexF64.(x))
  kind == "r2c" && return FFTW.rfft(Float64.(x))
  kind == "c2r" && return FFTW.brfft(ComplexF64.(x), n1)
  FFTW.r2r(Float64.(x), getfield(FFTW, Symbol(kind[5:end])))
end

# Measures the cases of the catalog on the GPU. `cases` are (kind, shape, family); rows go to `out`
# in the CPU raw format with threads = 0. Progress lines go to stdout for monitor.jl.
function gpu_measure(out, cases, libs, rounds, machine_label, tier_name)
  total = length(cases) * length(FLOATS)
  done = 0
  for (kind, shape, family) in cases, T in FLOATS
    println("PROGRESS $done $total"); flush(stdout)
    done += 1
    n = prod(shape)
    reps = n >= 2^22 ? 20 : 100
    xr = randn(MersenneTwister(1), T, shape...)
    x = kind == "c2c" ? randn(MersenneTwister(1), Complex{T}, shape...) :
        kind == "c2r" ? Complex{T}.(FFTW.rfft(Float64.(xr))) : xr
    n1 = shape[1]
    ref = gpu_reference(kind, x, n1)
    refmax = maximum(abs.(ref))
    xd = AT(x)
    OT = kind in ("c2c", "r2c") ? Complex{T} : T
    yshape = kind == "r2c" ? (n1 ÷ 2 + 1, shape[2:end]...) : shape
    libs_here = filter(l -> available(l, kind), libs)
    # Julia-side plans are made and checked once per case; the separate-process libraries have none.
    plans = Dict{String,Any}()
    for lib in libs_here
      if lib in ("fftl", "mpsgraph")
        tp, p = julia_plan(lib, kind, xd, n1)
        y = AT(Array{OT}(undef, yshape))
        mul!(y, p, xd); SYNC()
        err = maximum(abs.(ComplexF64.(Array(y)) .- ref)) / refmax
        plans[lib] = (tp = tp, p = p, y = y, err = err)
      end
    end
    isempty(libs_here) && continue
    for round in 1:rounds
      order = isodd(round) ? libs_here : reverse(libs_here)
      for lib in order
        if haskey(plans, lib)
          pl = plans[lib]
          sec = per_transform(() -> mul!(pl.y, pl.p, xd), reps)
          plan, err = pl.tp, pl.err
        else
          # separate-process libraries: the harness takes its own times
          kw = kind == "c2r" ? (n1 = n1,) : (;)
          yo, t = lib == "vkfft" ? vkfft(VKKIND[kind], x; reps, kw...) :
                                   cufft_native(Symbol(kind), x; reps, kw...)
          sec, plan = 1e-6 * t.exec_us, 1e-3 * t.plan_ms
          err = maximum(abs.(ComplexF64.(yo) .- ref)) / refmax
        end
        line = join((machine_label, tier_name, 0, lib, kind, T, length(shape), join(shape, "x"), n,
                     family, round, @sprintf("%.4e", sec), @sprintf("%.3e", plan), @sprintf("%.2e", err)), ",")
        open(io -> println(io, line), out, "a")
      end
    end
    plans = nothing
    GC.gc(false)
  end
  println("PROGRESS $total $total"); flush(stdout)
end
