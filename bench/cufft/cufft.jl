# Julia side of the native cuFFT harness (cufft_bench.cpp): runs one transform in the harness
# process and returns its output and timings.
#   include("bench/cufft/cufft.jl"); y, t = cufft_native(:r2c, x)   # t = (plan_ms, first_ms, exec_us)
# Kinds: :c2c, :bc2c, :r2c, :c2r (needs n1 = size of the real array). ndims: transform along the
# first ndims dimensions, batch over the others.
const CUFFT_BIN = joinpath(@__DIR__, "cufft_bench")

function cufft_native(kind::Symbol, x::AbstractArray; n1 = nothing, ndims = Base.ndims(x), reps = 50, samples = 7)
  T = real(eltype(x))
  sz = kind === :c2r ? (n1, size(x)[2:end]...) : size(x)
  osz = kind === :r2c ? (size(x, 1) ÷ 2 + 1, size(x)[2:end]...) : kind === :c2r ? sz : size(x)
  OT = kind in (:c2c, :bc2c, :r2c) ? Complex{T} : T
  dir = mktempdir()
  fin = joinpath(dir, "in.bin"); fout = joinpath(dir, "out.bin")
  write(fin, collect(x))
  prec = T === Float64 ? "f64" : "f32"
  dims = sz[1:ndims]; b = prod(sz[ndims+1:end]; init = 1)
  args = b > 1 ? [string.(dims)..., "b$b"] : string.(dims)
  line = readchomp(`$CUFFT_BIN $kind $prec $fin $fout $reps $samples $args`)
  y = Array{OT}(undef, osz)
  read!(fout, y)
  rm(dir; recursive = true)
  v = parse.(Float64, split(line))
  y, (plan_ms = v[1], first_ms = v[2], exec_us = v[3])
end
