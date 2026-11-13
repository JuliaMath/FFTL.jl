# Julia side of the VkFFT harness (vkfft_bench.cpp): runs one transform in the harness process
# and returns its output and timings.
#   include("bench/vkfft/vkfft.jl"); y, t = vkfft(:r2c, x)   # t = (plan_ms, first_ms, exec_us, cached, batch_us)
# ndims: transform along the first ndims dimensions, batch over the others.
# Kinds: :c2c, :bc2c, :r2c, :c2r (needs n1 = size of the real array), :dct1…:dct4, :dst1…:dst4.
const VKFFT_BIN = joinpath(@__DIR__, "vkfft_bench")
const VKFFT_CACHE = joinpath(@__DIR__, "cache")

function vkfft(kind::Symbol, x::AbstractArray; n1 = nothing, ndims = Base.ndims(x), reps = 50, samples = 7, cache = true)
  T = real(eltype(x))
  sz = kind === :c2r ? (n1, size(x)[2:end]...) : size(x)
  osz = kind === :r2c ? (size(x, 1) ÷ 2 + 1, size(x)[2:end]...) : kind === :c2r ? sz : size(x)
  OT = kind in (:c2c, :bc2c, :r2c) ? Complex{T} : T
  dir = mktempdir()
  fin = joinpath(dir, "in.bin"); fout = joinpath(dir, "out.bin")
  write(fin, collect(x))
  prec = T === Float64 ? "f64" : "f32"
  c = "-"
  if cache && Sys.islinux()
    mkpath(VKFFT_CACHE)
    c = joinpath(VKFFT_CACHE, "$(kind)_$(prec)_$(join(sz, "x")).bin")
  end
  dims = sz[1:ndims]; b = prod(sz[ndims+1:end]; init = 1)
  args = b > 1 ? [string.(dims)..., "b$b"] : string.(dims)
  c == "-" || (c = replace(c, ".bin" => "_b$b.bin"))
  line = readchomp(`$VKFFT_BIN $kind $prec $fin $fout $reps $samples $c $args`)
  y = Array{OT}(undef, osz)
  read!(fout, y)
  rm(dir; recursive = true)
  v = parse.(Float64, split(line))
  y, (plan_ms = v[1], first_ms = v[2], exec_us = v[3], cached = v[4] == 1, batch_us = v[5])
end
