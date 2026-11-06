# Compares the outputs of the FFTW and libfftl builds of fftw3_test.c (fftw3_ctest.sh): every
# file of build/fftw3_test/fftw-<variant> against the same file of fftl-<variant> (and of
# dropin-plain: the FFTW-linked binary run on libfftl), relative to the largest value, and the
# libfftl runs with each other (they must agree exactly).

dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "build", "fftw3_test")
tol(name) = startswith(name, "64_") ? 1e-12 : 2e-5

function load(path, name)
  bytes = read(path)
  startswith(name, "64_") ? reinterpret(Float64, bytes) : Float64.(reinterpret(Float32, bytes))
end

ncmp = 0
nfailed = 0
for (fa, fb) in (("fftw-plain", "fftl-plain"), ("fftw-complex", "fftl-complex"), ("fftw-plain", "dropin-plain"))
  variant = fa * " vs " * fb
  a = joinpath(dir, fa)
  b = joinpath(dir, fb)
  names = sort(filter(endswith(".bin"), readdir(a)))
  missing_ = setdiff(names, filter(endswith(".bin"), readdir(b)))
  isempty(missing_) || (global nfailed += length(missing_); println("missing in the libfftl build: ", missing_))
  worst = (0.0, "")
  for nm in setdiff(names, missing_)
    x = load(joinpath(a, nm), nm); y = load(joinpath(b, nm), nm)
    global ncmp += 1
    if length(x) != length(y)
      global nfailed += 1; println("FAIL $variant $nm: lengths $(length(x)) and $(length(y))")
      continue
    end
    e = maximum(abs.(x .- y); init = 0.0) / max(maximum(abs.(x); init = 0.0), floatmin())
    e > worst[1] && (worst = (e, nm))
    e < tol(nm) || (global nfailed += 1; println("FAIL $variant $nm: rel diff $e"))
  end
  println("fftw3_compare $variant: $(length(names)) outputs, largest relative difference $(worst[1]) ($(worst[2]))")
end
# Same library, same inputs: identical bits. For dropin-plain this also shows that the
# FFTW-linked binary really ran on libfftl.
for other in ("fftl-complex", "dropin-plain")
  a = joinpath(dir, "fftl-plain"); b = joinpath(dir, other)
  for nm in filter(endswith(".bin"), readdir(a))
    global ncmp += 1
    read(joinpath(a, nm)) == read(joinpath(b, nm)) || (global nfailed += 1; println("FAIL fftl-plain vs $other: $nm differs"))
  end
end
println("fftw3_compare: $ncmp comparisons, $nfailed failed")
nfailed == 0 || exit(1)
