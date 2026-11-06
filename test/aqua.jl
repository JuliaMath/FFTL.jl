# Package quality checks (method ambiguities, unbound type parameters, piracy, project hygiene).
using Aqua

@testset "Aqua" begin
  # src/abstractffts.jl adds methods to AbstractFFTs's plan_fft, plan_rfft, … for
  # AbstractArrays of floating-point numbers: the function is AbstractFFTs's and the argument
  # types are Base's, which Aqua reports as type piracy. This is how an AbstractFFTs backend
  # works (FFTW.jl adds its StridedArray methods the same way); the signatures are less
  # specific than FFTW.jl's so that FFTW keeps its arrays when both are loaded.
  Aqua.test_all(FFTL; piracies = (treat_as_own = [AbstractArray],))
end
