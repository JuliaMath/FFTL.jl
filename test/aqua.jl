# Package quality checks (method ambiguities, piracy, compat bounds, stale dependencies, ...).
using Aqua

@testset "Aqua" begin
  # The AbstractFFTs interface (src/interface.jl, src/real.jl) adds methods to AbstractFFTs's
  # plan_fft/plan_bfft/… for AbstractArrays. The function is AbstractFFTs's and the argument
  # types are Base/AbstractFFTs's, which Aqua reports as type piracy. This is the intended
  # design (Variant A, decision 2026-11-06 in notes/decisions.md of the workspace): FFTL
  # extends AbstractFFTs's DFT interface exactly the way FFTW.jl does — FFTW.jl also adds its
  # StridedArray methods to AbstractFFTs.plan_fft. AbstractArray is therefore treated as own.
  Aqua.test_all(FFTL; piracies = (treat_as_own = [AbstractArray],))
end
