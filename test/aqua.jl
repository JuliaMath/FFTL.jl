# Package quality checks (method ambiguities, piracy, compat bounds, stale dependencies, ...).
using Aqua

@testset "Aqua" begin
  Aqua.test_all(FFTL)
end
