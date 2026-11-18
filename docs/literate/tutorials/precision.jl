# # One function, any precision
#
# FFTL's functions take `Float32`, `Float64` and every other floating-point type, and arrays on
# the GPU, with the same calls. A function written once against them works for all of them. Here
# it is a spectral derivative, a method whose accuracy is limited only by the arithmetic, so the
# precision of the type is what one sees.

using FFTL, CairoMakie

# ## The spectral derivative
#
# For a smooth periodic function sampled at ``n`` points of ``[0, 2\pi)``, the derivative is the
# inverse transform of ``i k \hat u_k``. The coefficient at the Nyquist frequency of an even
# length has no partner, so it is set to zero. Nothing in the function names a type:

function spectral_derivative(u::AbstractVector{T}) where {T<:Real}
  n = length(u)
  U = fft(u)
  k = [j <= n ÷ 2 ? j : j - n for j in 0:n-1]
  iseven(n) && (U[n÷2+1] = 0)
  real(ifft(Complex{T}.(im .* k) .* U))
end

# The test function ``u(x) = e^{\sin x}`` has the derivative ``\cos(x)\, e^{\sin x}``:

function derivative_error(T, n)
  x = [2T(π) * j / n for j in 0:n-1]
  u = exp.(sin.(x))
  maximum(abs, spectral_derivative(u) .- cos.(x) .* u)
end

derivative_error(Float64, 64)

# ## Three precisions
#
# The error falls exponentially with ``n`` until it reaches the rounding of the type. `BigFloat`
# is transformed in its own precision (here 256 bits), not converted to `Float64`, so it keeps
# falling far below what `Float64` can show. Every ``n`` from 4 to 120 is used, primes included:

setprecision(BigFloat, 256)
ns = 4:2:120
errs = Dict(T => [max(Float64(derivative_error(T, n)), 1e-80) for n in ns]
            for T in (Float32, Float64, BigFloat));

fig = Figure(size = (640, 380))
ax = Axis(fig[1, 1]; xlabel = "points n", ylabel = "max error", yscale = log10,
          limits = (nothing, (1e-80, 1)))
for T in (Float32, Float64, BigFloat)
  lines!(ax, ns, errs[T]; label = string(T))
  T === BigFloat || hlines!(ax, eps(T); linestyle = :dash, color = :gray)
end
axislegend(ax; position = :lb)
fig

# `Float32` and `Float64` level off near their ``\varepsilon`` (dashed) and then rise slowly: the
# factor ``k`` amplifies the rounding of the high frequencies. That is a property of
# differentiation, not of the transform.

# ## The same function elsewhere
#
# Other floating-point types work the same way, for example `Double64` from DoubleFloats.jl or
# `Float128` from Quadmath.jl. Transforms of these types take a generic path without SIMD and
# threads, which is slower, so they are the tool for checking results, not for production.
#
# On a GPU the arrays come from the GPU package. With KernelAbstractions.jl loaded, the same
# FFTL functions run there (see [GPU](../../howto/gpu.md)); the derivative would be written with
# broadcasts instead of the comprehension for `k`:
#
# ```julia
# using FFTL, Metal, KernelAbstractions
# u = MtlArray(ComplexF32.(exp.(sin.(2π .* (0:1023) ./ 1024))))
# U = FFTL.fft(u)
# ```
