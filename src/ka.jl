# Scalar versions of the codelet operations (complexvec.jl works on SIMD vectors): the generated
# codelets run unchanged on scalar Complex values, as in the KernelAbstractions extension.
@inline fcadd90(a::Complex{T}, b::Complex{T}) where {T} = Complex(real(a) - imag(b), imag(a) + real(b))
@inline fcadd270(a::Complex{T}, b::Complex{T}) where {T} = Complex(real(a) + imag(b), imag(a) - real(b))
@inline ccmul(a::Complex{T}, w::Complex{T}) where {T} =
  Complex(muladd(real(a), real(w), -imag(a) * imag(w)), muladd(real(a), imag(w), imag(a) * real(w)))
@inline cconst(::Type{Complex{T}}, re, im) where {T} = Complex{T}(re, im)

"""
    plan_ka(backend, T, dims, σ; region = 1:length(dims), gmax = 64, radices = nothing)

DFT plan (σ = -1 forward, +1 unnormalised backward) for arrays of size `dims` with element type
`Complex{T}` on a KernelAbstractions `backend` (Metal, CUDA, ROCm, oneAPI, CPU), along the
dimensions `region`. Requires `using KernelAbstractions`; with a GPU package loaded,
`FFTL.plan_fft(x)` etc. return these plans for GPU arrays. Apply with `p * x` or `mul!(y, p, x)`
(`y !== x`); kernels are queued on the backend without synchronization.
"""
function plan_ka end
