# FFTL.plan_fft & co. for GPU arrays: KernelAbstractions plans on the array's backend.
module FFTLGPUArraysExt

using FFTL, KernelAbstractions, GPUArraysCore
import AbstractFFTs

FFTL.plan_fft(x::AbstractGPUArray, region = 1:ndims(x); kw...) =
  FFTL.plan_ka(get_backend(x), FFTL.plan_type(x), size(x), -1; region, kw...)
FFTL.plan_bfft(x::AbstractGPUArray, region = 1:ndims(x); kw...) =
  FFTL.plan_ka(get_backend(x), FFTL.plan_type(x), size(x), 1; region, kw...)
FFTL.plan_ifft(x::AbstractGPUArray, region = 1:ndims(x); kw...) =
  AbstractFFTs.ScaledPlan(FFTL.plan_bfft(x, region; kw...), AbstractFFTs.normalization(FFTL.plan_type(x), size(x), region))


# real transforms: dims of the real array; T the real element type
rdims(sz, d, n) = ntuple(i -> i == d ? n : sz[i], length(sz))
FFTL.plan_rfft(x::AbstractGPUArray, region = 1:ndims(x); kw...) =
  FFTL.plan_ka_rfft(get_backend(x), FFTL.realtype(x), size(x); region)
FFTL.plan_brfft(X::AbstractGPUArray, d::Integer, region = 1:ndims(X); kw...) =
  FFTL.plan_ka_brfft(get_backend(X), FFTL.realtype(X), rdims(size(X), minimum(region), Int(d)); region)
function FFTL.plan_irfft(X::AbstractGPUArray, d::Integer, region = 1:ndims(X); kw...)
  p = FFTL.plan_brfft(X, d, region)
  AbstractFFTs.ScaledPlan(p, AbstractFFTs.normalization(FFTL.realtype(X), AbstractFFTs.output_size(p), region))
end
FFTL.plan_r2r(x::AbstractGPUArray, kinds, region = 1:ndims(x); kw...) =
  FFTL.plan_ka_r2r(get_backend(x), FFTL.realtype(x), size(x), kinds; region)

end
