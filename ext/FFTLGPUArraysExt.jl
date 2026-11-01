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

end
