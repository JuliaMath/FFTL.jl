# GPU

With [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and a GPU package
loaded, the same functions accept GPU arrays. The example needs a GPU, so it is shown but not
run during the documentation build.

```julia
using FFTL, Metal, KernelAbstractions  # or CUDA, AMDGPU, oneAPI

xd = MtlArray(randn(ComplexF32, 1024, 1024))
yd = fft(xd)                           # a new GPU array
p = plan_fft(xd, 2)                    # plan along dimension 2
yd2 = similar(xd)
mul!(yd2, p, xd)                       # out of place: yd2 !== xd

Rd = rfft(MtlArray(randn(Float32, 1024, 1024)))
```

## What to expect

- **Out of place.** GPU transforms write into a new array. `mul!(y, p, x)` needs `y !== x`.
- **Asynchronous.** Kernels are queued on the device without waiting. Copying a result to the
  host waits for them. For timing, call `KernelAbstractions.synchronize(backend)` before reading
  the clock.
- **Threads are ignored.** The device runs the parallelism.
- **Plans cost memory.** A plan holds about two buffers of the array's size on the device.
- **Plans compile.** Making a plan for a new size generates and compiles its kernels. Make plans
  once and keep them, as on the CPU.
- **Element types.** `Float32` and `Float64` where the device supports `Float64`. Complex
  transforms are the tuned path.

## What is tuned

Complex transforms use multi-stage kernels. Each kernel computes one pass of a large radix (up to
4096 points) in workgroup memory, and a transform is one to five passes per dimension. Pass and
workgroup sizes are tuned on an Apple M1 Max: there, `Float32` transforms were faster than
Metal.jl's FFT in nearly all measured cases. The CUDA path is tuned and measured on an NVIDIA RTX A6000 against cuFFT and VkFFT
([Benchmarks](@ref "Benchmarks")).

Real transforms (`rfft`, `irfft`, `brfft`) compute the half spectrum through a complex transform
of half the length, with the pre- and post-processing inside its first and last pass.

The real-to-real kinds are reduced to complex DFTs: of half the length for DCT-II to DCT-IV and
DST-II to DST-IV of even length, of length ``n - 1`` for DCT-I and ``n + 1`` for DST-I, mostly
with the extra steps inside the passes. Odd lengths of the other kinds take a complex DFT of
length ``2n``, the Hartley and halfcomplex kinds one of length ``n``. DCT-II and DST-II are the
most tuned of them; for the other kinds the speed relative to VkFFT depends on the kind and the
size, from faster to several times slower.

Other GPUs run through the same code. They are not tuned, and AMD and Intel GPUs have not been
tested.

## Making plans without a GPU array

`FFTL.plan_ka(backend, T, dims, σ; region)` makes a plan for any KernelAbstractions backend,
including the CPU backend, which is useful for testing GPU code without a GPU.
`plan_ka_rfft`, `plan_ka_brfft` and `plan_ka_r2r` do the same for the real transforms.
