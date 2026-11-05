# GPU

With [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and a GPU package
loaded, the same functions accept GPU arrays. The example needs a GPU, so it is shown but not
run during the documentation build.

```julia
using FFTL, Metal, KernelAbstractions     # or CUDA, AMDGPU, oneAPI

xd = MtlArray(randn(ComplexF32, 1024, 1024))
yd = FFTL.fft(xd)                          # a new GPU array
p = FFTL.plan_fft(xd, 2)                   # plan along dimension 2
yd2 = similar(xd)
mul!(yd2, p, xd)                           # out of place: yd2 !== xd

Rd = FFTL.rfft(MtlArray(randn(Float32, 1024, 1024)))
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
Metal.jl's FFT in nearly all measured cases. The CUDA path is built and measured against cuFFT on an NVIDIA RTX A6000; the
benchmark results are not published yet ([Benchmarks](@ref "Benchmarks")).

Real transforms (`rfft`, `irfft`, `brfft`) run at about the speed of a complex transform of the
same size. The real-to-real kinds run through a twiddled complex DFT of twice the length. They are
correct for every kind but not tuned yet, so their speed is below the complex path.

Other GPUs run through the same code. They are not tuned, and AMD and Intel GPUs have not been
tested.

## Making plans without a GPU array

`FFTL.plan_ka(backend, T, dims, σ; region)` makes a plan for any KernelAbstractions backend,
including the CPU backend, which is useful for testing GPU code without a GPU.
`plan_ka_rfft`, `plan_ka_brfft` and `plan_ka_r2r` do the same for the real transforms.
