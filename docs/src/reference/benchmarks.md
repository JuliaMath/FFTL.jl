# Benchmarks

!!! note "In preparation"
    The benchmark results are not published yet. This page lists what the release will contain
    and how it is measured; no number on it is a result.

## What the release covers

- **Complex, real and real-to-real transforms** on the CPU, 1D to 3D, Float32 and Float64, against
  the competitors that support the same kind. Each kind has its own column; kinds without a
  competitor are shown separately.
- **Threads.** The same sizes with one thread and with the number of threads of the machine,
  with the noise band of each machine.
- **GPU.** Complex, real and real-to-real transforms against the vendor library (cuFFT on CUDA,
  MPSGraph and Metal.jl's FFT on Metal) and against VkFFT.
- **libfftl.** The C library against the Julia package, to show that the library costs nothing.
- **Planning.** The cost of the first plan in a fresh process, separate from the execution time.
- **Accuracy.** Error against a high-precision reference over the same sizes.

## How it is measured

The protocol is described on the [Performance](@ref "Performance") page: A/B runs on the same
machine, medians of interleaved runs, A/A runs for the noise band, and sizes reported separately
from the sizes used for tuning.

## Machines

| machine | role |
|:--|:--|
| Apple M1 Max | development machine, CPU and Metal GPU |
| AMD EPYC 7763 (2 sockets) | CPU server, x86-64 with AVX2 |
| NVIDIA RTX A6000 | CUDA GPU |

Each result names the library, its version, the build flags and the thread count. Competitors are
built from their current releases with native CPU flags.

## Status

| part | status |
|:--|:--|
| harness and protocol | prepared |
| CPU, M1 Max | planned |
| CPU, EPYC | planned |
| GPU, Metal | planned |
| GPU, CUDA | planned |
| libfftl | planned |
| accuracy | planned |

The results come with the first release that carries them. Until then, the functions are
documented and tested, and the speed statements in this manual are statements about design.
