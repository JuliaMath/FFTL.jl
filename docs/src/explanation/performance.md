# Performance

This page describes what the design does for speed and how speed is measured. The numbers themselves
come with the [benchmark release](@ref "Benchmarks"), not here.

## Where the speed comes from

Speed depends on four things, and each has its own part of the code:

1. **Kernels.** Codelets are straight-line, SIMD and specialised for the radix, the element type
   and the vector width (see [Implementation](@ref "Implementation")).
2. **Radix sequence.** The rule in `src/factor.jl` decides how many stages a length takes and how
   much each stage moves through the cache. It was derived from exhaustive searches, and the
   measured effect of a choice is in the comments of that file.
3. **Memory traffic.** Stockham chains never transpose, batches of lines are blocked to fit the
   cache of one core, and two dimensions are fused when their plane fits in cache.
4. **Threads.** Work is split only when the transform is large enough to pay for it, see
   [Threads](@ref "Threads").

Plans carry everything that does not change between transforms, so the cost of planning is paid
once. The plan cache and the precompile workload remove the compilation cost from the first call.

## Benchmark protocol

A claim about speed is a measurement made under a fixed protocol. The benchmark release will
follow this one:

- **A/B on the same machine.** Two versions run interleaved, one process per variant, on the same
  inputs. A change is accepted only when its median is better outside the noise.
- **Noise is measured.** The same version is run against itself (A/A) on each machine and thread
  count. A difference inside that band is reported as a tie.
- **Competitors at their best.** Each competitor is built and planned with its own recommended
  settings, and the exact library and version are named in the results.
- **Held-out sizes.** Sizes used for tuning are never used for the reported comparison; the
  reported set is drawn before the run.
- **Accuracy next to speed.** Every size is also checked for accuracy against a high-precision
  reference, so a faster kernel cannot win by being less exact.

The [Benchmarks](@ref "Benchmarks") page lists what is measured and on which machines.

## What a plan costs

Three costs are separate in every measurement: the first plan in a fresh process (which compiles),
the plan once compiled, and the transform itself. Benchmarks report them separately, because in an
application they happen at different times.

## Trade-offs

- Other number types take the generic path, which has no SIMD and no threads.
- DCT-I and DST-I are computed through a real DFT of about twice the length, and DCT-IV of odd
  length likewise. The extra length shows in the measurements.
- On the GPU, the real-to-real kinds are not tuned yet.
- A length with prime factors above 73 is resampled: one transform of a smooth length
  ``m ≥ 2q`` and ``w`` multiply-adds per output, where ``q`` is at most the part of the length
  made of those prime factors (the smooth part is split off, see
  [Algorithms](@ref "Lengths with a smooth part")). A prime length pays this in full.
