# Precompilation

The first plan for a new radix sequence compiles its kernels. Without preparation, that takes
seconds for a size nobody has used before, which matters in short scripts, in a notebook that
restarts, or in a service that starts and must answer quickly.

FFTL ships a precompile workload that runs common plans and transforms while the package is
precompiled, so the compiled code is part of the package image. It is selected with a preference:

```julia
using Preferences, FFTL
Preferences.set_preferences!(FFTL, "precompile_workload" => "broad"; force = true)
```

The next load recompiles FFTL with the new setting. The values are:

| value | content |
|:--|:--|
| `"lean"` (default) | lengths 1024, 1000 and 1009 in one dimension, 16 × 16 and 8 × 8 × 8, both element types, complex transforms, real transforms and `plan_pass` |
| `"broad"` | lengths 10⁵, 2²⁰, 128, 360, 243, the sizes 256 × 256 and 64 × 64 × 64, and a four-step case with four threads |
| `"none"` | no workload: smallest image, compilation at the first use |

The workload applies plans and runs transforms, not just constructs them, because most of the
compiled code is generated for the transform itself. A plan of a size the workload did not cover
still compiles its own kernels when first used; the workload only moves that cost to package
precompilation.

`FFTL.PRECOMPILE_WORKLOAD` reports the value the package was built with.

The trade-off is size against latency. On an Apple M1 Max, the lean workload took about 32 s to
precompile and makes the package image about 96 MB. The first plan and transform of a covered size
then take milliseconds instead of seconds. Choose `"none"` when the image size matters more than
the first call, for example in continuous integration.
