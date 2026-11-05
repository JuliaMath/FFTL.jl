# Contribute

FFTL is a small code base with one hard requirement: every change is checked against the
tests, and every claim about speed is checked against a measurement. This page shows how to
set up, what to run, and what a change must come with.

## Set up

```bash
git clone https://github.com/JuliaMath/FFTL.jl
cd FFTL.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Work on a branch and open a pull request against `main`. Commit messages are in English and
describe what the code does now, not how it got there.

## Run the tests

```julia
using Pkg; Pkg.test("FFTL")
```

This runs a quick selection that covers every feature, in about 1.5 minutes. Two environment
variables change it:

| variable | effect |
|:--|:--|
| `FFTL_TEST_FULL=true` | the full lists of sizes, kinds and element types (weekly in CI) |
| `FFTL_TEST_VERBOSE=true` | the time of every test set |

The test files: `runtests.jl` (transforms against a reference DFT, both precisions, all kinds),
`ka.jl` (KernelAbstractions plans, on the CPU backend), `inference.jl` (type inference of the
public functions), `radix_table.jl` (the radix choice), and `aqua.jl` (package quality checks).
CI runs the quick tests on Linux and macOS for pushes to `main` and for pull requests, and the
full lists and Julia 1.10 once a week. The C library has its own CI job.

## Build the documentation

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The output is in `docs/build`. Every Literate example in the getting started page and in the
tutorials runs during the build, so a broken example fails the build.

## Changes to performance

A change that claims to be faster comes with an A/B measurement on the same machine, in one
process per variant, with the same inputs. Report the medians of several interleaved runs and
the spread between two runs of the same code: a difference inside that spread is no difference.
The [Performance](@ref "Performance") page describes the protocol.

## Where things are

| file | what it does |
|:--|:--|
| `src/codegen.jl` | the symbolic generator of the straight-line codelets |
| `src/kernels.jl`, `src/complexvec.jl` | codelets on SIMD vectors, the complex arithmetic |
| `src/factor.jl` | the radix sequence chosen for a length |
| `src/plan.jl`, `src/rchain.jl` | Stockham chains and Bluestein's algorithm; `rchain.jl` is the form that libfftl compiles |
| `src/threads.jl` | threads, lane blocks, the four-step algorithm |
| `src/interface.jl` | the complex plans and the public functions |
| `src/real.jl` | real transforms (r2c, c2r) |
| `src/r2r.jl` | the real-to-real kinds |
| `src/rreal.jl` | run-time real plans, the form that libfftl compiles |
| `src/generic.jl` | the path for other number types |
| `src/ka.jl`, `ext/` | GPU through KernelAbstractions |
| `src/precompile.jl` | the precompile workload |
| `src/limits.jl`, `src/tuning.jl` | cache sizes and thresholds |
| `libfftl/` | the C library |

The thresholds in `src/tuning.jl` are numbers measured on two machines. A change to them needs a
measurement on at least one machine, reported with the change.

## Licensing

FFTL is MIT-licensed. Code in a contribution must be your own or come from a source under a
permissive license (MIT, BSD, Apache), with the notice kept. Do not copy or translate code from GPL-licensed FFT
implementations, such as FFTW or KFR. Algorithms published in papers can be implemented
from the paper.
