# Package size and load time

A Julia package that is precompiled stores its compiled code and its type information in a package
image, a shared library in the compiled cache (`~/.julia/compiled/v1.x/FFTL/*.dylib`, `.so` on Linux).
Loading the package maps that file and restores its contents. FFTL's image used to be 146 MB, 130 MB of
it stored intermediate code, and `using FFTL` took 0.17 s. This page explains where that came from, what
keeps it small now, what is left, and how to check it.

## Results

Apple M1 Max, Julia 1.13.3, the default (`"lean"`) precompile workload:

| | before | plain methods | plain methods + lazy radices |
|:--|--:|--:|--:|
| package image | 146 MB | 50 MB | 32 MB |
| optimised IR stored in the image | 469 000 statements | 0 | 0 |
| precompiling FFTL | 51 s | 47 s | 25 s |
| `using FFTL` | 0.17 s | 0.12 s | 0.10 s |
| first call, n = 1088 = 17·64 | 13 ms | 13 ms | 68–80 ms |
| steady state | | unchanged (ratio 0.985–1.023) | unchanged except 0.02 µs per transform of a length with a factor from 17 to 31 |

The steady-state ratios are interleaved A/B runs with one thread over 20 cases (`Float64` and `Float32`,
n = 64 … 2²⁰, 256², 100³). The full test suite passes on x86 with all of it in place, and the C library
builds with `juliac --trim` and passes its differential test against FFTW.

## Why the image was large

Julia drops the optimised intermediate code (IR) of a method that is not worth inlining when it writes
the package image; only the machine code stays. There is one exception: a method that has no stored
source keeps its IR. In Julia 1.13 (`src/staticdata.c`, the loop that decides what to discard) the test is
`def->source == NULL`, and a `@generated` method has no source, because the source is produced by running
its generator. Every precompiled specialisation of a `@generated` method therefore keeps its complete
optimised IR in the image.

FFTL's stage kernels, `run!` and `run_lanes!` are `@generated`: the codelet of a radix is a long
straight-line body that the generator builds for one radix, sign and vector type. The workload
precompiled 1776 such specialisations with 469 000 statements in total, which is the 130 MB. The machine
code of the whole package was 4.6 MB. This was checked in a small package with the same bodies written as
30 specialisations of 100 to 3000 statements: 281 MB as `@generated`, 13.8 MB as plain methods built with
`@eval`. Which methods keep their IR is a property of Julia's package image writer; it may change in
later Julia versions, so re-run the count below after an upgrade.

## What keeps it small

1. **Plain methods for the fixed set.** The stage kernels (`STAGE_KERNELS` in `src/kernels.jl`: radix ×
   sign × vector type) are defined in `src/kernel_methods.jl` with `@eval`, from the functions that build
   their bodies. A plain method keeps its (compressed) source and its machine code, not its optimised IR.
   There are no `@generated` stage kernels.
2. **One kernel for the large prime radices.** The radices 17 to 73 have no codelets of their own: one
   kernel (`src/primestage.jl`) takes the radix and a table as arguments and is compiled once per vector
   type. Straight-line codelets for 17, 19, 23, 29 and 31 were 54 % of the statements of the stage kernels;
   they used to be left out of the image and compiled at the first call (70–100 ms, the "lazy radices"
   column of the table above).

To add a radix up to 16, put it in `RADICES`; `kernel_methods.jl` defines its methods. A larger prime
needs no code: raise `MAX_RADIX_PRIME` (see [Design decisions](@ref "Design decisions") for where the
kernel stops being faster than Bluestein's algorithm).

The numbers on this page were measured on the M1 while the smooth lengths up to 128 had compiled chains
of their own (583 more plain methods) and the radices 17–31 were compiled at first use. On an AMD EPYC
9354 (AVX-512, which has seven vector types where the M1 has three) the image file went from 68.5 MB
with the compiled chains to 58.6 MB without them and to 66.2 MB with the kernel for the large primes
(which is in the image, where the lazy codelets were not). The C library, which contains every kernel
in either case, went from 55.6 MB to 49.5 MB and to 26.8 MB (`libfftl.so`, x86-64-v3, `--trim=safe`). The M1 numbers have not been measured again.

With the workload, nearly all of the image is the stage kernels: the dispatch switches of `RChain` name
every kernel, so the first transform compiles all of them. On the EPYC 9354 (measured with the prime
limit at 127, which does not change the kernels) the image was 65 MB with the workload and 20.5 MB
without, precompilation took 61 s and 13 s, `using FFTL` 185 ms and 110 ms, and the first transform
16–37 ms with the workload and 23–35 s without.

## What is left

The 32 MB consist of about 3.8 MB for the package without the kernels, 8.6 MB for the definitions of the
plain kernel methods (mostly their stored sources), and 19.6 MB that the precompile workload adds, of
which 2.5 MB is machine code and the rest metadata of the compiled instances. With `"none"` as the
workload (see [Precompilation](@ref "Precompilation")) the image is 12 MB.

Load time is not proportional to size. The 56–61 ms that FFTL itself takes with the workload fall to
20–26 ms without it, about 2 ms per MB of workload; SIMD.jl takes 20–26 ms, Dates 16–20 ms (it comes
through PrecompileTools, Preferences and TOML), and AbstractFFTs 2–3 ms.

Further steps that were considered:

| step | gain | cost |
|:--|:--|:--|
| methods generic in the element and vector type instead of one per vector type | about 4 MB (stored sources ÷ 3) | the compiler must fold `ncomplex` and `sizeof` of the type parameters; needs a speed check |
| no PrecompileTools | about 25 ms of load time (Dates, TOML, Preferences) | the workload is written with `precompile` calls |
| no `Float32` in the workload | 16 MB (at 73 MB) | 6 s at the first `Float32` call |

Replacing the preference by environment variables did not shorten the load, and Julia does not notice a
changed variable, so a stale image would stay in use. The pass plans do not take zero masks any more (their
kernels were `@generated` with a mask per segment), so no pruned kernel exists to store.

## Checking it

The size of the image is the size of the file in the compiled cache. To count the code instances of
FFTL's own methods that hold optimised IR:

```julia
using FFTL

function stored_ir(mod::Module)
    instances = 0; statements = 0
    for name in names(mod; all = true)
        isdefined(mod, name) || continue
        f = getfield(mod, name)
        f isa Function && parentmodule(f) === mod || continue
        for m in methods(f), mi in Base.specializations(m)
            mi isa Core.MethodInstance && isdefined(mi, :cache) || continue
            ci = mi.cache
            while true
                if isdefined(ci, :inferred) && ci.inferred isa Core.CodeInfo
                    instances += 1; statements += length(ci.inferred.code)
                end
                isdefined(ci, :next) || break
                ci = ci.next
            end
        end
    end
    (; instances, statements)
end

stored_ir(FFTL)    # (instances = 0, statements = 0) with the plain methods
```

Measure the effect of a change in a fresh depot (`JULIA_DEPOT_PATH=scratch:$HOME/.julia`), so that the
cache of your normal depot is not overwritten, and compare the size of the new image file and the time
`Pkg.precompile()` reports for FFTL.
