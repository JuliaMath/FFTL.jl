# Implementation

The [algorithms](@ref "Algorithms") explain what is computed. This page explains how the code
is organised and why, for someone who wants to change it.

## Specialisation by type, not by configuration

A plan is a Julia value, and its type says what it is. A Stockham chain is
`Chain{T,Rs,σ,K}`: the element type `T`, the radix sequence `Rs` as a tuple of integers, the sign
`σ` and the kernel type `K` are type parameters. The compiler therefore sees the complete stage
sequence when it compiles the transform, unrolls the loop over stages where it can, and emits
machine code for exactly that sequence. The price is one compilation per radix sequence, which
is what the [plan cache](@ref "Plans and reuse") and the [precompile workload](@ref "Precompilation")
are for.

The codelets follow the same rule. `@generated` functions build the straight-line code for a
given radix, element type and vector width when they are first called with those types.

## Codelets on SIMD vectors

The arithmetic of a codelet is written on complex SIMD vectors (`src/complexvec.jl`). A vector
holds several complex values, and the same instruction stream runs on all of them. For a
`Complex{Float32}` array on a 128-bit machine that is two values per register. Lanes of the
multidimensional transforms are the vector elements, so the same codelet serves a single
line and a batch of lines.

The scalar versions in `src/ka.jl` run the same generated codelets on plain `Complex` values. That
is how the KernelAbstractions kernels use the codelet generator without a second implementation.

## Kernel cache

Kernels are cached by kind, element type, length, sign and radix sequence. The cache has 128
entries and drops the least recently used one when it is full. A Bluestein kernel is two entries,
its chain and its tables. `clear_cache!` empties it. Within one plan, dimensions with the same
length and stride class share a kernel object.

## Cache sizes and blocks

The work of a batched transform is split into blocks of lines so that the work buffers of one
block stay in the cache of one core. The cache sizes are read when the package loads: `sysctl`
on macOS for the performance cores, `sysfs` on Linux for CPU 0. Where the query fails, defaults
for the two development machines stand in. `FFTL.cpu_caches()` reports the values, and
`FFTL.set_cpu_caches!` overrides them for a session.

The thresholds that decide between one thread and several, between the lane and the plane forms,
and between the four-step and the plain algorithm live in one place, `FFTL.TUNING` (`src/tuning.jl`).
Changing one changes the plans made afterwards.

## Thread tasks and buffers

Threads come from Julia's task scheduler. A threaded pass hands chunks of its work to a fixed
number of tasks, and each task owns its own buffers, allocated when the plan is made. Two
buffers per task are the minimum: the Stockham stages ping-pong between them.

That buffer ownership is why a plan must not run on two tasks at once. The plan is a
container for the buffers, not a stateless function.

## Two forms of the same chain

There are two representations of a Stockham chain:

- `Chain` (`src/plan.jl`) has the radix sequence in its type. It is the fastest form, and what
  the normal API uses.
- `RChain` (`src/rchain.jl`) has the radix sequence as data, and calls codelets from a fixed set
  of radices and signs in a plain loop.

The second form exists for the C library. `juliac --trim` removes everything it cannot resolve
at build time, and a chain whose type is only known at run time cannot be compiled that way.
`RChain` keeps every call resolvable. The price is a little run-time dispatch per stage, which
is why the Julia API uses `Chain`. The [libfftl how-to](@ref "C library (libfftl)") describes
what the library is built from.

## Lengths and the factorisation rule

`src/factor.jl` holds the rule that turns a length into radices. Its header comment records the
measurements the rule was derived from, and the constants in it are not tuned at run time. A
change to the rule needs a measurement on the machines it is meant for: the rule that is best on
one architecture can be slower on another.

## Element types

The fast path is written for `Float32` and `Float64`. Other floating-point types use `GPlan`
(`src/generic.jl`): a recursive mixed-radix transform on plain arrays, with Bluestein's algorithm
for large prime factors, twiddles computed in the type itself, no SIMD and no threads. It is much
slower, and its purpose is accuracy and generality.

## The GPU path

`ext/FFTLKernelAbstractionsExt.jl` holds the KernelAbstractions extension. Its kernels compute one
pass of a transform in workgroup memory. The pass sizes follow the workgroup memory the device
offers; the rules were measured on an Apple M1 Max, and the CUDA path on an RTX A6000.
