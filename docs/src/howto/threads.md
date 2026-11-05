# Threads

Plans take a `threads` keyword, which defaults to `Threads.nthreads()`. Start Julia with the
threads you want (`julia -t 8` or `julia -t auto`); FFTL does not start threads itself.

```@example threads
using FFTL, LinearAlgebra, Random
Random.seed!(6)

x = randn(ComplexF64, 512, 512)
p8 = FFTL.plan_fft(x; threads = 8)
y8 = p8 * x
y8 ≈ FFTL.fft(x)
```

The threaded result agrees with the single-threaded one up to rounding. Long one-dimensional
transforms use a different algorithm on several threads (four-step), so their last bits can differ.

## When threads are used

Threads only help when there is enough work to split. FFTL decides per dimension, from the
number of values the transform touches:

- A transform with fewer than ``2^{15}`` values in total runs on one thread, whatever `threads`
  says. Splitting it would cost more than it saves.
- A batch of lines along a dimension is split over the threads as blocks of lines, each block
  sized to fit the cache of one core.
- A single long one-dimensional transform (a smooth length of at least ``2^{13}``) uses the
  four-step algorithm when it has more than one thread: the line is written as an ``n_1 × n_2``
  matrix, transformed in both directions with threads, and the twiddle factors in between.
- A long line whose length has a large prime factor (at least ``2^{14}``) runs Bluestein's
  algorithm on threads.

A 1D transform on a machine where some cores are busy is held up by the slowest piece, so fewer
threads than cores can be the faster choice there.

## Nested threading

Code that already threads over its own work should not also ask FFTL for threads. The
oversubscription is worse than the parallelism it adds:

```julia
Threads.@threads for i in eachindex(images)
  mul!(out[i], plans[Threads.threadid()], images[i])   # plans made with threads = 1
end
```

Pass `threads = 1` when making the plans in that case.

## Plans are not shared between tasks

A plan owns its work buffers and its thread buffers. Two tasks applying one plan at the same
time corrupt each other's results. Make one plan per task (see [Plans and reuse](@ref "Plans and reuse")).

## Apple silicon

On an Apple M1 Max, `Threads.nthreads()` counts the performance cores only if Julia is started
that way; with `-t auto` the efficiency cores join, and a thread on an efficiency core can hold
up a threaded pass. Measured on the M1 Max, the 64³ transform with 8 threads was slower when the
efficiency cores took part. The libfftl build uses the performance cores for the same reason
(see [libfftl](@ref "C library (libfftl)")).
