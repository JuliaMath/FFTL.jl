# Threads

Plans take a `threads` keyword, which defaults to `Threads.nthreads()`. Start Julia with the
threads you want (`julia -t 8` or `julia -t auto`); FFTL does not start threads itself.

```@example threads
using FFTL, LinearAlgebra, Random
Random.seed!(6)

x = randn(ComplexF64, 512, 512)
p8 = plan_fft(x; threads = 8)
y8 = p8 * x
y8 ≈ fft(x)
```

The threaded result agrees with the single-threaded one up to rounding. Long one-dimensional
transforms use a different algorithm on several threads (four-step), so their last bits can differ.
Very long lines (16 times the last-level cache of a core and more) take the four-step on one
thread as well.

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
- A long line whose length has a large prime factor (at least ``2^{14}``) runs on threads as
  well. When the smooth part of the length is split off (``n = s\,q``, see
  [Algorithms](@ref "Lengths with a smooth part")) and ``s`` is at least the number of threads,
  the ``s`` Bluestein transforms of length ``q`` are spread over the threads; the result is
  bitwise the same as on one thread. Otherwise one Bluestein transform of the whole length
  runs with its convolution transforms as four-step transforms on the threads.

A 1D transform on a machine where some cores are busy is held up by the slowest piece, so fewer
threads than cores can be the faster choice there.

## Nested threading

Code that already threads over its own work should not also ask FFTL for threads. The
oversubscription is worse than the parallelism it adds:

```julia
chunks = Iterators.partition(eachindex(images), cld(length(images), Threads.nthreads()))
@sync for c in chunks
  Threads.@spawn begin
    p = plan_fft(images[first(c)]; threads = 1)
    for i in c
      mul!(out[i], p, images[i])
    end
  end
end
```

Each task makes its own plan with `threads = 1`; see [Plans and reuse](@ref "One plan per task").

## Plans are not shared between tasks

A plan owns its work buffers and its thread buffers. Two tasks applying one plan at the same
time corrupt each other's results. Make one plan per task (see [Plans and reuse](@ref "Plans and reuse")).

## Apple silicon

With `-t auto`, Julia starts one thread per core, efficiency cores included. A thread on an
efficiency core can hold up a threaded pass, so on Apple silicon start Julia with the number of
performance cores (`julia -t 8` on an M1 Max). Measured on the M1 Max, the 64³ transform with 8 threads was slower when the
efficiency cores took part. The libfftl build uses the performance cores for the same reason
(see [libfftl](@ref "C library (libfftl)")).
