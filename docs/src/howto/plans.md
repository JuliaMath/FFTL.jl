# Plans and reuse

`fft(x)` makes a plan, applies it and throws the plan away. That is the right call for a
one-off transform. In a loop, make the plan once: everything that depends only on the size and
the dimensions is computed in the plan, and applying it costs only the arithmetic.

```@example plans
using FFTL, LinearAlgebra

x = randn(ComplexF64, 512, 512)
p = plan_fft(x)             # forward plan for arrays like x
y = similar(x)
mul!(y, p, x)               # out of place, into a preallocated y
y ≈ fft(x)
```

## In place

A plan can transform an array into itself, as `mul!(x, p, x)`. The plan's buffers are used
internally, so this does not allocate the output. `p * x` allocates a new array, and
`mul!(y, p, x)` writes into a preallocated `y`.

```@example plans
x2 = copy(x)
mul!(x2, p, x2)                  # in place
x2 ≈ y
```

## Inverses

`plan_ifft` and `plan_bfft` are the normalised and the unnormalised backward transform. An
existing forward plan has its inverse as `inv(p)`, and `p \ y` applies it:

```@example plans
pinv = inv(p)
pinv * y ≈ x

p \ y ≈ x
```

## One plan per task

A plan owns its work buffers. Two tasks applying the same plan at the same time would write into
the same buffers, so a plan must not be shared between tasks. Plans of the same size share their
twiddle tables and kernels, so making one more plan is cheap. The pattern for a threaded loop
is one plan per task, each task working through its own chunk of the items:

```julia
chunks = Iterators.partition(eachindex(xs), cld(length(xs), Threads.nthreads()))
@sync for c in chunks
  Threads.@spawn begin
    p = plan_fft(xs[first(c)]; threads = 1)   # this task's plan
    for i in c
      mul!(ys[i], p, xs[i])
    end
  end
end
```

Indexing a vector of plans with `Threads.threadid()` is not safe: a task can move to another
thread while it runs, and two tasks can then use the same plan.

## Threads inside a plan

Plans take a `threads` keyword, which defaults to `Threads.nthreads()`. Code that is already
threaded over its own work should pass `threads = 1`, see [Threads](@ref "Threads").

```@example plans
p1 = plan_fft(x; threads = 1)
p1 * x ≈ fft(x)
```

## When a plan is expensive

A plan for a new radix sequence compiles its kernels, which takes milliseconds for the sizes
in the precompile workload and up to a few seconds for others. After that, plans of the same
size are a lookup in a cache, and transforms never compile. Kernels are kept in a
cache of 128 entries that drops the least recently used entry when it is full
(`FFTL.clear_cache!()` empties it). To remove the compilation cost
from the first call entirely, see [Precompilation](@ref "Precompilation").
