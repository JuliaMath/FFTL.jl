# Plans and reuse

`FFTL.fft(x)` makes a plan, applies it and throws the plan away. That is the right call for a
one-off transform. In a loop, make the plan once: everything that depends only on the size and
the dimensions is computed in the plan, and applying it costs only the arithmetic.

```@example plans
using FFTL, LinearAlgebra

x = randn(ComplexF64, 512, 512)
p = FFTL.plan_fft(x)             # forward plan for arrays like x
y = similar(x)
mul!(y, p, x)                    # out of place: y must not be x
y ≈ FFTL.fft(x)
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
is one plan per task:

```julia
Threads.@threads for i in 1:n
  p = plans[Threads.threadid()]   # one per thread, made before the loop
  mul!(out[i], p, in[i])
end
```

## Threads inside a plan

Plans take a `threads` keyword, which defaults to `Threads.nthreads()`. Code that is already
threaded over its own work should pass `threads = 1`, see [Threads](@ref "Threads").

```@example plans
p1 = FFTL.plan_fft(x; threads = 1)
p1 * x ≈ FFTL.fft(x)
```

## When a plan is expensive

A plan for a new radix sequence compiles its kernels, which takes milliseconds for the sizes
in the precompile workload and up to a few seconds for others. After that, plans of the same
size are a lookup in a cache, and transforms never compile. Kernels are kept in a
cache of 128 entries, least recently used first out (`FFTL.clear_cache!()` empties it). To remove the compilation cost
from the first call entirely, see [Precompilation](@ref "Precompilation").
