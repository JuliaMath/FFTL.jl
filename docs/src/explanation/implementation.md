# Implementation

The [algorithms](@ref "Algorithms") explain what is computed. This page explains how the code
is organised and why, for someone who wants to change it.

## Specialisation by type for the codelets, data for the plan

A codelet is specialised by type: the stage kernels take the radix, the sign and the vector type as
type parameters, and the straight-line code of each combination is compiled once. The combinations
form a fixed set (radix × sign × vector type × element type), so all of them are compiled with the
package.

The stage sequence of a length is data. A Stockham chain (`RChain`, `src/rchain.jl`) holds its
radix sequence and one record per stage (codelet index, stride, twiddle table), and a loop calls the
codelet of each stage. Nothing is compiled when a plan of a new length is made.

## Codelets on SIMD vectors

The arithmetic of a codelet is written on complex SIMD vectors (`src/complexvec.jl`). A vector
holds several complex values, and the same instruction stream runs on all of them. For a
`Complex{Float32}` array on a 128-bit machine that is two values per register, and eight with
AVX-512. Lanes of the
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

## The stage loop

`RChain` calls codelets from the fixed set in a plain loop, selected by an integer index. A short line
is a few nanoseconds of arithmetic, so the loop is written to cost little next to it:

- A line that is one stage is one codelet call, without the loop.
- The stage records lie in one flat vector and carry the address of their twiddle table, and the loop
  reads them through a raw pointer. A reference loaded from the chain inside the loop would need a GC
  frame, which costs more than a short stage.
- The chain is a mutable struct, so it is passed as one pointer; an immutable chain is copied to the
  stack of each caller.
- The switch over the codelet index is inlined into the loop: a stage is one jump and one call.
- The rows of a plane run inside one call of the loop function.

A stage of a prime radix from 17 on calls one kernel for all such radices (`prime_stage!` in
`src/primestage.jl`), with the radix and its table as arguments; it is compiled once per vector type.
Its calls sit behind a function of their own, outside the switch, because they take more arguments
than fit in registers and would cost every case of the switch a larger stack frame.

The addresses in the stage records are valid for the chain object they were taken for. A chain
remembers its own address at that moment; a copy of a plan, or a plan stored in a package image and
loaded into another process, is at another address and links its records again before it uses them.

`RChain` is also what the C library is built from. `juliac --trim` removes everything it cannot resolve
at build time and compiles nothing at run time; the fixed codelet set is all a chain of any length needs. The
[libfftl how-to](@ref "C library (libfftl)") describes what the library is built from, and
[Design decisions](@ref "Design decisions") gives the measurements behind the single chain form. The stage kernels of
the fixed codelet set are plain methods defined with `@eval`, not `@generated` ones, to keep the package image small; see
[Package size and load time](@ref "Package size and load time").

## Lengths and the factorisation rule

`src/factor.jl` holds the rule that turns a length into radices. Its header comment records the
measurements the rule was derived from, and the constants in it are not tuned at run time. A
change to the rule needs a measurement on the machines it is meant for: the rule that is best on
one architecture can be slower on another.

## Strided layouts

Complex plans fix the shape, but read the input and output strides independently on every call.
Contiguous views use the dense executors, including plane fusion. Other strided layouts merge
leading dimensions into lanes only while both operands are contiguous there; the remaining
dimensions supply independent batch offsets. This avoids full-array copies without adding plan
storage. Arrays outside `StridedArray` retain the copying fallback.

`LinesMT` owns a child kernel and scratch buffers per task. The strided executor uses that task's
buffers and never passes more lanes than the child was built to hold. For a single strided
`Composite` line, it gathers the input into the plan's separate task scratch when that scratch
is large enough. A packed input lets the first stage process blocks instead of one small
transform per leftover element. The output is written directly with its own stride; insufficient
scratch keeps the direct strided path.

On an Apple M5 Max, macOS arm64, Julia 1.13.3 and two threads, `Float32` forward transforms of
`65×2018` views along dimension 2, with input step 2 and output step 3, took a median 469.80 µs
against 622.13 µs for the copying fallback at `289e9c6`. Forward, backward and normalized inverse
times were 24–25% lower; the maximum paired A/A differences were 1.96–3.77% across those cases.
These measurements used eight interleaved rounds in separate warmed processes. Threaded calls
allocated 592 bytes; gathering adds no plan storage. This is evidence for these layouts on this
processor, not a claim for every strided transform.

### Real plans

High-level real plans retain their dense offsets and refill separate input and output offset
vectors for strided execution. Keeping the two sets separate allows a plan to alternate between
dense arrays and views without corrupting its dense layout. The extra payload is
`2 * number_of_lines * sizeof(Int)`, or 64 bytes for four lines on a 64-bit machine, plus vector
metadata. The inverse spectrum scratch is allocated in its final shape during planning, so
preserving the input during complex passes needs no execution-time reshape or extra element storage.

On the same processor and Julia version, with one thread, a `Float64` forward `10240×4` column
block took 43.39 µs and allocated zero bytes. The explicit dense-copy workaround at `289e9c6`
took 59.85 µs and allocated 671,904 bytes; native real-view `mul!` is unavailable at that baseline.
Eight interleaved rounds gave a 27.5% time reduction, against a maximum paired A/A difference of
2.41%. The [real-data guide](@ref "Real data") shows the layout contract and an executable example.

## Element types

The fast path is written for `Float32` and `Float64`. Other floating-point types use `GPlan`
(`src/generic.jl`): a recursive mixed-radix transform on plain arrays, with Bluestein's algorithm
for large prime factors, twiddles computed in the type itself, no SIMD and no threads. It is much
slower, and its purpose is accuracy and generality.

## The GPU path

`ext/FFTLKernelAbstractionsExt.jl` holds the KernelAbstractions extension. Its kernels compute one
pass of a transform in workgroup memory. The pass sizes follow the workgroup memory the device
offers; the rules were measured on an Apple M1 Max, and the CUDA path on an RTX A6000.

A workgroup holds `G` lines (lanes) of a pass. Normally `G` divides the number of lines; a pass
over all of a dimension (one pass) may instead take the lines in memory order with any `G`
(ragged lanes: the lanes past the last line repeat it and store nothing new), which keeps the
workgroups full when the line count is prime. Lengths with a prime factor above 31 use a fused
kernel when the convolution fits one pass: Bluestein's (forward DFT of the chirped line, product
with the transformed chirp, backward DFT) or Rader's (the same with permuted loads and stores, for a
prime whose ``p - 1`` is 13-smooth), both in workgroup memory. These kernels split their length into
few, even radices (`local_radices(n, true)`), since a thread per butterfly of the largest radix
makes a small radix next to 16 take many rounds. With `rpack` the fused kernel also reads two real
lines as one complex line and writes the real transform (odd r2c, c2r, DCT-II, DST-II), see
`ext/ka_real.jl`.
