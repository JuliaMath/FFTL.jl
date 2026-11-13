# Design decisions

The decisions that shape FFTL, each with the reason and the evidence it rests on. A decision
stays here until something replaces it; a replaced decision is marked, so a reader who finds the
old reasoning elsewhere can see that it no longer holds. Numbers are measurements on an Apple M1 Max
and on AMD EPYC (Zen 2) unless a line says otherwise.

## Licence and provenance

**Decision.** FFTL is MIT-licensed. No GPL code is read, copied, translated or closely paraphrased.
FFTW is a benchmark competitor only. Concepts known from the literature are used; code structure,
naming and style deliberately do not follow FFTW (a 25-year-old C library with generated codelets and a
measuring planner). Permissive sources may be read; a close take-over keeps its notice. The rules for
contributors are on the [Contribute](@ref "Contribute") page and in `AGENTS.md`.

## Codelets with complex values in SIMD registers

**Decision.** A codelet works on complex values interleaved in SIMD registers. On aarch64 with the
ARMv8.3 complex-number extension it uses `FCADD` and `FCMLA`, elsewhere shuffles (x86: `vfmaddsub` for
products, `addsub` for ±i). Whether the extension is present is decided when the package is compiled:
Apple silicon always, Linux by the `fcma` flag in `/proc/cpuinfo` (LLVM's host feature query returns
nothing on macOS).

**Why.** On the M1 this layout with `FCADD`/`FCMLA` was the first variant that beat FFTW; keeping
split real and imaginary parts in registers spilled for `Float64`. The layout is a per-platform choice
behind dispatch on the vector type, so the generator and the plan structure stay shared.

## A heuristic radix choice, no measuring planner

**Decision.** The radix sequence of a length comes from a fixed rule on its prime factorisation
(`src/factor.jl`). There is no measuring planner by default.

**Why.** Brute-force searches over all decompositions put the best and the second-best within 1–3 %,
and the large losses of an estimate-only planner are cache effects the algorithm has to avoid anyway.
A per-machine calibration or an optional measuring mode can follow if the rule leaves too much on the
table.

One such cache effect is part of the rule: single lines on AVX-512 take radix 16 only while the 16
streams of a stage are less than an L1 set stride (4 KB) apart, that is up to 4096 for `Float32` and
2048 for `Float64`. Beyond that the streams share one cache set and radix 8 wins. AMD EPYC 9354, one
line, all buffers at fixed page offsets, time of the radix-16 sequence over the radix-8 sequence:
`Float32` 4096: 0.88, 8192: 1.06, 16384: 1.06; `Float64` 4096: 1.33, 8192: 1.03, 16384: 1.11.

## Rules for lengths are derived on some lengths and checked on others

**Decision (2026-11-04).** The radix choice and the Bluestein length are rules on the prime
factorisation, not tables per length. A rule is derived from one part of the lengths (a brute-force
search over all decompositions, or a cost model) and checked on lengths that were not used for the
derivation, for example by parity of n or by a random split; only the checked result counts as
evidence. A mechanism with a stated cause (a cost estimate) is preferred over a list of lengths. Rules
are per architecture where the search differs. After every change of a rule, the dense sweep is
repeated and its distribution (median, share above 1.2×, 1.5×, 2×, worst lengths) is the criterion,
not single points.

**Why.** The first rules were fitted to a few dozen lengths, mostly powers of two and smooth lengths on
the M1. A dense sweep of every length 2…1024 against FFTW MEASURE (one thread, Zen 2, 2046 cases)
showed 100 `Float32` and 64 `Float64` lengths slower than 1.5×, the worst at 3.4×.

## Lengths with large prime factors: split off the smooth part, Bluestein on the rest

**Decision (2026-11-07).** For n = s·q, with q the part made of primes above `MAX_RADIX_PRIME` (73, at the
time of the decision 31): one Cooley–Tukey
step, a chain of length s, twiddles, and s Bluestein transforms of length q (`Composite`). A leftover q
stays one Bluestein while its convolution is at most 4 MiB; above that, q is split into its prime
powers, recursively. Bluestein convolutions of 8 MiB and more run as a serial four-step algorithm on
x86 (1.3× from 2²⁰ on, Zen 2) but not on aarch64, where the chain is faster up to 2²² (M1 Max). With
threads, `CompositeMT` is used when s is at least the number of threads and is bitwise equal to the
serial kernel; otherwise the whole length is one `BluesteinMT`.

**Why.** The convolutions of length about 2q stay in cache, and the arithmetic equals one Bluestein of
length n. On 45 non-smooth lengths (Zen 2) FFTL moved from ×0.64 to ×1.04 of the fastest other library
with one thread, and from ×0.77 to ×1.16 with 16 threads.

**Not done / not checked.** Rader's algorithm (5 of the 45 sweep lengths qualify). Both thresholds were
measured on Zen 2 only, and the 4 MiB rule is checked on 12 lengths that also took part in its
derivation.

## One chain form: the radix sequence is data

**Decision (2026-11-13).** Every smooth length runs on `RChain`: the radix sequence and the stage
records are data, the codelets come from a fixed set. There is no chain type with the radix sequence
in its type, and nothing is compiled per length. `RChain` is also the only form the trimmed C library
can contain.

**Why.** The compiled chain (`Chain{T,Rs,σ,K}`, used for smooth lengths up to 128 until this decision)
existed because the run-time chain cost 9–20 ns more per call on x86. That cost was not the choice of
the codelet at run time (about 1 ns per stage) but the code around it: GC frames for references loaded
from the chain, the copy of an immutable chain into each caller, bounds checks, and one more call level
per stage. With those removed (see [Implementation](@ref "Implementation")) the compiled chain has no
advantage left. AMD EPYC 9354 (AVX-512), one thread, `plan_fft` and `mul!`, 92 cases (`Float64` and
`Float32`; lines of 2 to 4096, planes of 8 × 8 to 128 × 128, cubes of 8³ to 64³, in-place, backward and
real transforms), 8 rounds in alternating processes, median time over the median time with the compiled
chain:

- geometric mean 0.973; 27 cases below 0.95, 8 above 1.05;
- lines of one stage (2, 4, 16): 0.80–0.95; 8 × 8, 8 × 8 × 8, 7 × 12, 5 × 64: 0.85–0.91;
- the cases above 1.05 are short lines of two or three stages: `Float32` 20, 32, 96 and 100 (1.09–1.10),
  37 rows of 64 (1.13), `rfft` of 64 (1.07 and 1.09), and `Float64` 12 × 7 (1.05). The stage loop runs
  about 50 instructions per stage around the codelet (no mispredicted branches), 2.4 ns per stage
  against direct calls of the two kernels of `Float32` 32.
- The machine was shared during the measurement: over the 8 rounds the slowest time of the compiled
  chain over its fastest is 1.04 in the median over the cases, 1.19 at the 90th percentile and 1.55 at
  most.

Not verified on aarch64.

What the single form removes: 583 compiled chain methods and their generators, the dispatch trees from
an `RChain` of a short length into the compiled chains, and the compilation at the first plan of a
short length with a radix sequence outside the compiled set (a short dimension with a large stride).
The package has 1796 methods instead of 2385, the package image on this machine is 61 MB instead
of 72 MB, and `libfftl.so` (x86-64-v3, `--trim=safe`) is 49.5 MB instead of 55.6 MB.

**Replaced.** The two chain forms (2026-11-09: compiled `Chain` for smooth lengths up to 128, `RChain`
for the rest), and before that the idea of removing `Chain` without changing the stage loop
(2026-11-08) and the separate JIT and run-time plan families (2026-11-02).

## Kernels of a fixed set are plain methods, not `@generated`

**Decision (2026-11-10, fallback removed 2026-11-13).** The stage kernels of the codelet set that `RChain`
dispatches to are plain methods, defined with `@eval` in `src/kernel_methods.jl` from the functions that
build their bodies. One table (`STAGE_KERNELS` in `src/kernels.jl`) lists the kernels and their arguments;
the methods and the dispatch switches of `RChain` are made from it. There are no `@generated` stage
kernels: every combination the chain can select has a plain method, so a fallback would never run.

**Why.** A package image keeps the complete optimised IR of every precompiled `@generated`
specialisation, 130 MB of the 146 MB image. See [Package size and load time](@ref "Package size and load time").

## The first stage of a single line runs at full vector width

**Decision (2026-11-12).** Where the widest vector holds 8 complex values (`Float32` on AVX-512), the
first stage of a single line runs `stage_small!`: vectorised over the butterflies, with the outputs
interleaved in registers into contiguous stores. It exists for the power-of-two radices and belongs to
the fixed set of plain methods.

**Why.** The grouped first stage handles at most 4 complex values per vector. Intel Core i7-1185G7,
one line, `Float32`, lengths 128–16384 with radices of 16s and 8s: 12–26 % faster (the same code
against itself differs by up to 3 %).

**Not done.** The same for later stages with a stride of 2 or 4. It needs a per-lane twiddle table and
one kernel per stride, and only `Float32` lengths whose radices start with 4, 4 have such a stage. On an
AMD EPYC 9354 it gained 6–8 % at 192 and 320 and nothing measurable at other lengths.

## Stage kernels keep twiddles in memory and address streams from few registers

**Decision (2026-11-12).** In the stages vectorised over q, the twiddle multiply reads re(w) and im(w)
from the table (`ccmul_at`; memory operands on x86), and the R input and R output streams are addressed
as base + odd·stride·2^e from one base per 8 streams (`stream_addr`). The base pointer is hidden from
LLVM's loop optimiser (`opaque_ptr`), which otherwise goes back to one pointer per stream.

**Why.** A radix-16 stage has 15 twiddles and 32 streams, more than fit in registers, so its inner loop
reloaded vectors and pointers from the stack. AMD EPYC 9354 (AVX-512), instructions per butterfly:
radix 16 middle stage 284 → 198, radix 8 100 → 79. One line, `plan_fft`, medians of 8 interleaved
rounds: lengths 512–16384 take 0.79–0.94 (`Float32`) and 0.82–0.92 (`Float64`) of the time, lengths
from 32768 on are unchanged (the same code against itself differs by 1–5 %). Not verified on aarch64.

## The prime radices from 17 to 73 share one looped kernel

**Decision (2026-11-13).** A stage of a prime radix from 17 on runs `prime_stage!` (`src/primestage.jl`):
one kernel for all such radices, with the radix and a table of cosines and sines as arguments, compiled
once per vector type. It uses the symmetric form of the odd-prime codelet as loops and computes up to
eight output pairs (four with 16 vector registers) in one pass over the inputs. It covers the stage
kinds that the codelets have separate kernels for: stages vectorised over the lanes, the last stage,
lane blocks, and the grouped first stage. `MAX_RADIX_PRIME` is 73. The radices up to 16 keep their
straight-line codelets. The KernelAbstractions kernels and the generic number types keep their limit
of 31.

**Why.** A straight-line codelet of a prime r has about r²/2 statements with as many different
constants, which do not fit in registers (525 statements for 31), and every radix needed 82 kernel
methods on AVX-512 (kind × sign × vector type). The codelets of 17–31 were therefore compiled at first
use and not with the package (decision of 2026-11-10, replaced by this one). With the loop, a prime
costs a table.

Measurements: AMD EPYC 9354 (AVX-512), one thread, `plan_fft` and `mul!`, rounds in alternating
processes, median time over the median time of the previous code.

- Lengths and shapes with a factor 17–31 against the codelets (40 cases, 4 rounds): geometric mean
  0.82, 36 cases below 0.95, one above 1.05 (`Float64` 100 × 31, 1.06: a single stage in place, which
  copies the inputs of a butterfly first). 17 and 31 alone: 0.63–0.84; 1088 = 17·64: 0.96 and 0.89;
  837 = 27·31 and 841 = 29²: 0.72–0.82; 62 × 62: 0.70–0.74.
- Long lengths (36 cases, 3 rounds; 31·2⁶ to 17·2²⁰, and a factor 17–31 times a prime above 1000):
  geometric mean 0.94, one case above 1.05 (`Float64` 19·2¹⁷, 1.08). 31·2⁶ to 31·2¹⁴: 0.74–0.91;
  17·2²⁰: 0.99 and 1.01.
- The primes 37–73 against Bluestein's algorithm, which they used before (100 cases, 2 rounds; p, 8p,
  64p, 1024p, 16384p): geometric mean 0.29. The prime alone: 0.43–0.79 (`Float64`) and 0.48–0.90
  (`Float32`); as a factor of a longer length: 0.13–0.53.
- 92 general cases without such factors (the set of the decision above, 4 rounds): geometric mean 1.00
  without the two cases with a dimension of 37 (0.10 and 0.16), 3 cases above 1.05 (`Float64` lines of
  32, 48 and 60: 1.07–1.15). In 6 rounds on 20 short lines alone these three were at 1.04, 0.83 and
  1.05 and the geometric mean 1.00; the stage kernels of these lengths compile to the same machine code
  as before.
- Accuracy against a 200-bit reference, `Float64`, 841, 6256 and 6859: 1.4–1.8 ε, with the codelets
  3.5–4.6 ε (the table is built from angles reduced to one turn).

What made the loop as fast as the unrolled sums: several rows per pass, so that the inputs are loaded
once for all of them and the multiply-add chains of the rows run side by side; no buffer for the sums
and differences of the inputs (storing whole registers was the slowest part: 1.3–1.4× per butterfly
with the buffer); the pass over a block of rows out of line and the outputs of a block written by
recursion over the tuples of accumulators (a loop with an index into them, or a closure over them, put
the accumulators on the stack: up to 1.4× for a whole transform).

**The limit.** With the limit at 127 for the measurement, a prime from 79 on is still 1.3–4× faster than
Bluestein's algorithm as a factor of a longer length, but the line of that prime alone is slower from
79 on for `Float32` (1.05, 1.5 at 127) and from 97 on for `Float64` (1.01, 1.15 at 127). 73 is the
largest prime for which every measured case is faster.

**Not done.** The loop for 11 and 13: with one kernel per radix (an earlier form of the loop) it was at
0.98 of the codelets for `Float64` and 1.05–1.06 for `Float32`, and 1.08–1.10 for 7; not measured with
the single kernel. A larger limit for lengths that are not a prime alone. aarch64 is not verified.

**Costs.** The kernel and its tables are in the package image, which the lazy codelets were not: the
image file is 66 MB instead of 59 MB on this machine (68.5 MB before the decision on the chain form).
The C library contains every kernel in either case and shrinks: `libfftl.so` (x86-64-v3, `--trim=safe`)
is 26.8 MB instead of 49.5 MB, its code 8.8 MB instead of 18.6 MB.

## Run-time pass plans for trimmed programs

**Decision (2026-11-10).** A pass plan (`plan_pass`, `plan_r2c_pass`, `plan_c2r_pass`) built with
`_runtime` (`plan_pass_runtime`, `plan_r2c_pass_runtime`, `plan_c2r_pass_runtime`) uses run-time kernels
only: RChain, Bluestein and composite transforms over RChain, the four-step algorithm over RChain. Its
type depends on the element type and the number of dimensions, not on the length. The specialised plans
are unchanged.

**Why.** A library that holds passes of many lengths in one container (as NFFT.jl does for its boxes)
has a container of an abstract or length-dependent type, and `juliac --trim` cannot resolve the calls
through it. The first attempt was a keyword `runtime = true`; the verifier then still checks the
specialised branch, whose kernel types depend on the length, so the separate functions were needed.
A probe library compiled with `juliac --trim=safe` holds run-time pass plans in a vector, applies them,
and matches the direct DFT to 3·10⁻¹⁴ when called from C. Planning on 64 × 64 × 64 views is 1.8 µs for
`d = 1` and 0.9 µs for `d = 2` on the M1 (the specialised plan: 4.7 µs and 2.8 µs). Applying the
specialised and the run-time plans is within noise of each other on the same views.

**Costs.** For real lengths with large prime factors, the run-time kernel is a Bluestein transform,
without the composite split of the specialised plan.

## The interface: FFTW.jl's names, FFTL's own functions, AbstractFFTs as a backend

**Decision (2026-11-06).** FFTL exports FFTW.jl's names (`fft`, `bfft`, `ifft`, `rfft`, `brfft`,
`irfft`, the `plan_` and in-place forms, the `dct` family, and the AbstractFFTs helpers), so `using FFTL`
can take the place of `using FFTW`. `r2r`, the kinds, the flags and `set_num_threads` stay public with
the prefix, as in FFTW.jl. The DFT and DCT functions are FFTL's own, not AbstractFFTs's; the package only
does `import AbstractFFTs`, never `using AbstractFFTs` or `import AbstractFFTs: fft`. Generic code that
calls `AbstractFFTs.fft` reaches FFTL through methods of `plan_fft` and its relatives for
`AbstractArray`s of floating-point numbers (`src/abstractffts.jl`), which are less specific than
FFTW.jl's `StridedArray` methods: with FFTW loaded, the generic functions take FFTW for its arrays and
FFTL for the rest.

**Why, and what was rejected.**
- Exported names that are AbstractFFTs's functions (re-exported bindings) made `FFTL.fft === FFTW.fft`:
  any dependency that loads FFTW would take the standard arrays away from code that wrote `using FFTL`,
  and the GPU extension's methods would have landed next to CUDA.jl's.
- Methods with FFTW.jl's exact signatures: the package loaded last overwrites the other for the whole
  process, and a package depending on both fails to precompile ("Method overwriting is not permitted
  during Module precompilation").
- Methods are global per process, so this is the only arrangement in which a scope with `using FFTL`
  reaches FFTL whatever a dependency loads. Tested: with `B -> FFTL` and `B -> A -> FFTW`, B gets FFTL and
  A gets FFTW; a scope that loads both must qualify the name.

## Scope

**Decision (2026-11-01).** Real transforms (r2c, c2r, all real-to-real kinds including halfcomplex), a
KernelAbstractions GPU path (complex; Metal and CUDA tuned), a trimmed C library (`libfftl`, all CPU
transforms) and, from 2026-11-02, generic float types are part of FFTL.

**Why.** Real passes are needed by NFFT.jl, and the GPU path reuses the generator and the plan structure.
The parts that do not depend on the backend (codelet generation, factorisation, twiddles, pass
structure) stay separate from the CPU execution so that the GPU path can share them.
