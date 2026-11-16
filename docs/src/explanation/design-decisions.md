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

**Decision (2026-11-04).** The radix choice and the grid length of a resampled transform are rules on the prime
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

## Lengths with large prime factors: split off the smooth part, resample the rest

**Decision (2026-11-07).** For n = s·q, with q the part made of primes above `MAX_RADIX_PRIME` (73, at the
time of the decision 31): one Cooley–Tukey step, a chain of length s, twiddles, and s resampled
transforms of length q (`Composite`): one batched pass from 32 lanes on, lane by lane below
(2026-11-12). A leftover q stays one resampled transform while its grid is at most 4 MiB; above that,
q is split into its prime powers, recursively. Grids (and Bluestein convolutions) on one thread are
transformed by a four-step where `grid_fourstep` says so: with 64-byte vectors where a line of their
length would (see "Long lines: the four-step's rows go through cache buffers"), on other x86 machines
from 8 MiB (1.3× from 2²⁰ on, Zen 2), not on aarch64, where the chain is faster up to 2²² (M1 Max).
With threads, `CompositeMT` is used when s is at least the number of threads and is bitwise equal to
the serial kernel; otherwise the whole length is one `ResampleMT`.

**Why.** The grids of length about 2q stay in cache. On 45 non-smooth lengths (Zen 2) FFTL moved from
×0.64 to ×1.04 of the fastest other library with one thread, and from ×0.77 to ×1.16 with 16 threads
(measured with Bluestein's algorithm on the leftover). Resampling needs one transform of a length of
at least 2q where Bluestein's algorithm needs two: it took 0.51–0.82 of Bluestein's time for primes
from 1009 to 100003 (Tiger Lake, one thread), at an error of 2–5·10⁻¹⁵ in `Float64` (Bluestein about
5·10⁻¹⁶). The batched pass vectorises across the lanes and loses with few of them: against the lanes one
by one it took 1.7–3.8 times as long at s = 2, 1.1–1.6 at s = 7, 0.73–1.04 at s = 16 and 0.56–1.04 at
s = 32 (`Float64`, q = 37, 1009, 10007, EPYC 9354, one thread; `Float32` 2.1–7.3, 1.6–2.9, 0.84–1.31,
0.87–0.96). The switch is on the lane count and not on the number of threads, which keeps serial and
threaded plans bitwise equal.

The grid of a single resampled transform has a margin on each side that holds copies of the other end
(2026-11-12), so every output is interpolated from one contiguous load. Without it the outputs whose
window wraps around the grid ends (7 per `Float64` transform, 3 per `Float32`) took a scalar loop at
about ten times the cost of the others: 250 of 559 ns at n = 37 and 290 of 1104 ns at n = 101. With the
margin, time over the time without it: n = 37: 0.55, 101: 0.75, 127: 0.79, 1009: 0.96, 100003: 0.98,
101³: 0.78; against `main`: 0.81, 0.81, 0.77, 0.64, 0.80, 0.83, and 0.85 for 101³ on 8 threads (1.17
before). `Float64`, EPYC 9354, minima of interleaved runs, the same code against itself within 6 %.
The margin is a whole page: a margin of 128 bytes moves the grid off the page offset of the work
buffers, and the prime length 100003 then took 1.03–1.05 of the time without margins.

Against Bluestein's algorithm with the looped prime kernel in both (2026-11-14; `main` at 540d876, AMD
EPYC 9354, one NUMA node of a shared machine, one process per variant and thread count, 3 rounds in
rotating order, a second copy of `main` as A/A): over 73 one-thread cases (1-D lines up to 10⁷, planes,
3-D, `rfft`, run-time plans, composite dimensions) the geometric mean of the time ratio is 0.74, 55
cases faster beyond the A/A band and none slower; over 30–31 cases at 2, 4, 8, 16 and 32 threads 0.83,
0.66, 0.68, 0.68 and 0.66, none slower beyond the band (A/A median deviation 5–23 %, largest at 2, 4
and 32 threads). The seven cases with a median above 1.05 were measured again in 7 rounds: 0.60–1.01.
1009·2¹⁶ (`Float64`, 16 threads) took 165 ms instead of 552–555 ms: on `main` the second step, lane
by lane, took 445 of 554 ms, almost all of it in writing each output 2¹⁶ values from the last (107 ms
with the same outputs written contiguously); the batched pass writes runs of consecutive outputs. Error against FFTW in
`Float64` precision, largest over the cases: `Float64` 3.3·10⁻¹⁵ (Bluestein 1.0·10⁻¹⁵), `Float32`
1.8·10⁻⁷ (Bluestein 3.1·10⁻⁷).

**Replaced.** Bluestein's algorithm on the leftover and `BluesteinMT` for a whole line on threads
(2026-11-07). Generic number types and the GPU passes keep Bluestein's algorithm.

**Exception (2026-11-16): `Float64` on aarch64 keeps Bluestein's algorithm for single lines**
(`bluestein_lines`): a whole line of a non-smooth length, a composite leftover lane by lane (fewer
than 32 lanes), and the run-time plans. With two `Float64` values per NEON register the 16-tap
interpolation costs more than Bluestein's second transform. Apple M1 Max, one thread, resampled
against Bluestein (`main` at 540d876): `Float64` 1.18–1.53 for 10007, `rfft` 20014, 100003, 101²
and 127², `Float32` 0.60–0.84. The batched leftover pass (`RsBatch`, from 32 lanes) and the threaded
kernels (`ResampleMT`, `CompositeMT` with the batched pass) resample in both precisions; they were
faster in `Float64` as well (64·1009 one thread 0.80; 1000003 on 8 threads 0.72). With the exception,
against `main` at 1ae621e (M1 Max, minimum over 5 blocks, 2 rounds in alternating processes, a
loaded machine): `Float64` one thread 0.96–1.02 for the single lines above, 0.79 for 64·1009 and 0.85
for 64·79; `Float32` one thread 0.57–0.81; smooth lengths 0.97–1.01. On x86 nothing changes.

**Not done / not checked.** Rader's algorithm (5 of the 45 sweep lengths qualify). Both thresholds were
measured on Zen 2 only, and the 4 MiB rule is checked on 12 lengths that also took part in its
derivation. The 32-lane rule was measured on one AVX-512 machine only, not on AVX2 or aarch64, and the grid
margin on x86 only (512-bit and 256-bit vectors), not on aarch64.

## Plan tables of long lines from a few exact values

**Decision (2026-11-14).** Two tables that a plan builds over the indices of a long line no longer
take one transcendental call per entry:

- The deconvolution factors of a resampled transform (`resample_corr`) are a sum of 26 (`Float64`) or
  18 (`Float32`) cosines over the quadrature nodes, each sampled at consecutive indices t. With
  t = t0 + k (t0 a multiple of 64), every cosine is one product of two correctly rounded values: one
  `sincospi` per node and block of 64, and a table over k shared by all blocks. The node weights
  a_q = w_q ψ(z_q) are computed once, not per index, and the factors are even in t, so half of them
  are computed.
- The factor tables of a long four-step (above `TUNING[].factor_twiddles`) take their n1 + n2 roots
  from `unit_root`, which applies the symmetries of `unit_roots` to one index. The tables are bitwise
  the same as before, without the table of all n roots they were taken from (2 GiB for 2²⁷).

**Why.** Planning a prime length near 2²⁶ took 32 s, 30 of them in the deconvolution factors (67·10⁶
indices, 26 `exp`, `sqrt` and `cospi` each). AMD EPYC 9354, 16 threads, a fresh plan with the cache
cleared, 2 rounds in alternating processes, before → after:

| length | kernel | plan |
|:--|:--|:--|
| 2²⁶ − 5 | `ResampleMT` | 32.0–32.8 s → 0.70 s |
| 2²⁷ | `FourStep` | 1.65–1.69 s → 0.003 s |
| 2²⁶ | `FourStep` | 0.81 s → 0.002 s |
| 100003 | `ResampleMT` | 0.041 s → 0.006 s |

`resample_corr` alone for 2²⁶ − 5 on one thread: 30.2 s → 0.68 s (`Float32` 0.43 s). Against 256-bit
arithmetic (the same quadrature rule; n from 2 to 2²⁶ − 5, both ends of the index range and random
indices), the product form has the error of one `cospi` per term: median 2.4–2.7 ulp, at most 8.9 ulp
in `Float64` for both forms; in `Float32` the factors are bitwise the same for n up to 10⁶. A rotation
recurrence (one complex multiply per step) is as fast but its error grows along the block: 1.2·10⁻¹³
after 512 steps. Transform times are unchanged (the four-step's output is bitwise the same).

**Not done.** The twiddles of the composite kernel (`Composite.tw`, `CompositeMT.tw`) are still a table
of all n roots: 1 GiB and 1.4 s of planning for 1009·2¹⁶. The kernel reads it at run time, so factor
tables there change the kernel.

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
LLVM's loop optimiser (`opaque_ptr`), which otherwise goes back to one pointer per stream. The base is
formed in the integer domain (`inttoptr` of the address), which scalar evolution does not analyse; the
same stage kernels as with `llvm.launder.invariant.group` (inner loops identical up to register names
and, in three `Float32` range kernels, instruction order; AMD EPYC 9354, Julia 1.13). The scaled
addresses stay a typed `getelementptr` in `llvmcall`: Julia's pointer arithmetic is a byte GEP of a
separate shift, which LLVM hoists and merges into an invalid x86 scale, and the middle stages of
radix 8 and 16 then spill (radix 16, `Float64`: 196 → 224–232 instructions per inner loop). On Julia 1.10
(LLVM 15, typed pointers, a `Ptr` passed to `llvmcall` as an integer) `scaled_ptr` is the same GEP
written with typed pointers between `inttoptr` and `ptrtoint`; LLVM folds the conversions into the
address of the load. The kernels compiled by 1.10 are not the same as those of 1.13 (some are
unrolled, and some range and middle stages of radix 5, 7 and 16 reload values from the stack);
1.10 is supported for correctness, its speed is not tuned.

**Why.** A radix-16 stage has 15 twiddles and 32 streams, more than fit in registers, so its inner loop
reloaded vectors and pointers from the stack. AMD EPYC 9354 (AVX-512), instructions per butterfly:
radix 16 middle stage 284 → 198, radix 8 100 → 79. One line, `plan_fft`, medians of 8 interleaved
rounds: lengths 512–16384 take 0.79–0.94 (`Float32`) and 0.82–0.92 (`Float64`) of the time, lengths
from 32768 on are unchanged (the same code against itself differs by 1–5 %). Not verified on aarch64.

**Not done.** Radix 16 as two radix-4 passes (the first into the output slots, the second in place on
them, blocked so that a block stays in L1), which needs no addressing tricks: its inner loops do not
spill, but the extra pass through L1 costs what radix 16 saves over radix 8. AMD EPYC 9354, one line,
medians of 5–6 interleaved rounds, against the one-pass radix 16: `Float32` 256, 2048, 4096: 1.24,
1.24, 1.31; `Float64` 256, 2048: 1.17, 1.05 (same code against itself: 0.99–1.02), and slower than
the best radix-4/8 chain at each length. With 256-bit vectors and 16 registers (`-C haswell`,
`simd_bytes = 32`) 1.05–1.08 against the one-pass radix 16, and both lose to radix-4/8 chains by
7–20 %.

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

Against resampling, which replaced Bluestein's algorithm (2026-11-14, EPYC 9354, one thread, the kernels
called directly, median of 3 rounds): as a factor of 64p the looped kernel takes 0.29–0.44 of the
composite kernel's time for every prime p from 37 to 73, in both precisions. A line of the prime alone
in `Float64` takes 0.59–0.97 of the resampled time (0.97 at 73). In `Float32` resampling is faster
from 47 on: the looped kernel takes 1.10–1.96 times as long (1.96 at 73), at 37–43 they are even
(0.91–1.08). The looped kernel takes the same time in both precisions, the resampled transform
0.49–0.64 of its `Float64` time in `Float32`. The limit stays at 73; a rule for `Float32` lines of a prime alone is not
done.

**Not done.** The loop for 11 and 13: with one kernel per radix (an earlier form of the loop) it was at
0.98 of the codelets for `Float64` and 1.05–1.06 for `Float32`, and 1.08–1.10 for 7; not measured with
the single kernel. A larger limit for lengths that are not a prime alone. aarch64 is not verified.

**Costs.** The kernel and its tables are in the package image, which the lazy codelets were not: the
image file is 66 MB instead of 59 MB on this machine (68.5 MB before the decision on the chain form).
The C library contains every kernel in either case and shrinks: `libfftl.so` (x86-64-v3, `--trim=safe`)
is 26.8 MB instead of 49.5 MB, its code 8.8 MB instead of 18.6 MB.

## Long lines: the four-step's rows go through cache buffers

**Decision (2026-11-14).** The four-step (`FourStep`, `src/threads.jl`) works on blocks of columns
that pass through a buffer in cache:

- Step 1 copies a block's rows into the buffer and transforms them there; no butterfly stage reads the
  input with its row stride n1.
- The twiddle multiplication and the transposition into the intermediate array A are done in
  registers, C × C complex values at a time (`ctranspose`, C values per vector), and A is written
  with non-temporal stores where each store is a whole aligned line (64-byte vectors; the rows of A
  are padded by 64 bytes).
- Step 2 writes its output through the buffer with non-temporal stores from
  `TUNING[].fourstep_nt_bytes` (8 MiB) of output, below that its last stage writes the output.
- A block's rows are at least 128 bytes on several threads (`TUNING[].fourstep_row_bytes`; `Float32`
  takes 16 columns, `Float64` 8) and 256 bytes on one thread.
- On one thread a single line takes the four-step from 16 times the last-level cache of a core,
  in `Float32` for powers of two only (`serial_fourstep`); the convolution grids of Bluestein's
  algorithm follow the same rule.

**Why.** AMD EPYC 9354 (32 MiB L3 per four cores), measurements in one process with the variants
alternating unless noted:

- A large power of two was the slow case: 2²⁶ `Float64` took 1.6–1.8 times as long per element as
  10⁸ (16 threads and one thread), in both steps, while its chains in cache cost about the same. The chains' first stage read the
  input with a stride of 2¹³ values. Without the copy, 2²⁶ took 1.41 times as long (16 threads).
- Writing A cost about as much as the chains (35–48 % of step 1 on one thread), one complex value
  per instruction. In registers: 1.7–2.2 times faster (`Float64`), 3–3.6 (`Float32`) alone. In the
  whole transform with ordinary stores it was slower for `Float32` at 2²² and 3·2²⁰ (16 threads,
  step 1 0.36 → 0.50 ns per element): step 2 reads A on other threads, so the lines of A are still
  held by other cores when step 1 writes them, and every store first takes ownership. With the
  lines read by other threads first, the vectorised writes took 7 times as long as on untouched
  lines. Non-temporal whole-line stores need no ownership: time with them over the time without,
  16 threads, `Float32` 2¹³ 0.90, 2¹⁶ 0.83, 2²² 0.67, 6¹⁰ 0.89; `Float64` 2¹⁶ 0.81, 10⁶ 0.54,
  2²² 0.73, 2²⁶ 0.96; 4 threads 0.56–0.85.
- Step 2 without non-temporal output: 1.05–1.32 times as long in `Float64` from 16 MB, 1.01–1.14 in
  `Float32` from 8 MiB; with a threshold of 4 MiB instead of 8, `Float64` 2¹⁸ took 0.87 of the time
  (not adopted: the output then leaves the cache, which a benchmark of the transform alone does
  not charge). Writing the output through the buffer with ordinary stores was never better
  than the last stage writing it.
- Rows of 128 bytes for `Float32` on 16 threads: 0.86–0.88 at 2²⁴ and 6¹⁰, 0.99–1.04 elsewhere.
  256-byte rows: 1.06–1.19 times as long for `Float64` at 2²⁶ and 6¹⁰. On one thread 256-byte rows
  took 2²⁶ `Float64` from 14.1 to 11.5 ns per element.
- Copying the rows below 16 MiB costs 1–5 % (`Float32` 10⁶ 0.95 without it, `Float64` 2¹⁶ and 2¹⁸
  0.96), within the noise of transforms of a millisecond; the copy is unconditional.
- One thread, four-step time over chain time: `Float64` 2¹⁹ 1.88, 2²² 1.08, 2²⁴ 0.96, 3·2²³ 0.81,
  2²⁵ 0.80, 2²⁶ 0.64, 6¹⁰ 0.98, 10⁸ 0.95, three lengths near 2²⁵ with power-of-two parts 2¹¹–2¹⁷
  0.88–0.97; `Float32` 2²⁰ 1.61, 2²⁵ 1.09, 2²⁶ 0.94, 2²⁷ 0.76, 10⁸ 1.17, three lengths near 2²⁶ with
  power-of-two parts 2¹²–2¹⁸ 1.11–1.16. Beyond the cache a chain makes one pass over memory per
  stage (10–13 at these lengths) at 34 GB/s on one core, 22 GB/s for 2²⁶ (strides); the four-step's
  work in cache costs about as much as 13 such passes. From 512 MiB the rule takes no measured case
  where the four-step is slower; `Float64` lengths of 384–506 MiB (0.81–0.97) stay on the chain.
- Against `main` (2026-11-14, `plan_fft` and `mul!`, one process per variant and thread count with
  as many threads in Julia's pool, 2 rounds in rotating order, A/A within 2.9 % at 4 threads and
  0.8 % at 16): 22 cases each (2¹⁶–2²⁶, 10⁶, 6¹⁰, 3·2²⁰, the prime 1000003, `rfft` 2²², both
  precisions), none slower; time ratio 0.46–0.81 at 4 threads (2²⁶ 0.55–0.57), 0.58–0.85 at 16
  (2²⁶ 0.58–0.59); geometric mean of the ratio normalised by FFTW in the same process 0.62 and 0.69.

**Not done / not checked.** The radix sequence inside the four-step: radix 8 or 16 for the chains of
a block (the rule given the block's stride, or the single-line rule) was not consistently faster,
mostly 1.00–1.24 times the current time; chains of 2048 with radix 8 took 0.89–0.90 (8 threads).
The lane-block kernels themselves are slower than single lines of the same length (1.1–1.9 times
per element in cache for powers of two), which is the larger gain left. A three-factor (nested) four-step for lengths whose blocks outgrow L2. The
multiple of the cache for the one-thread rule and every threshold here are measured on this machine
only; aarch64 and 32-byte vectors are not measured (there A is written with ordinary stores). 32
threads could not be measured on the shared machine (one thread of the pool waits for a core).

## Run-time pass plans for trimmed programs

**Decision (2026-11-10).** A pass plan (`plan_pass`, `plan_r2c_pass`, `plan_c2r_pass`) built with
`_runtime` (`plan_pass_runtime`, `plan_r2c_pass_runtime`, `plan_c2r_pass_runtime`) uses run-time kernels
only: RChain, resampled and composite transforms over RChain, the four-step algorithm over RChain. Its
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

**Costs.** For real lengths with large prime factors, the run-time kernel is one resampled transform,
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

## GPU: large prime factors in multi-dimensional transforms

**Decision (2026-11-14).** In the KernelAbstractions extension:

- **Ragged lanes.** A one-pass kernel takes the lines of a dimension in memory order with any number
  of lines per workgroup; the lanes past the last line repeat it and store nothing new. It is used
  where the old rule (the lines per workgroup divide the line count) leaves fewer than four lines per
  workgroup, and in the fused convolution kernels.
- **Rader's algorithm** for a prime length ``p`` whose ``p - 1`` has prime factors up to 13 and fits
  one pass: the cyclic convolution of length ``p - 1`` in one kernel, with the same structure as the
  fused Bluestein kernel (permuted loads and stores from a table, ``X_0`` from the forward result).
  A single line of a prime above 512 keeps Bluestein's algorithm.
- **Balanced radices** in the fused Bluestein and Rader kernels: the fewest stages of radices up to 16,
  the smallest largest radix among them (1008 as ``[12, 12, 7]``, 100 as ``[10, 10]``). Ordinary passes
  keep the chain of small radices.
- **Odd real transforms by a DFT of ``n`` per two lines.** r2c, c2r, DCT-II and DST-II of odd length
  pack two real lines into one complex line (Makhoul's permutation for DCT-II/DST-II) instead of a
  complex DFT of ``2n`` (DCT-II/DST-II) or of each real line (r2c/c2r). With a fused line plan the
  packing and the separation of the pair run inside that kernel.
- On Apple GPUs the Bluestein length is the next power of two when it fits one pass.

**Why.** On a GPU, multi-dimensional transforms with a dimension of 101 or 1009 were 5–20× slower than
cuFFT or VkFFT (harness dev runs on an NVIDIA RTX A6000 and an Apple M1 Max). The causes, measured one
by one on the M1 Max:

- A prime line count (101, 101²) left one line per workgroup, 14 threads for a pass of 224: passes of
  128 and 224 along a stride of 101 ran at 145 GB/s, along a stride of 128 at 470 GB/s. With ragged
  lanes the stride of 101 runs at 470 GB/s.
- Bluestein's convolution needs a length of at least ``2p - 1`` (224 or 256 for 101), Rader's ``p - 1``
  (100): 101³ dimension 2, 89 µs against 141 µs.
- In the fused kernels the split of the length into radices decides the speed: one thread per
  butterfly of the largest radix, so a small radix next to 16 takes many rounds per thread (1009²
  dimension 1: 1008 as ``[12, 12, 7]`` 141 µs, as ``[8, 2, 9, 7]`` 249 µs; 101³: 224 as ``[16, 2, 7]``
  335 µs, 256 as ``[16, 16]`` 142 µs). A single line runs in one workgroup, where more threads win, so
  ordinary passes keep many small stages (r2c 3000, a line of 1500: 300 threads 41 µs, 100 threads 58 µs).
- Code size: loops over long lines unrolled at generation (1009 elements, 84 threads) made the fused
  kernel slower than separate kernels; as loops in the kernel the packed kernel is faster.

Measurements (`Float32` unless noted; µs; harness timing, back-to-back transforms):

| case | before | after | fastest other library |
|---|---|---|---|
| Apple M1 Max, c2c 101³ | 1333 | 222 | 228 (VkFFT) |
| Apple M1 Max, REDFT10 101³ | 3405 | 257 | 210 (VkFFT) |
| Apple M1 Max, REDFT10 1009² | 1592 | 279 | 373 (VkFFT) |
| NVIDIA RTX A6000, c2c 101³ | 372 | 85 | 80.5 |
| NVIDIA RTX A6000, c2c 101³ `Float64` | 3904 | 868 | 935 |
| NVIDIA RTX A6000, REDFT10 101³ | 1035 | 61 | 49 |
| NVIDIA RTX A6000, REDFT10 1009² | 548 | 60 | 61.5 |

Harness dev runs on the M1 Max GPU (341 cases, three rounds, the code before and after measured on the
same day): FFTL's speed-up over the fastest of MPSGraph and VkFFT, geometric mean, went from 0.76 to
1.20 on the non-smooth sizes, from 1.08 to 1.20 on the held-out sizes (not used during this work), and
stayed at 1.52/1.54 on the tuned sizes. An A/B over 245 tuned and non-smooth cases against the previous
code, three rounds: geometric mean of the time ratio 0.95; cases above 1.1 were within the noise of the
previous code in four further rounds.

**Provenance.** Rader's algorithm (1968) inside the radix kernels instead of Bluestein, and the limit of
13 for the prime factors of ``p - 1``, follow the VkFFT paper (D. Tolmachev, IEEE Access 11, 2023,
§IV.A); the limit was confirmed by measurement (139, with ``138 = 2·3·23``, is 1.7× slower with Rader).
No VkFFT code was read. Ragged lanes, the packed odd real transforms and the radix split were found by
measurement before the paper was read.

**Not done.** Rader for primes whose ``p - 1`` has a factor above 13 (REDFT10 139×117×156 is at 0.92 of the
fastest library on the M1 and 2.3× slower than it on the A6000), and for a ``p - 1`` beyond one pass. The
radix split of ordinary passes on many lines. The length rule for Bluestein on CUDA (224 kept there).

## GPU: small real transforms in one launch

**Decision (2026-11-15).** A real transform along the first dimension (r2c, c2r, DCT-II, DST-II)
runs in a single kernel launch wherever its complex DFT is a single kernel:

- **Odd lengths:** a plain one-pass kernel packs two real lines into one complex line, as the fused
  kernels already did.
- **Even lengths whose half is a Rader or one-pass Bluestein length:** that fused kernel reads the
  real line as a complex line of half the length (Makhoul's permutation for DCT-II and DST-II) and
  separates the result in its last loop.
- **Less host work per call:** no tuple slices and a function barrier in the r2r `mul!`, and
  user arrays reinterpreted on the device instead of on the host.

**Why.** Small GPU transforms are bound by the host's launch rate. On an NVIDIA RTX A6000 with an
AMD EPYC 7763 host, a raw CUDA launch costs about 2.3 µs of host time and a KernelAbstractions launch
about 3.8 µs, while the kernels of these transforms run for 1.5–3 µs. FFTL ran a pack kernel, the
DFT and a post kernel, three launches where cuFFT and VkFFT use one: c2r 21 `Float32` took 14 µs
against 2.3 µs.

Measurements on the NVIDIA RTX A6000, µs. Interleaved, pinned to one core, 50 ms warm-up, blocking
wait, median of 7 samples of 100 back-to-back transforms; "before" is the previous `main`:

| case | before | after |
|---|---|---|
| r2c 27 `Float32` | 14.5 | 3.48 |
| c2r 225 `Float32` | 14.7 | 3.41 |
| REDFT10 21 `Float32` | 17.8 | 3.60 |
| REDFT10 74 `Float32` | 19.7 | 5.65 |
| r2c 32 `Float32` | 4.69 | 3.62 |
| REDFT10 21 `Float64` | 17.9 | 4.75 |
| controls: c2c 16, 128; REDFT10 64; r2c 37 | | 0.97–1.01 × before |

On the Apple M1 Max GPU, 25 cases: geometric mean 0.79 of the previous time (c2r 21 21.0 → 8.1 µs),
controls 0.97–1.00.

**Not done.** About 1 µs per call above cuFFT and VkFFT remains on the single launch. Real
transforms along other dimensions, and multi-pass lines, keep their pack and post kernels.
Multi-dimensional small cases are bound by the device, not by launches: REDFT10 37×37 runs two Rader
passes, 11.7 µs against 7.0 µs.

## Scope

**Decision (2026-11-01).** Real transforms (r2c, c2r, all real-to-real kinds including halfcomplex), a
KernelAbstractions GPU path (complex; Metal and CUDA tuned), a trimmed C library (`libfftl`, all CPU
transforms) and, from 2026-11-02, generic float types are part of FFTL.

**Why.** Real passes are needed by NFFT.jl, and the GPU path reuses the generator and the plan structure.
The parts that do not depend on the backend (codelet generation, factorisation, twiddles, pass
structure) stay separate from the CPU execution so that the GPU path can share them.
