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

**Decision (2026-11-07).** For n = s·q, with q the part made of primes above 31: one Cooley–Tukey
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

## Two chain forms: compiled for short smooth lengths, run-time data for the rest

**Decision (2026-11-09).** `kernel` builds the compiled `Chain` (radix sequence in the type) for smooth
lengths n ≤ 128 (`SMALL_N`) and the `RChain` (radix sequence as data, a fixed set of codelets) for
every other smooth length. The choice is made when the plan is built, so a call carries no dispatch.
Both forms share the twiddle tables, the radix rule and the vector-type rules (bitwise equal in 552
checks). `RChain` is also the only form the trimmed C library can contain.

**Why.** A plan of a new size with `Chain` is compilation, not planning: 0.31–0.48 s for plan and first
call against 0.013–0.044 s with the run-time plan (M1, one fresh process per case). But on x86 the
run-time choice of the codelet costs 9–20 ns per call (n = 16, `Float64`: `Chain` 22 ns, `RChain`
39 ns), which is 1.2–1.5× for short lines. With the split, short lengths are at parity or better. Warm execution of the run-time plan against the
compiled one was 1.00–1.08 on the cases measured, with one exception (Bluestein, 100003, `Float32`: 1.42,
cause open).

**Replaced.** The idea of removing `Chain` altogether (2026-11-08) and the earlier decision to keep the
JIT and run-time plan families as separate types (2026-11-02).

## Kernels of a fixed set are plain methods, not `@generated`

**Decision (2026-11-10).** The stage kernels of the codelet set that `RChain` dispatches to, and the
compiled chains of the short lengths, are plain methods defined in `src/kernel_methods.jl` from the
bodies the `@generated` methods build. The `@generated` methods remain as the fallback.

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

## Codelets of the radices 17–31 are compiled on first use

**Decision (2026-11-10).** The codelets of the radices 17, 19, 23, 29 and 31 are not part of the package
image. The first call compiles them (70–100 ms); later calls go through a table of callable objects at a
cost of about 0.02 µs per transform.

**Why.** These radices were 54 % of the statements of the stage kernels and are rarely needed. The image
shrinks from 50 MB to 32 MB. Making 11 and 13 lazy as well would save 4 MB more; it was not done, because
lengths with those factors are common. The list is the preference `lazy_radices`; the trimmed C library
sets it to empty, because `juliac` compiles nothing at run time.

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

**Costs.** The run-time kernels dispatch per stage, which costs a little on short lines (the reason the
specialised plans stay the default). For real lengths with large prime factors, the run-time kernel is a
Bluestein transform, without the composite split of the specialised plan.

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
