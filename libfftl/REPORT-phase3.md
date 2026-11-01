# libfftl phase 3: port to RPlan, trim build, C tests

Task: step 3 of the libfftl plan — merge `origin/main` (the RPlan interface), port the boundary
to `FFTL.RPlan` / `FFTL.execute!`, update `include/fftl.h` per the review, build with
`juliac --trim=safe`, run trim checks and C tests, measure the bundle. Everything is on branch
`libfftl` in this worktree; package `src/` is untouched (the package changes still needed are
in the Requests at the end — request 2 blocks the trimmed build, see STATUS; the untrimmed
library builds and its whole C matrix is green).

Machine: remote03, AMD EPYC 7763 (Zen 3, AVX2, no AVX-512), shared. Julia 1.13.3, JuliaC 0.3.12
(`libfftl/jc`). Runs pinned with `taskset -c 100-107` where threads matter.

## 1. Merge and package tests

`origin/main` @ eba99fb (RPlan 74c6b2f, RChain 67bfdf0, kernel cache 2cce802) merged into
`libfftl` (merge commit bf5163a), then the package quick suite of the worktree:

```
julia -t 2 --project=. -e 'using Pkg; Pkg.test()'
819 passed, 10 broken, 829 total, 4m13.6s
```

The 10 broken are the construction `@test_broken` cases in `test/inference.jl` — by design:
construction goes through `plan_fft`/`FFTLPlan`, not RPlan, so they stay broken (position of
the merge in step 1: the libfftl-relevant tests of the merged package pass).

After request 1 (below) landed on main, `origin/main` @ 09558f8 was merged the same way — no
rebase (merge commit e855818: the rchain fix plus the Threads changes 0044b76/07dcf23 it
carried) — and the quick suite of the merged worktree is unchanged:

```
julia -t 2 --project=. -e 'using Pkg; Pkg.test()'
819 passed, 10 broken, 829 total, 4m13.3s
```

## 2. Boundary port to RPlan (commits a59420b, c8a6796, 6ac978e)

`libfftl/src/libfftl.jl` no longer contains any phase-0 copy or variant table. The entry
points:

- `fftl_plan_dft(rank, dims, trank, region, precision, sign, threads)` validates precision, rank
  >= 1, 0 <= trank <= rank, threads >= 1, sizes >= 1 (via RPlan), and the region: 0-based,
  strictly ascending, each below rank — duplicates and out-of-range indices are errors, which
  the boundary must reject itself (RPlan silently merges duplicates). Signs other than ±1,
  sizes below one: RPlan's own `ArgumentError`s, relayed through `fftl_last_error`.
- Plans live in two per-precision registries, `RPlan{Float64}` / `RPlan{Float32}` behind
  `Union{Nothing,RPlan{T}}`. The handle packs the registry index (low 32 bits) and the
  `FFTL_PRECISION_*` value (1/2, bits 32+); executes and destroys decode monomorphically per
  precision.
- `fftl_plan_execute(handle, out, in)` → `FFTL.execute!(p, py, px)` (in place: `out == in`),
  `fftl_plan_destroy(handle)` clears the registry slot (a second destroy returns -1, invalid
  handles return -1 with a message), `fftl_last_error` / `fftl_version` write snprintf-style.

`include/fftl.h` follows the review, all six points: any rank (no upper limit), any length >= 1
(Bluestein is inside RPlan), documented region errors, the Threading section (below), sketched
sections for pass plans with pruning and r2c/c2r, and the phase-0 `fftl_plan_create` section is
gone. `fftl_version` is new.

`build.sh`: the runtime worker pool is baked at build time via `--jl-option threads=$FFTL_THREADS`
(default 4, `auto` allowed); Julia and JuliaC versions are printed in the build log; single CPU
target `x86-64-v3` (JuliaC.jl#190). `FFTL_TRIM` picks the juliac mode (new): `safe` (default,
the verifier must resolve every statically reachable call — the state the library ships in)
or `off` (plain `--output-lib`, used in section 6 for the C matrix while construction idioms
fail the verifier — request 2). The phase-0 prune list now also drops libLLVM and
libjulia-codegen (section 7).

## 3. Threading: measured, not guessed

Question of the review: how is the embedded runtime's thread count set, and does
`JULIA_NUM_THREADS` reach it? Probes with two tiny `@ccallable` libraries built exactly like
`libfftl` (`nthreads()` entry point, /tmp/libfftl-p3/probe):

| experiment | result |
|---|---|
| `--jl-option threads=2` baked, env unset | `Threads.nthreads() == 2` |
| `--jl-option threads=2` baked, `JULIA_NUM_THREADS=4` in the loading process | 2 — **the env var is ignored** |
| `--jl-option threads=auto` baked, driver pinned to 4 CPUs | 4 |
| `--jl-option threads=auto` baked, driver pinned to 8 CPUs | 8 |
| `--jl-option threads=auto` baked, `JULIA_NUM_THREADS=7` set, 4 CPUs pinned | 4 (affinity, not env) |
| julia CLI check of the same precedence: `JULIA_NUM_THREADS=4 julia -t2` | 2 — an explicit `-t` beats the env var |

Mechanism: JuliaC's generated loader calls `jl_parse_opts` with the baked options in a library
constructor before `jl_init` — the same precedence as an explicit `-t` on the command line.
`JULIA_NUM_THREADS` is parsed inside `jl_parse_opts` too (the string lives in
`libjulia-internal.so`, not in `libjulia.so`), i.e. an un-overridden env var WOULD be honored
— but juliac always bakes `--threads` (`get!(recipe.jl_options, "threads", "1")`), which
overrides it. Consequences, measured and written into the header: the worker pool is a
**build-time** property (`FFTL_THREADS` in build.sh; `auto` = one worker per CPU of the loading
process, counted from its affinity at load time); plan `threads` above the pool is correct but
gives no speedup; the loading thread is one of the workers.

## 4. Trim check after the RPlan port (libfftl/trimcheck/run.sh)

New entries around the run-time plan path the boundary binds (RChain/RPlan build+execute,
construction-only, execution-only, threaded, Bluestein isolation, and codelet isolation).
Same machine, Julia 1.13.3; the Mac baseline (1.13.2) differs by precompile state — both
columns are shown. Attribution: an unresolved edge is reported once, credited to the entry
whose exploration found it first, so per-entry numbers are an upper mapping of a stable total.

| entry | Mac baseline (142) | remote03, RPlan build | after 09558f8 |
|---|---|---|---|
| fftl_trim_1d_f64 / 1d_f32 / 2d_f64 (public path, unchanged) | 2 / 26 / 22 | 26 / 26 / 22 | 26 / 26 / 22 |
| fftl_trim_rfft_f64 / r2r_f64 (public path, out of scope) | 50 / 16 | 36 / 16 | 36 / 16 |
| fftl_trim_plan_f64 (public construction) | 24 | 0 | 0 |
| fftl_trim_exec_f64 / exec_threaded_f64 (public execution) | 2 / 0 | 0 / 0 | 0 / 0 |
| fftl_trim_rchain_f64 (RChain build + run!, 256 = grouped first stage) | — | 0 | 0 |
| fftl_trim_rplan_f64 / rplan2d / rplan_plan / rplan_exec | — | 2 / 0 / 0 / 0 | 2 / 0 / 0 / 0 |
| fftl_trim_rplan_exec_threaded_f64 (threads=4 spawn path) | — | 0 | 0 |
| fftl_trim_rplan_bluestein_f64 (n as argument) / rplan_smooth_plan | — | 0 / 0 | 0 / 0 |
| fftl_trim_bluestein_ctor / bluestein_exec (97 prime) | — | 0 / 0 | 0 / 0 |
| fftl_trim_rplan_bluestein_plan (97 constant) | — | 139 | 59 |
| fftl_trim_first_group_f64 (direct, 2-complex vector) | — | 0 | 0 |
| fftl_trim_first_group1c_f64 (direct, 1-complex vector) | — | 2 | retired with the fix |
| fftl_trim_rstage_group_f64 (rstage_group! from Main) | — | 0 | 0 |
| (unattributed) | — | 1 | 1 |
| **total** | **142** | **270** | **188** |

Reading after the fix: the dead-branch family is gone everywhere, and with every execution-only,
codelet-isolation and RChain entry at 0 the target path's *execution* is fully trim-clean. What
remains of the target path (62 attributed + 1 unattributed = 63 of the 188; the untouched
public path accounts for the other 126) all sits in **plan construction**, unmasked by the fix:
while the dead branches were statically reachable, the verifier stopped along each exploration
path at them and never walked deeper. The `rplan_bluestein_plan` walk now continues into the
construction machinery (the `cached` kernel cache, closure captures, sort-by, `@sync`) and finds
a fixed set of idioms the verifier cannot compile ahead of time, itemized in section 5. The
`fftl_trim_first_group1c_f64` repro is retired with the fix — the generator now *rejects*
one-complex vectors on purpose, so the direct-call entry measures the intended rejection; it is
removed from entry.jl/count.jl (repro stays in the git history and in section 5).

## 5. The trim blockers: dead branches (fixed), construction idioms (open)

**Resolved: the dead 1-complex branches of `rstage_group!`.** Phase 3's first blocker, in this
report's first version (git history): `rstage_group!` emitted `vi == …` branches for every
ladder vector including the 1-complex `Vec{2,T}`, runtime only ever picks 2- or 4-complex
(`ncs[vi] in (2, 4)`), and the grouped-first-stage generator cannot generate codelets for one
complex value (JIT repro: `BoundsError … at index [5]`; the untrimmed `--output-lib` build
aborted in LLVM `Do not know how to split the result of this operator!`). Claude Code fixed it
on main 09558f8 and hardened the generator to reject other widths. After the merge (e855818):
`rstage_group_f64` = 0, `first_group_f64` = 0, the family is gone from every entry — and the
untrimmed `--output-lib` build that this family used to kill at LLVM now completes (section 6).

**Open: plan-construction idioms the trim verifier cannot compile ahead of time.** With the dead
branches gone, the verifier walks the whole construction path for the first time (its walk
stops at the first unresolved edge — the dead branches were masking everything below them).
Plan *execution* is fully clean (every execution-only, codelet and RChain entry is 0; the
boundary contributes nothing). The trimmed library build (merge e855818, both precisions, build
with default `FFTL_THREADS=4`) fails with **120 errors** — a fixed set of root causes, one row
per package location below, attribution by the failing stack's deepest FFTL frame, measured
from the build log (reproduce: `bash libfftl/build.sh`):

| package location (deepest FFTL frame of the failing stack) | # | idiom the verifier rejects |
|---|---|---|
| plan.jl:275/277/279/280 — body of Bluestein's `cached` do-block | 42 | the block's local `b` shares its name with the outer destructure target → `Core.Box`ed capture; its stores, `cptr(b)` inside `run!`, `b ./= M` read `getfield(box)::Any` |
| plan.jl:270 — `(chirp, b) = cached(…)` in `Bluestein{T}` | 8 | `cached` returns `::Any` (untyped `f`), the tuple destructure is `indexed_iterate(::Any, i)` |
| plan.jl:283 — `Bluestein{…}(n, chirp, b, …)` | 4 | inner ctor must convert fields from `::Any` values |
| plan.jl:28/30/31 — `cached` body, `KCACHE::Dict{Any,Any}` | 22 | `f()` with `f::Any` is a dynamic call; `get`/`delete!`/`setindex!` carry `::Any` keys/values (`isequal`, `hashindex`) |
| rchain.jl:20 — `cached(() -> build_rchain(T, …), key)` | 4 | the closure argument through the untyped `f` |
| rplan.jl:61/64 — `max(nwork, …)` in `RPlan{T}(…; threads)` | 8 | `nwork`/`ntwork` are captured by the tbuf comprehension → boxed → `max(::Any, ::Int)` |
| rplan.jl:66/67 — RPlan inner ctor + `[Vector{Complex{T}}(undef, ntwork) …]` | 16 | the comprehension allocates `Array(undef, ::Any)` over the boxed `ntwork`, `collect` leaves a plain `::Vector`, the struct ctor then converts from it |
| threads.jl:75 — `@sync` in `spawn_items` | 2 | Base's `sync_end` calls `wait(r)` on a non-Task item of a `Channel{Any}` → `wait(::Any)` |
| threads.jl:183 — `FourStep{…}` keyword constructor | 4 | the same `cached`/capture pattern as Bluestein |
| factor.jl:124 — `sort(…; by = …, rev = …)` in `heuristic_radices` | 10 | `By(::Function)`, `Lt`, `ReverseOrdering` hide the comparator from the verifier |
| **total** | **120** | |

The boxing mechanism, one error in full (error #2 of the trimmed build; all 120 are in the same
shape tree — the full list is the output of `bash libfftl/build.sh`):

```
unresolved call from statement max(
  getfield(Box()::Core.Box, :contents::Symbol
  )::Any,
  (…)::Int64
)::Any
Stacktrace:
 [1] RPlan{Float32}(sz::Vector{Int64}, region::Vector{Int64}, σ::Int64; threads::Int64)
   @ FFTL src/rplan.jl:61
 [2] plan_dft(prec::Int64, ...) @ Main libfftl/src/libfftl.jl:105
 [3] fftl_plan_dft(rank::Int32, ...) @ Main libfftl/src/libfftl.jl:118
```

`nwork` is captured by the thread-buffer comprehension five lines below (`[Vector{Complex{T}}
(undef, ntwork) for _ in 1:2nt]`), so reads go through a `Core.Box` and arrive as `::Any`. The
same shape explains the Bluestein rows (`b`), and the untyped `f` of `cached(f, key)` makes every
`f()` a dynamic call. None of this is a *runtime* problem: the JIT compiles these paths fine —
the package quick suite, jltest and even the full C matrix against the untrimmed library (which
AOT-compiles everything the tests execute, section 6) are green. The trim verifier, however,
must name the method for every statically reachable call, and an argument or receiver of `::Any`
defeats it. With 280 errors devoured by the dead branches before, and their absence since
09558f8, this masked set is now the whole distance to a trimmed library.

No boundary-side workaround exists: construction runs per `fftl_plan_dft` call and the idioms
live inside the package's plan/KCACHE machinery. Suggestions per family, all in package `src/`,
none semantically load-bearing:

1. `cached(f::F, key) where {F}` — one word; makes `f()` and everything it feeds resolvable.
2. Bluestein's do-block (plan.jl:270-282): a named function with `where {T}` taking
   `(n, σ, M, fwd)`, called through the typed `cached` — kills the `b` box and the `::Any`
   destructure; `FourStep`'s constructor (threads.jl:183) is the same pattern.
3. The tbuf comprehension (rplan.jl:67): a named helper that preallocates
   `Vector{Vector{Complex{T}}}(undef, 2nt)` and fills it — no capture, no box, and `ntwork`
   reads stay `Int64`.
4. `heuristic_radices` (factor.jl:124): sort a `Vector{<:Tuple}` without `by`/`rev` (Base's
   `isless` on tuples AOT-compiles), or a small hand-rolled selection.
5. `spawn_items` (threads.jl:75): collect the tasks in a `Vector{Task}` and `foreach(wait, ts)`
   — `wait(::Task)` resolves; `@sync`'s `sync_end(::Channel{Any})` cannot.

## 6. Functional validation: jltest (no juliac needed), 229 checks

`libfftl/test/jltest.jl` runs the same matrix as `test/test.c` in plain Julia (the
@ccallable entry points are plain methods) against the real RPlan path — this is the
no-build variant that stays useful in CI:

```
julia -t 4 --project=. test/jltest.jl   # from libfftl/
jltest: 229 checks, 0 failed
```

Covered: 1D smooth (1, 2, 256, 1000, 65536, 131072 round trips) and prime 101/9973 (Bluestein),
2D/3D, partial regions ({0,2}: non-adjacent), Bluestein inside higher ranks, lane blocking
((1024,7) region {1}, 2D (33,1024) threads 1 vs 4, 3D (64,64,8) f32 threads=4), in place and
out of place, both precisions, empty region (copy / no-op), error paths (18 invalid-argument
combinations checked for rejection + message), invalid handles, double destroy, 64 plans
alive at once.

jltest also flushed out two boundary bugs that the blocked C build could never have reached
and that are fixed now:

- `codeunits(::String)` is a `CodeUnits`, not a `Vector` — `fftl_version`/`fftl_last_error`
  threw `MethodError` (copy_cstr takes `AbstractVector{UInt8}` now).
- `fftl_plan_destroy` returned 0 for an already-destroyed handle; it clears the slot only if
  it is live and returns -1 otherwise (fftl.h says a used-again handle must be rejected).

`test/test.c` is the same matrix as a real C program against `libfftl.so` (created for a
trimmed build, threads=1 and 4, plus C-specific behavior: NULL region for trank=0, region
NULL, poison checks on out-of-place writes). The `cc` line is in build.sh. It **ran**: the
dead-branch family that used to kill even the untrimmed `--output-lib` build at LLVM is fixed,
so with the new `FFTL_TRIM=off` knob (section 2) build.sh completes the bundle and compiles the
test as-is:

```
FFTL_TRIM=off bash libfftl/build.sh    # 0 errors, build/lib/libfftl.so
taskset -c 100-107 libfftl/test/test
libfftl tests: 378 checks, 0 failed
```

(threads=4 baked as by build.sh default, pinned to 8 CPUs; without the pin: same result.) The C matrix exercises exactly the paths the
trim verifier still rejects — construction through the boundary incl. Bluestein (prime
lengths), threaded `execute!` — which answers the *functional* half of the remaining blocker:
the boundary and its tests are done; only verifiable-ahead-of-time code is missing for the
trimmed image. jltest re-run after the merge: unchanged, 229 checks, 0 failed.

## 7. Sizes

The **untrimmed** bundle — the first `libfftl.so` that exists (see 6) — measured right after
build.sh's phase-0 prune list, still including the two JIT libraries:

```
bundle total                       492 MB   (du -sh build/)
lib/libfftl.so (AOT, untrimmed)    263 MB
lib/julia/ (runtime libraries)    229 MB
  libLLVM.so.20.1jl               120 MB   (JIT only)
  libjulia-codegen.so.1.13.3       84 MB   (JIT only)
  libjulia-internal.so.1.13.3      16 MB
  libblastrampoline.so.5          3.5 MB
  gmp, mpfr, pcre2, unwind, uv, z, zstd, gcc_s, openlibm — together ≈ 6 MB
```

**libLLVM + libjulia-codegen out (request 3 below): measured, and yes.** With both moved out of the
bundle, the full C matrix passes unchanged (378 checks, 0 failed) — the tests execute plan
construction, Bluestein, threaded execution and error paths, and nothing they call reaches
codegen, because juliac AOT-compiles everything the entry graph infers. This is the *untrimmed*
image, the harder case (it could JIT); a trimmed image is stricter by construction. A missing
JIT also fails loudly if some future path would want it (abort at the first compile attempt),
not silently. build.sh now prunes both by default (with that comment), so the canonical
untrimmed bundle is:

```
bundle total                       289 MB
lib/libfftl.so (AOT, untrimmed)    263 MB
lib/julia/                          26 MB
```

For scale, the earlier probe library (/tmp/libfftl-p3/probe/bA: threads=2, same layout,
before any prune ran) measured 571 MB total = lib-image 274 MB + lib/julia 298 MB (of which
libLLVM 120, codegen 84, openblas 35, stdc++ 21, gfortran 11, libjulia-internal 16,
blastrampoline 3.5); build.sh's prune list covers openblas, gfortran, quadmath, gomp, crypto,
ssl, git2, ssh2, curl, nghttp2, stdc++, ssp, gmpxx, dSFMT, atomic — and now LLVM + codegen.

The **trimmed** bundle still cannot be produced: the 120 construction-idiom errors of section 5
abort juliac before anything is written. Once they are fixed, `bash libfftl/build.sh` puts the
trimmed `libfftl.so` next to the 263 MB untrimmed image here.

## 8. Requests to Claude Code

1. **[landed] rstage_group! must not emit branches for 1-complex vectors** — fixed on main
   09558f8 (branch emission in src/rchain.jl, plus the generator hardening that rejects other
   widths in src/kernels.jl), merged here as e855818. Verified: the family is gone from every
   trim entry, and the untrimmed `--output-lib` build it used to kill at LLVM completes. Thank
   you — the turnaround made everything in sections 6 and 7 possible today. The first version
   of this report (git history) carries the full diagnosis with the suggested patch, the JIT
   repro (`BoundsError … at index [5]` from the 10-arg generator, while the 2-complex sibling
   compiles) and the old repro entry, now retired.

2. **[blocking] Make plan construction trim-verifiable** — the 120 errors of section 5 are
   the last distance to a trimmed `libfftl.so`. Five idioms, none load-bearing at runtime
   (the same code runs green in the JIT, the package suite, jltest and the C matrix):

   - `cached(f, key)` (plan.jl:26): `f` is untyped, `f()` is an `::Any` call — and `::Any`
     flows on into `KCACHE`'s `get`/`delete!`/`setindex!` edges. One word fixes the head of
     the family: `cached(f::F, key) where {F}`.
   - Bluestein's `cached` do-block (plan.jl:270-282): the block's `b` is also the outer
     destructure target → `Core.Box`; its stores, `cptr(b)` and `b ./= M` read
     `getfield(box)::Any`, and the `::Any` inherits into the destructure at :270 and the
     struct ctor at :283. Suggestion: a named `bluestein_tables(::Type{T}, …) where {T}`
     called through the typed `cached` — same pattern for `FourStep{T}` (threads.jl:183).
   - The thread-buffer comprehension (rplan.jl:67) boxes the `nwork`/`ntwork` accumulators →
     `max(::Any, ::Int)` at :61/:64, `Array(undef, ::Any)` in the loop, and a plain
     `::Vector` reaching the RPlan ctor at :66. Suggestion: a named
     `thread_buffers(::Type{T}, ntwork, nt) where {T}` that preallocates
     `Vector{Vector{Complex{T}}}(undef, 2nt)` and fills it.
   - `heuristic_radices` (factor.jl:124): `sort(…; by = …, rev = …)` hides the comparator
     behind `By(::Function)`/`Lt(::Function)`. Suggestion: sort a tuple vector without
     `by`/`rev` — Base's `isless` on tuples AOT-compiles.
   - `spawn_items` (threads.jl:75): `@sync` expands to `sync_end(::Channel{Any})`, whose
     non-Task branch is `wait(::Any)`. Suggestion: collect the spawned tasks in a
     `Vector{Task}` and `foreach(wait, ts)` — `wait(::Task)` resolves.

   Reproduce: `bash libfftl/build.sh` (120 errors, full stacks), `bash
   libfftl/trimcheck/run.sh` (per-entry attribution — 62 + 1 unattributed of the 188 total
   are the target path, the other 126 the unchanged public entries). After it lands I rerun
   both, fill §7 with the trimmed sizes and flip STATUS.

3. **[answered by measurement] libLLVM/libjulia-codegen can be left out of the bundle** —
   section 7: with both moved out, the full C matrix passes unchanged on the *untrimmed*
   image (the harder case, it could still JIT); nothing the tests execute reaches codegen
   because juliac AOT-compiles the visible entry graph. build.sh prunes them by default now,
   with the measurement recorded in its comment. The trimmed image stays to be re-measured
   once request 2 lands (trimmed is stricter by construction, so this only gets safer).

4. Non-blocking: with RPlan-based construction inferable, the construction cases in
   `test/inference.jl` can flip from `@test_broken` to `@test` with your merge (they stay
   broken against `plan_fft`, which is what they measure — perhaps add RPlan construction
   cases instead).

## 9. Open questions

- The untrimmed build's `LLVM ERROR: Do not know how to split` is explained after the fact:
  with the dead family removed, the very same `--output-lib` build completes — the abort was
  indeed the partial `(R, C=1)` generation dragged into AOT codegen, as suspected. Closed.
- Trimmed bundle size (blocked on request 2; the untrimmed bundle measures 289 MB, §7).
- The plan-cache (KCACHE) is shared between plans of the same size — a C API to clear it
  (phase-0 `FFTL.clear_cache!()` exists in the package) was not added to the header yet; the
  review's interface stays as it is.

## Conclusion

Everything on my side of the fence is done and committed: both merges with green package
suites, the RPlan boundary, the reviewed header with measured threading semantics, the build
pipeline (now with the `FFTL_TRIM` knob and a 289 MB validated bundle), the full C matrix
green against the real library (378 checks, 0 failed), the Julia-side boundary tests
(229 checks) and the extended trim check with before/after numbers around your fix. What
remains between this and `STATUS: done` is one request, request 2: five small construction
idioms must become trim-verifiable — precisely diagnosed, with counts, per-location
attribution and repros in section 5. STATUS follows the task rules (Requests instead of
touching package src/).

STATUS: blocked (juliac --trim=safe still reports 120 verifier errors, all in package plan construction — untyped-f cached() calls, Core.Boxed closure captures (Bluestein/tbuf), sort-by in heuristic_radices, @sync in spawn_items — see REPORT section 5 and request 2; the boundary builds untrimmed with FFTL_TRIM=off and passes the full C matrix against the real shared library)
