# libfftl phase 3: port to RPlan, trim build, C tests

Task: step 3 of the libfftl plan — merge `origin/main` (the RPlan interface), port the boundary
to `FFTL.RPlan` / `FFTL.execute!`, update `include/fftl.h` per the review, build with
`juliac --trim=safe`, run trim checks and C tests, measure the bundle. Everything is on branch
`libfftl` in this worktree; package `src/` is untouched (the one required package change is in
the Requests at the end, which also blocks the build — see STATUS).

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
target `x86-64-v3` (JuliaC.jl#190); the phase-0 library prune list is unchanged.

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

| entry | Mac baseline (142) | remote03, RPlan build |
|---|---|---|
| fftl_trim_1d_f64 / 1d_f32 / 2d_f64 (public path, unchanged) | 2 / 26 / 22 | 26 / 26 / 22 |
| fftl_trim_rfft_f64 / r2r_f64 (public path, out of scope) | 50 / 16 | 36 / 16 |
| fftl_trim_plan_f64 (public construction) | 24 | 0 |
| fftl_trim_exec_f64 / exec_threaded_f64 (public execution) | 2 / 0 | 0 / 0 |
| fftl_trim_rchain_f64 (RChain build + run!, 256 = grouped first stage) | — | 0 |
| fftl_trim_rplan_f64 / rplan2d / rplan_plan / rplan_exec | — | 2 / 0 / 0 / 0 |
| fftl_trim_rplan_exec_threaded_f64 (threads=4 spawn path) | — | 0 |
| fftl_trim_rplan_bluestein_f64 (n as argument) / rplan_smooth_plan | — | 0 / 0 |
| fftl_trim_bluestein_ctor / bluestein_exec (97 prime) | — | 0 / 0 |
| fftl_trim_rplan_bluestein_plan (97 constant) | — | 139 |
| fftl_trim_first_group_f64 (direct, 2-complex vector) | — | 0 |
| fftl_trim_first_group1c_f64 (direct, 1-complex vector) | — | 2 |
| fftl_trim_rstage_group_f64 (rstage_group! from Main) | — | 0 |
| (unattributed) | — | 1 |
| **total** | **142** | **270** |

Reading: through RPlan/RChain, construction and execution resolve — every target-path entry is
0 **except** one family, itemized below. It also appears in the *public* path now (e.g. inside
`fftl_trim_2d_f64`): the same dead branches are statically reachable from any `run!`.

## 5. The one blocker: dead 1-complex branches of rstage_group!

All trim errors outside the untouched public path share the same signature, e.g.

```
unresolved invoke from statement stage_first_group!(py::Ptr{Float64}, …, m::Int64, 0::Int64,
…, Val{2}()::Val{2}, Val{-1}()::Val{-1}, Vec{2, Float64}::Type{Vec{2, Float64}},
Val{0}()::Val{0})::Any
```

The vector type `SIMD.Vec{2, Float64}` carries **one** complex value. Facts, all measured:

1. `FFTL.rstage_group!` (src/rchain.jl:54-67) emits its `vi == …` dispatch branches for
   **every** entry of `vec_ladder(T)` — that ladder ends in a 1-complex vector (`Vec{2,T}`).
   Runtime never picks them: `run!` calls the grouped first stage only with vectors of 2 or 4
   complex values (`ncs[vi] in (2, 4)`, rchain.jl:96-101). In JIT mode nothing ever compiles
   the dead branches — that is why the package tests pass.
2. The generator of the grouped first stage (src/kernels.jl:76-116) only supports
   `ncomplex(VT) in (2, 4)`: for one complex value it takes the `C == 2 ? pick(outs[k+1],
   outs[k+2], l) : shufflevector(pick(outs[k+1], outs[k+2], l), pick(outs[k+3], outs[k+4], l),
   …)` branch with only `k + 1 <= R` guaranteed, and indexes `outs` out of bounds. JIT repro:

   ```
   julia --project=libfftl/trimcheck -e 'using FFTL; import SIMD
     … stage_first_group!(…, 2, 0, 2, Val(4), Val(-1), SIMD.Vec{2,Float64}, Val(0))'
   → BoundsError: attempt to access 4-element Vector{Any} at index [5]
   (the 2-complex sibling, Vec{4,Float64}, compiles and runs)
   ```

3. AOT builds cannot skip statically reachable branches. `juliac --trim=safe` therefore fails
   with 280 verifier errors — all this family, none other — and produces no library (build/
   is not written). The committed repro entry `fftl_trim_first_group1c_f64` reproduces it from
   Main with 2 errors per (radix, sign) signature.
4. Even the **untrimmed** `--output-lib` build dies on this code on AVX2 — with exactly the
   dead family forced into AOT codegen it aborts:

   ```
   LLVM ERROR: Do not know how to split the result of this operator!
   signal 6: Aborted — ERROR: Failed to compile src/libfftl.jl
   ```

   (juliac implicitly AOT-compiles all @ccallable entry points and their inferred call graph
   for `--output-lib`; there is no flag that disables it — `add_ccallables` is force-set.)

So the phase-2 milestone "0 errors through RPlan" is reached **except** for this package-side
dead code; the boundary itself contributes 0 errors. No boundary-side change can avoid the
verifier walking those branches: every `RChain.run!` statically contains the
`single && i == 1 && grp != 0 && ncs[vi] == 1` arm, and building a CPU target without a
1-complex ladder entry is impossible (the ladder always ends at one complex value; and this
host has no AVX-512, so `x86-64-v4` is not a way out either).

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
NULL, poison checks on out-of-place writes). The `cc` line is in build.sh. It cannot run
until the package fix — then:

```
bash libfftl/build.sh     # target: 0 verifier errors, build/lib/libfftl.so
libfftl/test/test          # runs the matrix
```

## 7. Sizes

The trimmed bundle could not be produced (see 5 and STATUS) — no number for it exists yet.
What was measured on this machine, for scale: a **probe** library built exactly like libfftl
but without trim (same bundle layout, threads=2) measures

```
bundle total                      571 MB   (du -sh, /tmp/libfftl-p3/probe/bA)
lib/libprobe.so (untrimmed image) 274 MB
lib/julia/ (runtime libraries)    298 MB
  libLLVM.so.20.1jl               120 MB   (JIT only — a trimmed image does not compile)
  libjulia-codegen.so.1.13.3       84 MB   (JIT only)
  libopenblas64_.so.0.3            35 MB   (removed by build.sh, as in phase 0)
  libblastrampoline.so.5           3.5 MB  (removed by build.sh)
  libstdc++.so.6.0.34              21 MB    (removed by build.sh)
  libgfortran.so.5.0.0             11 MB    (removed by build.sh)
  libjulia-internal.so.1.13.3     16 MB
```

The phase-0 sizes on the Mac are not in the journal, so there is no before/after column to
fill; after the package fix the real numbers go here (build.sh also deletes the
networking/git/C++ libraries as in phase 0 — the list is in build.sh).

## 8. Requests to Claude Code

1. **rstage_group! must not emit branches for 1-complex vectors** (fixes trim and C tests).
   In src/rchain.jl, the `byv` map over the ladder emits `stage_first_group!` calls for every
   `vi`; the grouped first stage supports only 2 or 4 complex values (generator indexes
   `outs[k+3..k+4]` guarded by `k + C <= R`, valid only for C ∈ {2, 4}), and runtime only ever
   calls it with `grp` from `ncs[vi] in (2, 4)`. Suggested patch:

   ```julia
   byv = map(enumerate(ladder)) do (vi, V)
     name != :rstage_group! || ncomplex(V) in (2, 4) || return :()
     byr = [:(R == $r && return $(call(r, σ, V))) for r in RADICES]
     :(if vi == $vi; $(byr...); end)
   end
   ```

   (the runtime `throw(ArgumentError(…))` fallback already covers any other `vi`, which is
   unreachable anyway). Optional hardening: assert `C in (2, 4)` in the generator (kernels.jl)
   so future misuse fails loudly at generation time rather than as a verifier error or an
   LLVM abort. Repros: section 5 (JIT one-liner with the exact BoundsError, the committed
   entry `fftl_trim_first_group1c_f64` → 2 verifier errors, and `build.sh` → 280).

2. After it lands: rerun `build.sh` (this report updates with the 0-error build, the test.c
   results and the trimmed bundle sizes) and consider whether the trimmed bundle still needs
   `libLLVM`/`libjulia-codegen` (JIT-only, 204 MB of the probe bundle — the trimmed library
   never compiles; phase 0 kept them).

3. Small, non-blocking: with RPlan-based construction inferable, the construction cases in
   `test/inference.jl` can flip from `@test_broken` to `@test` with your merge (they stay
   broken against `plan_fft`, which is what they measure — perhaps add RPlan construction
   cases instead).

## 9. Open questions

- Where exactly the untrimmed build's `LLVM ERROR: Do not know how to split` arises (it is
  clearly the same dead 1-complex family dragged into AOT codegen — a partial throw-free
  generation for some (R, C=1) with wildly mixed vector widths — but the trim verifier's
  BoundsError surfaces earlier and is the cleaner repro; not pursued further because the
  fix removes the family entirely).
- Trimmed bundle sizes (blocked; then also the LLVM/codegen question of request 2).
- The plan-cache (KCACHE) is shared between plans of the same size — a C API to clear it
  (phase-0 `FFTL.clear_cache!()` exists in the package) was not added to the header yet; the
  review's interface stays as it is.

## Conclusion

Everything on my side of the fence is done and committed: merge + tests, RPlan boundary,
reviewed header with measured threading semantics, build pipeline, the full C matrix, the
Julia-side boundary tests, the extended trim check with before/after numbers — and the one
blocker is a two-line package change, precisely diagnosed, with three repros and a suggested
patch. STATUS below follows the task rules (Requests instead of touching package src/).

STATUS: blocked (the trim build needs the package-side fix of request 1 — src/rchain.jl emits dead rstage_group! branches for 1-complex vectors whose grouped-first-stage codelets cannot be generated; no library can be built until Claude Code lands it; everything else is done and committed)
