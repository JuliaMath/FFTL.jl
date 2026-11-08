# libfftl C API

libfftl's interface is FFTW3's, as the [FFTW manual](https://www.fftw.org/fftw3_doc/) describes
it; the manual is the reference for every function, argument and convention. libfftl's own
header files are [`fftw3.h`](@ref "C header (fftw3.h)"), with the list of differences at its
top, and [`fftl.h`](@ref "C header (fftl.h)") for the two functions FFTW has no counterpart for.
The Fortran include files `fftw3.f` and `fftw3.f03` are in the same directory
(`libfftl/include`). This page lists what is supported, what differs and what is missing.

## Supported

Every function below exists in double (`fftw_`) and single precision (`fftwf_`), from one
library:

| group | functions |
|:--|:--|
| basic planners | `plan_dft{,_1d,_2d,_3d}`, `plan_dft_r2c{,_1d,_2d,_3d}`, `plan_dft_c2r{,_1d,_2d,_3d}`, `plan_r2r{,_1d,_2d,_3d}` |
| advanced planners | `plan_many_dft`, `plan_many_dft_r2c`, `plan_many_dft_c2r`, `plan_many_r2r` |
| guru planners | `plan_guru{,64}_dft`, `_dft_r2c`, `_dft_c2r`, `_r2r`, and the split-array `plan_guru{,64}_split_dft`, `_split_dft_r2c`, `_split_dft_c2r` |
| execution | `execute`, `execute_dft`, `execute_dft_r2c`, `execute_dft_c2r`, `execute_r2r`, `execute_split_dft`, `execute_split_dft_r2c`, `execute_split_dft_c2r` |
| plans | `destroy_plan`, `copy_plan`, `cleanup`, `set_timelimit`, `cost`, `flops`, `print_plan`, `fprint_plan`, `sprint_plan` |
| memory | `malloc`, `free`, `alloc_real`, `alloc_complex`, `alignment_of` |
| wisdom | `export_wisdom{,_to_filename,_to_file,_to_string}`, `import_wisdom{,_from_filename,_from_file,_from_string}`, `import_system_wisdom`, `forget_wisdom` |
| threads | `init_threads`, `plan_with_nthreads`, `planner_nthreads`, `cleanup_threads`, `make_planner_thread_safe`, `threads_set_callback` |

All eleven real-to-real kinds (`FFTW_R2HC`, `FFTW_HC2R`, `FFTW_DHT`, `FFTW_REDFT00` …
`FFTW_REDFT11`, `FFTW_RODFT00` … `FFTW_RODFT11`), any rank, any lengths (lengths with large
prime factors are resampled onto a smooth grid), and every layout the manual describes: strides,
embedded arrays, vector loops, different input and output layouts, in place and out of place,
interleaved and split complex arrays.

The legacy Fortran interface has the routines the manual gives it: every function above
without `FILE`, string or memory arguments, as subroutines `dfftw_…` and `sfftw_…` with the
return value as first argument, each with one and with two trailing underscores. The Fortran
2003 interface (`fftw3.f03`) declares the C functions for `iso_c_binding`; as the manual
describes, it leaves out `fftw_execute`.

libfftl's own functions (`fftl.h`):

| function | purpose |
|:--|:--|
| `fftl_last_error(buf, len)` | the message of the last failed call on this thread: every `NULL` plan, rejected executions and failed allocations leave one |
| `fftl_version(buf, len)` | the library version as `"major.minor.patch"` |

## Differences from FFTW

The list at the top of [`fftw3.h`](@ref "C header (fftw3.h)") is the complete one. In short:

- **Layouts outside one contiguous array are copied.** Plans whose input and output form one
  dense array (in any order of dimensions; for r2c/c2r FFTW's padded in-place layout, with the
  halved dimension the fastest of the transform dimensions) run on the arrays. All other
  layouts, and all split arrays, are gathered into a buffer of the plan and scattered back,
  one extra pass over the input and one over the output.
- **Planner flags change nothing**, except `FFTW_WISDOM_ONLY`, which gives `NULL` because there
  is never wisdom. Planning never touches the arrays. Out-of-place c2r keeps its input.
- **Wisdom.** Export writes a placeholder; import accepts any input and returns 1 (0 for a file
  that cannot be opened); `fftw_import_system_wisdom` returns 0.
- **`fftw_cost` and `fftw_flops`** return an estimate (5 N log₂ N for a complex DFT of N points,
  half of it for real transforms), not a count of FFTL's operations.
- **Alignment does not matter**: `fftw_alignment_of` returns 0 for every pointer, and the
  new-array execute functions accept arrays of any alignment. `fftw_malloc` aligns to 64 bytes.
- **Threads.** The worker pool is fixed when the library is built (`FFTL_THREADS`, see
  [C library (libfftl)](@ref "C library (libfftl)")); `fftw_plan_with_nthreads` chooses how many
  of its threads later plans use. `fftw_threads_set_callback` is accepted and ignored.
- **Leniency.** Some calls that FFTW rejects or leaves undefined have a defined result: in-place
  split r2c/c2r of rank above 1 is planned, plans survive `fftw_cleanup`, and execute functions
  check the plan, its transform type and the in-place-ness of the arrays and set
  `fftl_last_error` instead of running.
- **Split plans** execute through `fftw_execute` or the split execute functions only, and
  interleaved plans through the interleaved ones.

## Not implemented

| item | why |
|:--|:--|
| the variables `fftw_version`, `fftw_cc`, `fftw_codelet_optim` | `juliac` exports functions only; a program that reads them does not link against libfftl |
| `fftw_estimate_cost` | not described in the manual |
| long double (`fftwl_`) and quad precision (`fftwq_`) | FFTL's Julia API supports any `AbstractFloat`, but the library is built for `float` and `double` only |
| MPI (`fftw_mpi_`) | no distributed-memory transforms |
| FFTW's wisdom format | the manual does not specify it; libfftl's export cannot be read by FFTW |

## Limits

| | limit |
|:--|:--|
| sizes | any rank, any lengths ≥ 1 for transform dimensions, ≥ 0 for vector dimensions (0: a plan that does nothing) |
| strides | any, including negative ones |
| normalisation | none; the backward transform multiplies by the number of transformed elements |
| memory | bounded by the available memory and by 64-bit element offsets |
| concurrency | plans are created, executed and destroyed from any thread; one plan may run on several threads at once through the new-array execute functions; destroying a plan during its execution is not allowed |
| number types | `float` and `double` only; no GPU |

## Build and size

On macOS (arm64), the build output in `libfftl/build/lib` is 32 MB: the library (23 MB) and the
Julia runtime next to it (10 MB). No Julia installation is needed to run a program that links it. The build
needs Julia 1.12 or newer with `juliac`. See [C library (libfftl)](@ref "C library (libfftl)")
for the build variables.
