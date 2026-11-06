# C library (libfftl)

`libfftl` is FFTL compiled into a shared library that C, C++, Fortran or any language with a C
interface can call. Its interface is FFTW3's: the C functions `fftw_*` (double) and `fftwf_*`
(single precision) of the [FFTW manual](https://www.fftw.org/fftw3_doc/), FFTW's legacy Fortran
routines and its Fortran 2003 interface. A program written for FFTW3 compiles against
libfftl's `fftw3.h` and links against libfftl without changes. The interface was designed by
the FFTW authors; libfftl implements it from the manual, with its own header files.

No transform is written in C: `juliac --trim=safe` compiles the Julia package, through the
run-time plans of [Compiling ahead of time](@ref "Compiling ahead of time"), and Julia files in
`libfftl/src` define the entry points. A program that links the library needs no Julia
installation; the Julia runtime it needs is shipped next to it. The
[C API reference](@ref "libfftl C API") lists what is supported and what differs from FFTW.

!!! note "Experimental"
    The library is built and tested in CI on Linux and macOS. Tests compare it with FFTW
    through the same C and Fortran programs.

## Building

The build needs Julia 1.12 or newer with `juliac` (JuliaC.jl), which the build script installs
in its own environment:

```bash
cd libfftl
./build.sh
```

The result is in `libfftl/build/lib`: the library (15 MB on macOS arm64) and the Julia runtime
next to it (10 MB: `libjulia` and the few libraries it loads, without LLVM and without BLAS).
Next to `libfftl` the script creates the FFTW library names as links to it: `libfftw3`,
`libfftw3f`, `libfftw3_threads`, `libfftw3f_threads`, `libfftw3_omp` and `libfftw3f_omp`
(`.so` and `.so.3` on Linux, `.dylib` and `.3.dylib` on macOS). One library serves both
precisions and the threaded variants.

Three environment variables change the build:

| variable | meaning |
|:--|:--|
| `FFTL_THREADS` | number of worker threads baked into the library, or `auto` for one per CPU of the machine that loads it. Default: `auto` on Linux; on macOS the number of performance cores of the build machine. |
| `FFTL_TRIM` | `safe` (default): trimmed image, the trim verifier must resolve every call. `off`: an untrimmed library, for diagnosing a failing trim build. |
| `JULIA_CPU_TARGET` | the CPU target of the compiled code. The script picks `apple-m1` on macOS arm64, `x86-64-v3` on x86-64 and `generic` elsewhere. One target is used because a library built for several targets hangs in a garbage collection triggered by a second thread. |

The thread count is fixed when the library is built. The Julia runtime cannot grow its worker
pool after it starts, so changing it needs a rebuild.

## Using it from C

A complete program that transforms a 2D array forward and backward and checks the result
against the input:

```c
#include <stdio.h>
#include "fftw3.h"
#include "fftl.h"

int main(void) {
  int n0 = 100, n1 = 256;                       /* row major, as in FFTW: n1 varies fastest */
  int n = n0 * n1;
  fftw_complex *x = fftw_alloc_complex(n), *X = fftw_alloc_complex(n);

  fftw_plan_with_nthreads(4);
  fftw_plan fwd = fftw_plan_dft_2d(n0, n1, x, X, FFTW_FORWARD, FFTW_MEASURE);
  fftw_plan bwd = fftw_plan_dft_2d(n0, n1, X, X, FFTW_BACKWARD, FFTW_MEASURE);
  if (!fwd || !bwd) {
    char msg[256];
    fftl_last_error(msg, sizeof msg);          /* why planning returned NULL */
    fprintf(stderr, "plan: %s\n", msg);
    return 1;
  }

  for (int i = 0; i < n; i++) {                 /* fill after planning, as FFTW requires */
    x[i][0] = (double)(i % 7) - 3.0;
    x[i][1] = (double)(i % 5) - 2.0;
  }
  fftw_execute(fwd);                            /* out of place */
  fftw_execute(bwd);                            /* in place */

  /* the backward transform is unnormalised: divide by n */
  double err = 0;
  for (int i = 0; i < n; i++)
    for (int c = 0; c < 2; c++) {
      double d = X[i][c] / n - x[i][c];
      if (d < 0) d = -d;
      if (d > err) err = d;
    }
  printf("max round-trip error %.3g\n", err);

  fftw_destroy_plan(fwd);
  fftw_destroy_plan(bwd);
  fftw_free(x);
  fftw_free(X);
  return err < 1e-10 ? 0 : 1;
}
```

Compile it against libfftl's headers and link it with `-lfftw3`, as for FFTW, or with
`-lfftl` (the same library). From `libfftl/`:

```bash
cc -O2 -Iinclude example.c -o example -Lbuild/lib -lfftw3 -Wl,-rpath,"$(pwd)/build/lib" -lm
./example
```

`fftl.h` is only needed for `fftl_last_error` and `fftl_version`; a program that uses nothing
but FFTW's functions includes `fftw3.h` alone. Single precision uses the `fftwf_` functions
and links with `-lfftw3f`, which is again the same library.

## Running an existing FFTW program

A program already built against FFTW's shared library loads libfftl instead when the library
path puts `libfftl/build/lib` first, because the links there carry FFTW's library names
(`libfftw3.so.3` on Linux, `libfftw3.3.dylib` on macOS):

```bash
LD_LIBRARY_PATH=/path/to/libfftl/build/lib ./program      # Linux
DYLD_LIBRARY_PATH=/path/to/libfftl/build/lib ./program    # macOS
```

On Linux this works when the program was linked with `RUNPATH` (`-Wl,--enable-new-dtags`, the
default of most current toolchains) or without a run path; an `RPATH` entry takes precedence over
`LD_LIBRARY_PATH`. On macOS, `DYLD_LIBRARY_PATH` is ignored for programs protected by System
Integrity Protection. The program must not use one of the functions libfftl lacks (see the
[reference](@ref "libfftl C API")); the dynamic loader then fails at start with a missing symbol.

## Using it from Fortran

Both of FFTW's Fortran interfaces are part of the library. The Fortran 2003 interface binds to
the C functions; include `fftw3.f03` from `libfftl/include`:

```fortran
use, intrinsic :: iso_c_binding
include 'fftw3.f03'
type(C_PTR) :: plan
complex(C_DOUBLE_COMPLEX), dimension(256, 100) :: x, y
plan = fftw_plan_dft_2d(100, 256, x, y, FFTW_FORWARD, FFTW_ESTIMATE)   ! dimensions reversed
call fftw_execute_dft(plan, x, y)
call fftw_destroy_plan(plan)
```

The legacy interface (`dfftw_*`, `sfftw_*`) takes the plan as an `integer*8` first argument and
the dimensions in Fortran order; its constants are in `fftw3.f`:

```fortran
include 'fftw3.f'
integer*8 plan
double complex x(256, 100), y(256, 100)
call dfftw_plan_dft_2d(plan, 256, 100, x, y, FFTW_FORWARD, FFTW_ESTIMATE)
call dfftw_execute_dft(plan, x, y)
call dfftw_destroy_plan(plan)
```

Compile with `-I/path/to/libfftl/include` and link with `-lfftw3 -lfftw3f` (or `-lfftl`). The
legacy routines exist under both name manglings the FFTW manual describes, one trailing
underscore (gfortran, ifort, flang) and two (g77).

## Rules to know

- **FFTW's conventions.** Arrays are row major in C (the last dimension varies fastest),
  complex values are interleaved (real part, imaginary part) unless the split interface is
  used, and nothing is normalised: the backward transform multiplies by the number of
  transformed elements.
- **Any layout.** Strided, embedded and split arrays are accepted. Layouts that are one
  contiguous array run on the arrays directly; the others are copied into a buffer of the plan
  and back, which adds one pass over the input and one over the output.
- **Errors.** A planner that cannot plan returns `NULL`, and `fftl_last_error` gives the reason.
  Execute functions called with an invalid plan, a plan of another type, or arrays that differ
  in in-place-ness from the planning arrays do nothing and set the message as well.
- **Threads.** `fftw_plan_with_nthreads` sets how many of the library's worker threads later
  plans use; the worker pool itself is fixed when the library is built. One plan may execute
  on several threads at once through the new-array execute functions, as in FFTW. Do not
  destroy a plan while it executes.
- **Planner flags** are accepted and change nothing, except `FFTW_WISDOM_ONLY`: libfftl keeps no
  wisdom, so such a plan is `NULL`. Planning never reads or writes the arrays.
