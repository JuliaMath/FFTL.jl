# C library (libfftl)

`libfftl` is FFTL compiled into a shared library that C, C++ or any language with a C interface
can call. No transform is written in C: `juliac --trim=safe` compiles the Julia package, through
the run-time plans of [Compiling ahead of time](@ref "Compiling ahead of time"), and a thin
Julia file (`libfftl/src/libfftl.jl`) defines the C entry points. A program that links the
library needs no Julia installation; the Julia runtime it needs is shipped next to it. The interface is
described in the [C API reference](@ref "libfftl C API").

!!! note "Experimental"
    The library is built and tested in CI on Linux and macOS. The C interface may still change.

## Building

The build needs Julia 1.12 or newer with `juliac` (JuliaC.jl), which the build script installs
in its own environment:

```bash
cd libfftl
./build.sh
```

The result is in `libfftl/build/lib`: the library (14 MB on macOS arm64) and the Julia runtime
next to it (10 MB: `libjulia` and the few libraries it loads, without LLVM and without BLAS).
`build.sh` also compiles the C test `test/test.c` against the library into `test/test`; run it to check the build.

Three environment variables change the build:

| variable | meaning |
|:--|:--|
| `FFTL_THREADS` | number of worker threads baked into the library, or `auto` for one per CPU of the machine that loads it. Default: `auto` on Linux; on macOS the number of performance cores of the build machine. |
| `FFTL_TRIM` | `safe` (default): trimmed image, the trim verifier must resolve every call. `off`: an untrimmed library, for diagnosing a failing trim build. |
| `JULIA_CPU_TARGET` | the CPU target of the compiled code. The script picks `apple-m1` on macOS arm64, `x86-64-v3` on x86-64 and `generic` elsewhere. One target is used because a library built for several targets hangs in a garbage collection triggered by a second thread. |

The thread count is fixed when the library is built. The Julia runtime cannot grow its worker
pool after it starts, so changing it needs a rebuild.

## Using it

A complete program that transforms a 2D array and checks the result against the input after a
forward and backward transform:

```c
#include <stdio.h>
#include <stdlib.h>
#include "fftl.h"

int main(void) {
  int64_t dims[2] = {256, 100};        /* column major: dims[0] varies fastest */
  int32_t region[2] = {0, 1};          /* transform along both dimensions */
  size_t n = 256 * 100;

  fftl_plan fwd = fftl_plan_dft(2, dims, 2, region, FFTL_PRECISION_DOUBLE, -1, 4);
  fftl_plan bwd = fftl_plan_dft(2, dims, 2, region, FFTL_PRECISION_DOUBLE, +1, 4);
  if (fwd < 0 || bwd < 0) {
    char msg[256];
    fftl_last_error(msg, sizeof msg);
    fprintf(stderr, "plan: %s\n", msg);
    return 1;
  }

  /* interleaved complex values (re, im) */
  double *x = malloc(2 * n * sizeof *x), *X = malloc(2 * n * sizeof *X);
  for (size_t i = 0; i < 2 * n; i++) x[i] = (double)(i % 7) - 3.0;

  fftl_plan_execute(fwd, X, x);        /* out of place */
  fftl_plan_execute(bwd, X, X);        /* in place */

  /* the backward transform is unnormalised: divide by n */
  double err = 0;
  for (size_t i = 0; i < 2 * n; i++) {
    double d = X[i] / (double)n - x[i];
    if (d < 0) d = -d;
    if (d > err) err = d;
  }
  printf("max round-trip error %.3g\n", err);

  fftl_plan_destroy(fwd);
  fftl_plan_destroy(bwd);
  free(x); free(X);
  return err < 1e-10 ? 0 : 1;
}
```

Compile and link it against the build:

```bash
cc -O2 -Iinclude example.c -o example -Lbuild/lib -lfftl -Wl,-rpath,"$(pwd)/build/lib" -lm
./example
```

## Rules to know

- **Column major.** Arrays are stored with the first index fastest, as in Julia. A C array
  `double a[rows][cols]` is row major: pass its dimensions reversed, or transpose the layout.
- **Complex values are interleaved** (real part, imaginary part), 16 bytes per value in double
  precision and 8 in single precision.
- **Nothing is normalised.** The backward transform multiplies by the number of transformed
  elements.
- **Errors are values.** Constructors return a negative handle on error, executions return
  `-1`; `fftl_last_error` gives the reason. A destroyed or invalid handle is rejected by
  `fftl_plan_execute` and `fftl_plan_destroy`.
- **One plan, one thread at a time.** Different plans can execute in parallel. Do not destroy a
  plan while it executes.
- **Real transforms** are separate plan constructors, `fftl_plan_dft_r2c` and `fftl_plan_dft_c2r`,
  and real-to-real transforms use `fftl_plan_r2r` with one kind per dimension. The complex array
  holds the half spectrum along the first transformed dimension: `dim/2 + 1` values there, the
  other dimensions as in the real array. **In place, the real array is padded** along that
  dimension to `2 * (dim/2 + 1)` values, so that it occupies exactly the memory of the half
  spectrum, as FFTW pads it; the values at positions `dim` and above are ignored by `r2c` and
  undefined after `c2r`.
