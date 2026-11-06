/* Plan and execution time through FFTW3's C interface, complex double forward, out of place.
 * The same source is linked against FFTW (FFTW_jll) and against libfftl (-DLINKED_FFTL), so
 * both libraries run through the same calls. Execution: minimum over 15 samples of repeated
 * calls (each sample >= 2 ms). Planner flags: FFTW_MEASURE, or FFTW_ESTIMATE with
 * BENCH_ESTIMATE=1 in the environment (libfftl ignores them).
 *
 * Usage: bench [number of cases [largest thread count]]; build and run: bench/run.sh. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "fftw3.h"

#ifdef LINKED_FFTL
#define LIBNAME "libfftl"
#else
#define LIBNAME "FFTW"
#endif

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char **argv) {
  const int cases[][2] = {{1, 1024}, {1, 1000}, {1, 65536}, {1, 1048576}, {1, 101}, {2, 512}, {3, 64}};
  int ncases = argc > 1 ? atoi(argv[1]) : 7, maxthreads = argc > 2 ? atoi(argv[2]) : 8;
  const char *est = getenv("BENCH_ESTIMATE");
  unsigned flags = est && est[0] == '1' ? FFTW_ESTIMATE : FFTW_MEASURE;
  fftw_init_threads();
  for (int c = 0; c < ncases && c < 7; c++)
    for (int nt = 1; nt <= maxthreads; nt *= 8) {
      int rank = cases[c][0];
      int d[3] = {cases[c][1], cases[c][1], cases[c][1]};
      long n = 1;
      for (int i = 0; i < rank; i++) n *= d[i];
      fftw_complex *x = fftw_alloc_complex(n), *y = fftw_alloc_complex(n);
      fftw_plan_with_nthreads(nt);
      double t0 = now();
      fftw_plan p = fftw_plan_dft(rank, d, x, y, FFTW_FORWARD, flags);
      double tp = now() - t0;
      if (!p) {
        printf("%s rank %d n %d threads %d: no plan\n", LIBNAME, rank, cases[c][1], nt);
        continue;
      }
      double *xr = (double *) x;
      for (long i = 0; i < 2 * n; i++) xr[i] = rand() / (double) RAND_MAX;
      fftw_execute_dft(p, x, y);
      double best = 1e9;
      for (int s = 0; s < 15; s++) {
        int r = 1;
        double t;
        for (;;) {
          t0 = now();
          for (int k = 0; k < r; k++) fftw_execute_dft(p, x, y);
          t = now() - t0;
          if (t > 2e-3) break;
          r *= 2;
        }
        if (t / r < best) best = t / r;
      }
      printf("%s rank %d n %d threads %d plan_ms %.3f exec_us %.2f\n", LIBNAME, rank, cases[c][1], nt, 1e3 * tp, 1e6 * best);
      fftw_destroy_plan(p);
      fftw_free(x);
      fftw_free(y);
    }
  return 0;
}
