/* Plan and execution time of libfftl from C, complex Float64 forward, out of place.
 * Execution: minimum over 15 samples of repeated calls (each sample >= 2 ms).
 * Build and run: see bench/run.sh. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "fftl.h"

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(void) {
  const int64_t cases[][2] = {{1, 1024}, {1, 1000}, {1, 65536}, {1, 1048576}, {1, 101}, {2, 512}, {3, 64}};
  for (int c = 0; c < 7; c++)
    for (int nt = 1; nt <= 8; nt *= 8) {
      int rank = (int)cases[c][0];
      int64_t d[3] = {cases[c][1], cases[c][1], cases[c][1]}, n = 1;
      for (int i = 0; i < rank; i++) n *= d[i];
      int reg[3] = {0, 1, 2};
      double t0 = now();
      fftl_plan p = fftl_plan_dft(rank, d, rank, reg, FFTL_PRECISION_DOUBLE, -1, nt);
      double tp = now() - t0;
      double *x = malloc(16 * n), *y = malloc(16 * n);
      for (long long i = 0; i < 2 * n; i++) x[i] = rand() / (double)RAND_MAX;
      fftl_plan_execute(p, y, x);
      double best = 1e9;
      for (int s = 0; s < 15; s++) {
        int r = 1;
        double t;
        for (;;) {
          t0 = now();
          for (int k = 0; k < r; k++) fftl_plan_execute(p, y, x);
          t = now() - t0;
          if (t > 2e-3) break;
          r *= 2;
        }
        if (t / r < best) best = t / r;
      }
      printf("C rank %d n %lld threads %d plan_ms %.3f exec_us %.2f\n", rank, (long long) cases[c][1], nt, 1e3 * tp, 1e6 * best);
      fftl_plan_destroy(p);
      free(x);
      free(y);
    }
  return 0;
}
