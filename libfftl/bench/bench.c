/* Interleaved timing, C side: libfftl called from C, one measurement round per invocation
 * (run.sh alternates with jbench.jl round-wise; aggreg.jl computes min/median over rounds).
 * Every timed batch runs at least 50 µs; times are printed in ns per call. The create batch
 * pushes plan handles and destroys them untimed afterwards. */
#include <complex.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fftl.h"

static double now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

int main(int argc, char **argv) {
  static const int precs[] = {1, 2};
  static const long long sizes[] = {256, 1000, 65536, 1048576, 101};
  static const int signs[] = {-1, 1};
  static const int max_batch = 4096;

  int prec_list[2], ns = 2;
  long long size_list[5], nsz = 5;
  int sign_list[2], ng = 2;
  int single = 0;
  if (argc == 4) {                    /* bench prec n sign: a single variant */
    prec_list[0] = atoi(argv[1]);
    size_list[0] = atoll(argv[2]);
    sign_list[0] = atoi(argv[3]);
    ns = nsz = ng = 1;
    single = 1;
  } else {
    memcpy(prec_list, precs, sizeof precs);
    memcpy(size_list, sizes, sizeof sizes);
    memcpy(sign_list, signs, sizeof signs);
  }

  for (int ip = 0; ip < ns; ip++)
    for (int is = 0; is < nsz; is++)
      for (int ig = 0; ig < ng; ig++) {
        const int prec = prec_list[ip];
        const long long n = size_list[is];
        const int sign = sign_list[ig];
        (void)single;
        const size_t elt = prec == 1 ? sizeof(double complex) : sizeof(float complex);
        void *x = malloc(elt * (size_t)n);
        void *y = malloc(elt * (size_t)n);
        if (prec == 1) {
          double complex *z = x;
          for (long long j = 0; j < n; j++)
            z[j] = (double)(j % 7 - 3) + I * (double)(j % 5 - 2);
        } else {
          float complex *z = x;
          for (long long j = 0; j < n; j++)
            z[j] = (float)(j % 7 - 3) + I * (float)(j % 5 - 2);
        }
        if (x == NULL || y == NULL) { fprintf(stderr, "out of memory\n"); return 1; }

        /* warmup */
        for (int w = 0; w < 2; w++) {
          int64_t h = fftl_plan_create(prec, n, sign);
          fftl_plan_destroy(h);
        }
        int64_t h = fftl_plan_create(prec, n, sign);
        for (int w = 0; w < 3; w++) fftl_plan_execute(h, y, x);

        /* create (plan + the work buffers of the entry), batch of >= 50 µs */
        int64_t hs[max_batch];
        int k = 0;
        double t0 = now_ns();
        for (;;) {
          hs[k++] = fftl_plan_create(prec, n, sign);
          double dt = now_ns() - t0;
          if (dt >= 5e4 || k == max_batch) {
            for (int i = 0; i < k; i++) fftl_plan_destroy(hs[i]);
            hs[0] = hs[k - 1] = -1;
            break;
          }
        }
        double create_ns = (now_ns() - t0) / k;

        /* execute, batch of >= 50 µs on the warmed plan */
        double t1 = now_ns();
        long long m = 0;
        do {
          fftl_plan_execute(h, y, x);
          m++;
        } while (now_ns() - t1 < 5e4);
        double exec_ns = (now_ns() - t1) / m;

        printf("bench=c prec=%d n=%lld sign=%d create=%.4f exec=%.4f\n", prec, n, sign,
               create_ns, exec_ns);
        fftl_plan_destroy(h);
        free(y);
        free(x);
      }
  return 0;
}
