/* Calls the trim probe of the run-time pass plans; checks the norm of the transformed array of
   ones: n1 n2 n3 for the complex passes, n1 sqrt(n1 n2) for r2c followed by c2r (unnormalised). */
#include <math.h>
#include <stdio.h>

double fftl_probe_complex64(int, int, int, int);
double fftl_probe_complex32(int, int, int, int);
double fftl_probe_real64(int, int, int);
double fftl_probe_real32(int, int, int);

static int check(const char *what, double got, double want, double tol) {
  double e = fabs(got - want) / want;
  printf("%-10s %8.3g relative error %.2e\n", what, want, e);
  return e > tol;
}

int main(void) {
  /* smooth, large prime, composite with few lanes (Resample/Bluestein), with many lanes (RsBatch) */
  int lens[] = {64, 1009, 2 * 1009, 64 * 1009};
  int fail = 0;
  for (int i = 0; i < 4; i++) {
    int n = lens[i];
    for (int nt = 1; nt <= 2; nt++) {
      fail |= check("c64", fftl_probe_complex64(n, 3, 5, nt), (double)n * 15, 1e-12);
      fail |= check("c64 d2", fftl_probe_complex64(3, n, 2, nt), (double)n * 6, 1e-12);
      fail |= check("c32", fftl_probe_complex32(n, 3, 5, nt), (double)n * 15, 1e-4);
      fail |= check("r64", fftl_probe_real64(n, 3, nt), n * sqrt(3.0 * n), 1e-12);
      fail |= check("r32", fftl_probe_real32(n, 3, nt), n * sqrt(3.0 * n), 1e-4);
    }
  }
  printf(fail ? "FAILED\n" : "passed\n");
  return fail;
}
