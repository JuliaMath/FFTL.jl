/* C test of libfftl (phase 0 probe): every size against the naive DFT where that is affordable,
 * the large sizes by the forward-backward round trip (backward is unnormalised), single and
 * double precision, and the error paths. */
#include <complex.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "fftl.h"

static const char *last_error() {
  static char buf[256];
  fftl_last_error(buf, sizeof buf);
  return buf;
}

static int failures = 0;

static double pattern_re(int64_t j, int64_t n) {
  return 0.5 + cos(2.0 * M_PI * 3.0 * j / n);
}
static double pattern_im(int64_t j, int64_t n) {
  return sin(2.0 * M_PI * 7.0 * j / n) + 0.25;
}

/* forward (sign = -1) against the naive DFT, the reference always in double precision */
static void check_naive(int32_t prec, int64_t n) {
  double complex *ref = calloc(n, sizeof *ref);
  int64_t p = fftl_plan_create(prec, n, -1);
  if (p < 0) { printf("FAIL create prec %d n %" PRId64 ": %s\n", prec, n, last_error()); failures++; free(ref); return; }
  double err = 0, scale = 0;
  if (prec == 1) {
    double complex *x = malloc(n * sizeof *x);
    double complex *y = malloc(n * sizeof *y);
    for (int64_t j = 0; j < n; j++) x[j] = pattern_re(j, n) + I * pattern_im(j, n);
    for (int64_t k = 0; k < n; k++)
      for (int64_t j = 0; j < n; j++) {
        double ang = -2.0 * M_PI * (double)k * j / n;
        ref[k] += x[j] * (cos(ang) + I * sin(ang));
      }
    if (fftl_plan_dft(p, y, x) != 0) { printf("FAIL dft prec %d n %" PRId64 ": %s\n", prec, n, last_error()); failures++; }
    for (int64_t k = 0; k < n; k++) { scale += cabs(ref[k]); err += cabs(y[k] - ref[k]); }
    free(x); free(y);
  } else {
    float complex *x = malloc(n * sizeof *x);   /* element size of the library: 8 bytes */
    float complex *y = malloc(n * sizeof *y);
    for (int64_t j = 0; j < n; j++) x[j] = (float)pattern_re(j, n) + I * (float)pattern_im(j, n);
    for (int64_t k = 0; k < n; k++)
      for (int64_t j = 0; j < n; j++) {
        double ang = -2.0 * M_PI * (double)k * j / n;
        ref[k] += (double complex)x[j] * (cos(ang) + I * sin(ang));
      }
    if (fftl_plan_dft(p, y, x) != 0) { printf("FAIL dft prec %d n %" PRId64 ": %s\n", prec, n, last_error()); failures++; }
    for (int64_t k = 0; k < n; k++) { scale += cabs(ref[k]); err += cabs((double complex)y[k] - (double complex)(float complex)ref[k]); }
    free(x); free(y);
  }
  fftl_plan_destroy(p);
  double tol = prec == 1 ? 3e-9 : 1e-3;
  if (err > tol * scale) { printf("FAIL naive prec %d n %" PRId64 ": rel err %.3e\n", prec, n, err / scale); failures++; }
  else printf("pass naive prec %d n %" PRId64 ": rel err %.3e\n", prec, n, err / scale);
  free(ref);
}

/* forward then backward gives n times the input */
static void check_roundtrip(int32_t prec, int64_t n) {
  double complex *d = NULL;
  float complex *f = NULL;
  double *xr = malloc(n * sizeof *xr);
  double *xi = malloc(n * sizeof *xi);
  for (int64_t j = 0; j < n; j++) { xr[j] = pattern_re(j, n); xi[j] = pattern_im(j, n); }
  int64_t pf = fftl_plan_create(prec, n, -1), pb = fftl_plan_create(prec, n, 1);
  if (pf < 0 || pb < 0) { printf("FAIL create roundtrip prec %d n %" PRId64 ": %s\n", prec, n, last_error()); failures++; free(xr); free(xi); return; }
  double err = 0, scale = 0;
  if (prec == 1) {
    d = malloc(2 * n * sizeof *d);
    for (int64_t j = 0; j < n; j++) d[j] = xr[j] + I * xi[j];
    fftl_plan_dft(pf, d + n, d);
    fftl_plan_dft(pb, d, d + n);
    for (int64_t j = 0; j < n; j++) { err += cabs(d[j] - n * (xr[j] + I * xi[j])); scale += n * cabs(xr[j] + I * xi[j]); }
    free(d);
  } else {
    f = malloc(2 * n * sizeof *f);
    for (int64_t j = 0; j < n; j++) f[j] = (float)xr[j] + I * (float)xi[j];
    fftl_plan_dft(pf, f + n, f);
    fftl_plan_dft(pb, f, f + n);
    for (int64_t j = 0; j < n; j++) { err += cabs(f[j] - n * ((float)xr[j] + I * (float)xi[j])); scale += n * cabs((float)xr[j] + I * (float)xi[j]); }
    free(f);
  }
  fftl_plan_destroy(pf); fftl_plan_destroy(pb);
  double tol = prec == 1 ? 1e-8 : 1e-2;
  if (err > tol * scale) { printf("FAIL roundtrip prec %d n %" PRId64 ": rel err %.3e\n", prec, n, err / scale); failures++; }
  else printf("pass roundtrip prec %d n %" PRId64 ": rel err %.3e\n", prec, n, err / scale);
  free(xr); free(xi);
}

int main(void) {
  const int64_t naive_sizes[] = {256, 1000, 101};
  const int64_t trip_sizes[] = {256, 1000, 101, 65536, 1048576};
  for (int p = 1; p <= 2; p++)
    for (size_t i = 0; i < sizeof naive_sizes / sizeof *naive_sizes; i++)
      check_naive(p, naive_sizes[i]);
  for (int p = 1; p <= 2; p++)
    for (size_t i = 0; i < sizeof trip_sizes / sizeof *trip_sizes; i++)
      check_roundtrip(p, trip_sizes[i]);

  int64_t h = fftl_plan_create(1, 123, -1);
  if (h != -1) { printf("FAIL unsupported size returned a plan\n"); failures++; fftl_plan_destroy(h); }
  else printf("pass unsupported size: %s\n", last_error());
  if (fftl_plan_dft(12345, malloc(16), malloc(16)) != -1) { printf("FAIL invalid handle accepted\n"); failures++; }
  else printf("pass invalid handle: %s\n", last_error());

  printf(failures ? "%d FAILURES\n" : "all passed (%d)\n", failures);
  return failures ? 1 : 0;
}
