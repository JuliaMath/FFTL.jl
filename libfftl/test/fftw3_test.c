/* Differential test of the FFTW3 interface: one program, written against libfftl's fftw3.h,
 * built and run twice by fftw3_ctest.sh, once linked against FFTW itself (FFTW_jll) and once
 * against libfftl's libfftw3/libfftw3f links. Every case is checked here against a direct
 * DFT or r2r sum in long double, including strided, embedded and split layouts; the outputs go to binary files that fftw3_compare.jl
 * compares between the two builds (this also covers sizes beyond the direct sums).
 *
 * The FFTW build passing shows that fftw3.h (prototypes, fftw_iodim, kind and flag values)
 * matches the real library; the libfftl build passing shows the interface.
 *
 * Build options: -DWITH_COMPLEX_H includes <complex.h> before fftw3.h (fftw_complex is then the
 * C99 complex type); -DLINKED_FFTL adds the checks of what libfftl rejects by design.
 *
 * Usage: fftw3_test <output directory>
 */

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef WITH_COMPLEX_H
#include <complex.h>
#endif
#include "fftw3.h"
#ifdef LINKED_FFTL
#include "fftl.h"
#endif

static int ntests = 0, nfailed = 0;
static const char *outdir = ".";

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    ntests++;                                                                  \
    if (!(cond)) {                                                             \
      nfailed++;                                                               \
      printf("FAIL %s:%d: ", __FILE__, __LINE__);                              \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
    }                                                                          \
  } while (0)

/* Inputs from a fixed generator, identical in both builds: uniform in [-1, 1). */
static uint64_t rng_state = 1;
static void rng_seed(uint64_t s) { rng_state = s * 6364136223846793005ULL + 1442695040888963407ULL; }
static double urand(void) {
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double) (rng_state >> 11) / 4503599627370496.0 - 1.0;
}

/* The output of a case, as raw bytes, to <outdir>/<name>.bin */
static void dump(const char *name, const void *data, size_t bytes) {
  char path[1024];
  snprintf(path, sizeof path, "%s/%s.bin", outdir, name);
  FILE *f = fopen(path, "wb");
  if (!f) {
    printf("cannot write %s\n", path);
    exit(2);
  }
  fwrite(data, 1, bytes, f);
  fclose(f);
}

/* --------------------------------------------------------------- reference in long double */

typedef long double LD;
static const LD PI_L = 3.141592653589793238462643383279502884L;

/* Row-major array of rank dims n[]: stride of dimension d */
static long rm_stride(int rank, const int *n, int d) {
  long s = 1;
  for (int i = d + 1; i < rank; i++) s *= n[i];
  return s;
}
static long prod(int rank, const int *n) {
  long s = 1;
  for (int i = 0; i < rank; i++) s *= n[i];
  return s;
}

/* Complex DFT along dimension d of a row-major array of (re, im) pairs, in place. */
static void ref_dft_dim(LD *a, int rank, const int *n, int d, int sign) {
  long S = rm_stride(rank, n, d), len = n[d], total = prod(rank, n);
  LD *line = malloc(2 * len * sizeof(LD));
  for (long base = 0; base < total; base++) {
    if ((base / S) % len != 0) continue; /* base runs over the first element of each line */
    for (long j = 0; j < len; j++) {
      line[2 * j] = a[2 * (base + j * S)];
      line[2 * j + 1] = a[2 * (base + j * S) + 1];
    }
    for (long k = 0; k < len; k++) {
      LD sr = 0, si = 0;
      for (long j = 0; j < len; j++) {
        LD ang = 2 * PI_L * sign * (LD) ((j * k) % len) / len;
        LD c = cosl(ang), s = sinl(ang);
        sr += line[2 * j] * c - line[2 * j + 1] * s;
        si += line[2 * j] * s + line[2 * j + 1] * c;
      }
      a[2 * (base + k * S)] = sr;
      a[2 * (base + k * S) + 1] = si;
    }
  }
  free(line);
}

/* One r2r line of kind k (FFTW's definitions), length len, from x to y */
static void ref_r2r_line(const LD *x, LD *y, long n, int kind) {
  for (long k = 0; k < n; k++) {
    LD s = 0;
    switch (kind) {
    case FFTW_R2HC: /* halfcomplex: r_k at k <= n/2, i_k at n-k */
      if (k <= n / 2) {
        for (long j = 0; j < n; j++) s += x[j] * cosl(2 * PI_L * (LD) ((j * k) % n) / n);
      } else {
        long kk = n - k;
        for (long j = 0; j < n; j++) s -= x[j] * sinl(2 * PI_L * (LD) ((j * kk) % n) / n);
      }
      break;
    case FFTW_HC2R:
      s = x[0];
      for (long m = 1; 2 * m < n; m++) {
        LD ang = 2 * PI_L * (LD) ((m * k) % n) / n;
        s += 2 * (x[m] * cosl(ang) - x[n - m] * sinl(ang));
      }
      if (n % 2 == 0) s += x[n / 2] * ((k % 2) ? -1 : 1);
      break;
    case FFTW_DHT:
      for (long j = 0; j < n; j++) {
        LD ang = 2 * PI_L * (LD) ((j * k) % n) / n;
        s += x[j] * (cosl(ang) + sinl(ang));
      }
      break;
    case FFTW_REDFT00:
      s = x[0] + ((k % 2) ? -1 : 1) * x[n - 1];
      for (long j = 1; j < n - 1; j++) s += 2 * x[j] * cosl(PI_L * j * k / (n - 1));
      break;
    case FFTW_REDFT10:
      for (long j = 0; j < n; j++) s += 2 * x[j] * cosl(PI_L * (j + 0.5L) * k / n);
      break;
    case FFTW_REDFT01:
      s = x[0];
      for (long j = 1; j < n; j++) s += 2 * x[j] * cosl(PI_L * j * (k + 0.5L) / n);
      break;
    case FFTW_REDFT11:
      for (long j = 0; j < n; j++) s += 2 * x[j] * cosl(PI_L * (j + 0.5L) * (k + 0.5L) / n);
      break;
    case FFTW_RODFT00:
      for (long j = 0; j < n; j++) s += 2 * x[j] * sinl(PI_L * (j + 1) * (k + 1) / (n + 1));
      break;
    case FFTW_RODFT10:
      for (long j = 0; j < n; j++) s += 2 * x[j] * sinl(PI_L * (j + 0.5L) * (k + 1) / n);
      break;
    case FFTW_RODFT01:
      s = ((k % 2) ? -1 : 1) * x[n - 1];
      for (long j = 0; j < n - 1; j++) s += 2 * x[j] * sinl(PI_L * (j + 1) * (k + 0.5L) / n);
      break;
    case FFTW_RODFT11:
      for (long j = 0; j < n; j++) s += 2 * x[j] * sinl(PI_L * (j + 0.5L) * (k + 0.5L) / n);
      break;
    }
    y[k] = s;
  }
}

static void ref_r2r_dim(LD *a, int rank, const int *n, int d, int kind) {
  long S = rm_stride(rank, n, d), len = n[d], total = prod(rank, n);
  LD *x = malloc(len * sizeof(LD)), *y = malloc(len * sizeof(LD));
  for (long base = 0; base < total; base++) {
    if ((base / S) % len != 0) continue;
    for (long j = 0; j < len; j++) x[j] = a[base + j * S];
    ref_r2r_line(x, y, len, kind);
    for (long j = 0; j < len; j++) a[base + j * S] = y[j];
  }
  free(x);
  free(y);
}

/* The position of element idx (row-major index into an array of size counts) in a layout with
 * the given strides per dimension */
static long lay_off(int k, const int *counts, const long *strides, long idx) {
  long o = 0;
  for (int d = k - 1; d >= 0; d--) {
    o += (idx % counts[d]) * strides[d];
    idx /= counts[d];
  }
  return o;
}

/* The row-major index into an array of size to of the element with the same subscripts as
 * element idx of an array of size from */
static long lay_remap(int k, const int *from, const int *to, long idx) {
  long r = 0, m = 1;
  for (int d = k - 1; d >= 0; d--) {
    r += (idx % from[d]) * m;
    m *= to[d];
    idx /= from[d];
  }
  return r;
}

/* Relative maximum error of a computed array (stride 1, count values) against a reference */
static double relerr_r(const double *y, const LD *ref, long count) {
  LD emax = 0, rmax = 0;
  for (long i = 0; i < count; i++) {
    LD e = fabsl(y[i] - ref[i]), r = fabsl(ref[i]);
    if (e > emax) emax = e;
    if (r > rmax) rmax = r;
  }
  return (double) (emax / (rmax > 0 ? rmax : 1));
}

static double relerr_f(const float *y, const LD *ref, long count) {
  LD emax = 0, rmax = 0;
  for (long i = 0; i < count; i++) {
    LD e = fabsl(y[i] - ref[i]), r = fabsl(ref[i]);
    if (e > emax) emax = e;
    if (r > rmax) rmax = r;
  }
  return (double) (emax / (rmax > 0 ? rmax : 1));
}

/* The real/complex half spectrum: r2c of a real row-major array (logical dims n) in long
 * double, written as (re, im) pairs of the n[0] x ... x (n[rank-1]/2+1) array. */
static void ref_r2c(const LD *x, int rank, const int *n, LD *half) {
  long total = prod(rank, n);
  LD *a = malloc(2 * total * sizeof(LD));
  for (long i = 0; i < total; i++) {
    a[2 * i] = x[i];
    a[2 * i + 1] = 0;
  }
  for (int d = 0; d < rank; d++) ref_dft_dim(a, rank, n, d, FFTW_FORWARD);
  long last = n[rank - 1], h = last / 2 + 1, outer = total / last;
  for (long o = 0; o < outer; o++)
    for (long k = 0; k < h; k++) {
      half[2 * (o * h + k)] = a[2 * (o * last + k)];
      half[2 * (o * h + k) + 1] = a[2 * (o * last + k) + 1];
    }
  free(a);
}

/* ---------------------------------------------------- the cases, once per precision */

#define PREC 64
#define R double
#define X(name) fftw_##name
#define TOL 1e-12
#define RELERR relerr_r
#include "fftw3_test_cases.h"
#undef PREC
#undef R
#undef X
#undef TOL
#undef RELERR

#define PREC 32
#define R float
#define X(name) fftwf_##name
#define TOL 2e-5
#define RELERR relerr_f
#include "fftw3_test_cases.h"

int main(int argc, char **argv) {
  if (argc > 1) outdir = argv[1];
  run_64();
  run_32();
  printf("fftw3_test (%s, %s): %d checks, %d failed\n",
#ifdef LINKED_FFTL
         "libfftl",
#else
         "FFTW",
#endif
#ifdef WITH_COMPLEX_H
         "complex.h",
#else
         "double[2]",
#endif
         ntests, nfailed);
  return nfailed == 0 ? 0 : 1;
}
