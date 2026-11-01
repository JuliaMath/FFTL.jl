/* libfftl C tests (phase 3): the fftl_plan_dft / fftl_plan_execute / fftl_plan_destroy API
 * against a naive direct DFT in double precision.
 *
 * Covered: 1D smooth and prime lengths (Bluestein, also inside higher ranks), 2D and 3D full
 * and partial regions, in place and out of place, both precisions, threads 1 and 4,
 * empty regions (copy), round trips for lengths beyond the naive DFT, and the error paths
 * (invalid handles, precision, sign, rank, region violations, sizes, thread count).
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fftl.h"

static int ntests = 0, nfailed = 0;

#define CHECK(cond, name)                                                      \
  do {                                                                         \
    ntests++;                                                                  \
    if (cond) {                                                                \
    } else {                                                                   \
      nfailed++;                                                                \
      printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, name);                    \
    }                                                                          \
  } while (0)

typedef struct {
  double re, im;
} cpx;

/* ------------------------------------------------------------------ reference DFT */

/* One pass of the DFT along dimension d (0-based) of a column-major array with dims[0..rank-1],
 * applied in place to a; element (i1..ir) at i1 + dims[0]*(i2 + ...), like the library. */
static void dft_pass(cpx *a, const int64_t *dims, int rank, int d, int sign) {
  int64_t S = 1, n = dims[d], outer = 1;
  for (int i = 0; i < d; i++) S *= dims[i];
  for (int i = d + 1; i < rank; i++) outer *= dims[i];
  cpx *line = malloc(n * sizeof(cpx));
  for (int64_t o = 0; o < outer; o++) {
    for (int64_t l = 0; l < S; l++) {
      int64_t base = o * S * n + l;
      for (int64_t j = 0; j < n; j++) line[j] = a[base + S * j];
      for (int64_t k = 0; k < n; k++) {
        double sr = 0, si = 0;
        for (int64_t j = 0; j < n; j++) {
          double ang = 2.0 * M_PI * sign * (j * k % n) / (double) n;
          sr += line[j].re * cos(ang) - line[j].im * sin(ang);
          si += line[j].re * sin(ang) + line[j].im * cos(ang);
        }
        a[base + S * k].re = sr;
        a[base + S * k].im = si;
      }
    }
  }
  free(line);
}

/* DFT along every dimension in region, applied in the given (ascending) order. */
static void dft_region(cpx *a, const int64_t *dims, int rank, const int32_t *region,
                       int trank, int sign) {
  for (int i = 0; i < trank; i++) dft_pass(a, dims, rank, region[i], sign);
}

static int64_t prod(const int64_t *dims, int rank) {
  int64_t p = 1;
  for (int i = 0; i < rank; i++) p *= dims[i];
  return p;
}

/* max |a - b| / max |b| over complex values stored as consecutive double pairs. */
static double relerr(const double *a, const double *b, int64_t nval) {
  double maxdiff = 0, maxabs = 0;
  for (int64_t i = 0; i < 2 * nval; i += 2) {
    double dr = fabs(a[i] - b[i]), di = fabs(a[i + 1] - b[i + 1]);
    if (dr > maxdiff) maxdiff = dr;
    if (di > maxdiff) maxdiff = di;
    if (fabs(b[i]) > maxabs) maxabs = fabs(b[i]);
    if (fabs(b[i + 1]) > maxabs) maxabs = fabs(b[i + 1]);
  }
  return maxabs > 0 ? maxdiff / maxabs : maxdiff;
}

/* ------------------------------------------------------------------ test scaffolding */

static char errbuf[512];

static void expect_error(const char *name, fftl_plan p) {
  CHECK(p < 0, name);
  size_t n = fftl_last_error(errbuf, sizeof errbuf);
  CHECK(n > 0 && errbuf[0] != '\0', name);   /* the message explains the failure */
}

static fftl_plan must_plan(const char *name, int32_t rank, const int64_t *dims, int32_t trank,
                           const int32_t *region, int32_t precision, int32_t sign,
                           int32_t threads) {
  fftl_plan p = fftl_plan_dft(rank, dims, trank, region, precision, sign, threads);
  CHECK(p >= 0, name);
  return p;
}

/* Deterministic smooth pattern into nval complex double values. */
static void fill(double *v, int64_t nval) {
  unsigned s = 12345;
  for (int64_t i = 0; i < nval; i++) {
    s = s * 1664525u + 1013904223u;
    v[2 * i] = (double) (s >> 16) / 65536.0 - 0.5;
    s = s * 1664525u + 1013904223u;
    v[2 * i + 1] = (double) (s >> 16) / 65536.0 - 0.5;
  }
}

/* One library transform against the naive reference, out of place and in place. esize is the
 * library element size in bytes (8 single, 16 double); tol is the relative tolerance against
 * the double reference. */
static void check_region(const char *name, const int64_t *dims, int rank, const int32_t *region,
                         int trank, int sign, int32_t precision, int32_t threads, double tol) {
  int64_t nval = prod(dims, rank);
  int64_t esize = precision == FFTL_PRECISION_DOUBLE ? 16 : 8;
  size_t bytes = (size_t) nval * esize;
  cpx *ref = malloc(nval * sizeof(cpx));
  double *xin = malloc(nval * sizeof(cpx));
  unsigned char *lx = malloc(bytes), *ly = malloc(bytes), *lz = malloc(bytes);

  fill(xin, nval);
  if (precision == FFTL_PRECISION_DOUBLE) {
    memcpy(lx, xin, bytes);
  } else {
    float *lf = (float *) (void *) lx;
    for (int64_t i = 0; i < 2 * nval; i++) lf[i] = (float) xin[i];
  }

  fftl_plan p = fftl_plan_dft(rank, dims, trank, region, precision, sign, threads);
  CHECK(p >= 0, name);
  if (p < 0) goto done;

  /* reference: naive DFT on the double view of the input */
  memcpy(ref, xin, nval * sizeof(cpx));
  dft_region(ref, dims, rank, region, trank, sign);

  /* out of place into poisoned memory: every output element must be written */
  memset(ly, 0xAA, bytes);
  CHECK(fftl_plan_execute(p, ly, lx) == 0, name);
  if (precision == FFTL_PRECISION_DOUBLE) {
    CHECK(relerr((double *) (void *) ly, (double *) (void *) ref, nval) < tol, name);
  } else {
    float *yf = (float *) (void *) ly;
    double *yd = malloc(2 * nval * sizeof(double));
    for (int64_t i = 0; i < nval; i++) {
      yd[2 * i] = yf[2 * i];
      yd[2 * i + 1] = yf[2 * i + 1];
    }
    CHECK(relerr(yd, (double *) (void *) ref, nval) < tol, name);
    free(yd);
  }

  /* in place */
  memcpy(lz, lx, bytes);
  CHECK(fftl_plan_execute(p, lz, lz) == 0, name);
  if (precision == FFTL_PRECISION_DOUBLE) {
    CHECK(relerr((double *) (void *) lz, (double *) (void *) ref, nval) < tol, name);
  } else {
    float *zf = (float *) (void *) lz;
    double *zd = malloc(2 * nval * sizeof(double));
    for (int64_t i = 0; i < nval; i++) {
      zd[2 * i] = zf[2 * i];
      zd[2 * i + 1] = zf[2 * i + 1];
    }
    CHECK(relerr(zd, (double *) (void *) ref, nval) < tol, name);
    free(zd);
  }

  CHECK(fftl_plan_destroy(p) == 0, name);

done:
  free(ref); free(xin); free(lx); free(ly); free(lz);
}

/* Forward-backward round trip in double; bfft(fft(x)) = x * prod(dims over the region). */
static void check_roundtrip(const char *name, const int64_t *dims, int rank, int32_t trank,
                            const int32_t *region, int threads, double tol) {
  int64_t nval = prod(dims, rank);
  int64_t scale = 1;
  for (int i = 0; i < trank; i++) scale *= dims[region[i]];
  double *x = malloc(nval * sizeof(cpx)), *y = malloc(nval * sizeof(cpx)),
         *orig = malloc(nval * sizeof(cpx)), *scaled = malloc(nval * sizeof(cpx));
  fill(x, nval);
  memcpy(orig, x, nval * sizeof(cpx));

  fftl_plan pf = fftl_plan_dft(rank, dims, trank, region, FFTL_PRECISION_DOUBLE, -1, threads);
  fftl_plan pb = fftl_plan_dft(rank, dims, trank, region, FFTL_PRECISION_DOUBLE, 1, threads);
  CHECK(pf >= 0 && pb >= 0, name);
  if (pf < 0 || pb < 0) goto done;
  CHECK(fftl_plan_execute(pf, y, x) == 0, name);
  CHECK(fftl_plan_execute(pb, x, y) == 0, name);
  for (int64_t i = 0; i < 2 * nval; i++) scaled[i] = x[i] / (double) scale;
  CHECK(relerr(scaled, orig, nval) < tol, name);
  CHECK(fftl_plan_destroy(pf) == 0 && fftl_plan_destroy(pb) == 0, name);
done:
  free(x); free(y); free(orig); free(scaled);
}

/* -------------------------------------------------------------------------- tests */

static void test_version_and_error(void) {
  char buf[64];
  size_t n = fftl_version(buf, sizeof buf);
  CHECK(n == 5 && strcmp(buf, "0.1.0") == 0, "fftl_version 0.1.0");
  n = fftl_version(buf, 3);
  CHECK(n == 5 && strcmp(buf, "0.") == 0, "fftl_version truncates like snprintf");

  fftl_plan bad = fftl_plan_dft(1, (int64_t[]){256}, 1, (int32_t[]){0}, 99, -1, 1);
  CHECK(bad < 0, "precision 99 rejected");
  n = fftl_last_error(buf, sizeof buf);
  CHECK(n > 0 && buf[0] != '\0', "error message present");

  /* successful calls do not clear the message */
  fftl_plan p = must_plan("valid plan after an error", 1, (int64_t[]){64}, 1, (int32_t[]){0},
                          FFTL_PRECISION_DOUBLE, -1, 1);
  fftl_last_error(buf, sizeof buf);
  CHECK(buf[0] != '\0', "error survives a successful call");
  CHECK(fftl_plan_destroy(p) == 0, "destroy valid");
}

static void test_1d_double(void) {
  CHECK(prod((int64_t[]){1}, 1) == 1, "sanity");
  for (int c = 0; c < 5; c++) {
    int64_t len = ((int64_t[]){1, 2, 256, 1000, 101})[c];   /* 101 prime: Bluestein */
    char name[80];
    snprintf(name, sizeof name, "1d f64 n=%lld forward", (long long) len);
    check_region(name, (int64_t[]){len}, 1, (int32_t[]){0}, 1, -1, FFTL_PRECISION_DOUBLE, 1,
                 1e-13);
  }
  check_region("1d f64 n=256 backward", (int64_t[]){256}, 1, (int32_t[]){0}, 1, 1,
               FFTL_PRECISION_DOUBLE, 1, 1e-13);
  check_region("1d f64 n=101 backward", (int64_t[]){101}, 1, (int32_t[]){0}, 1, 1,
               FFTL_PRECISION_DOUBLE, 1, 1e-13);
  check_region("1d f64 n=9973 forward", (int64_t[]){9973}, 1, (int32_t[]){0}, 1, -1,
               FFTL_PRECISION_DOUBLE, 1, 1e-13);   /* large prime: Bluestein M = 20188 */
}

static void test_1d_single(void) {
  check_region("1d f32 n=256 forward", (int64_t[]){256}, 1, (int32_t[]){0}, 1, -1,
               FFTL_PRECISION_SINGLE, 1, 2e-6);
  check_region("1d f32 n=1000 forward", (int64_t[]){1000}, 1, (int32_t[]){0}, 1, -1,
               FFTL_PRECISION_SINGLE, 1, 2e-6);
  check_region("1d f32 n=101 forward", (int64_t[]){101}, 1, (int32_t[]){0}, 1, -1,
               FFTL_PRECISION_SINGLE, 1, 5e-5);
  check_region("1d f32 n=101 backward", (int64_t[]){101}, 1, (int32_t[]){0}, 1, 1,
               FFTL_PRECISION_SINGLE, 1, 5e-5);
  check_region("1d f32 n=9973 forward", (int64_t[]){9973}, 1, (int32_t[]){0}, 1, -1,
               FFTL_PRECISION_SINGLE, 1, 2e-4);
}

static void test_2d(void) {
  check_region("2d f64 (8,12) region {0,1}", (int64_t[]){8, 12}, 2, (int32_t[]){0, 1}, 2, -1,
               FFTL_PRECISION_DOUBLE, 1, 1e-12);
  check_region("2d f64 (5,37) region {0,1} Bluestein", (int64_t[]){5, 37}, 2, (int32_t[]){0, 1},
               2, -1, FFTL_PRECISION_DOUBLE, 1, 1e-12);
  check_region("2d f64 (3,8) region {1}", (int64_t[]){3, 8}, 2, (int32_t[]){1}, 1, -1,
               FFTL_PRECISION_DOUBLE, 1, 1e-12);
  check_region("2d f64 (3,8) region {1} backward", (int64_t[]){3, 8}, 2, (int32_t[]){1}, 1, 1,
               FFTL_PRECISION_DOUBLE, 1, 1e-12);
  check_region("2d f64 (1024,7) region {1} backward lanes", (int64_t[]){1024, 7}, 2,
               (int32_t[]){1}, 1, 1, FFTL_PRECISION_DOUBLE, 1, 1e-12);
  check_region("2d f32 (8,12) region {0,1}", (int64_t[]){8, 12}, 2, (int32_t[]){0, 1}, 2, -1,
               FFTL_PRECISION_SINGLE, 1, 1e-5);
}

static void test_3d(void) {
  int64_t dims[3] = {4, 6, 10};
  check_region("3d f64 (4,6,10) region {0,1,2}", dims, 3, (int32_t[]){0, 1, 2}, 3, -1,
               FFTL_PRECISION_DOUBLE, 1, 1e-12);
  check_region("3d f64 (4,6,10) region {0,2}", dims, 3, (int32_t[]){0, 2}, 2, -1,
               FFTL_PRECISION_DOUBLE, 1, 1e-12);
  check_region("3d f64 (4,6,10) region {1}", dims, 3, (int32_t[]){1}, 1, 1,
               FFTL_PRECISION_DOUBLE, 1, 1e-12);
  check_region("3d f32 (4,6,10) region {0,1,2}", dims, 3, (int32_t[]){0, 1, 2}, 3, -1,
               FFTL_PRECISION_SINGLE, 1, 1e-5);
  int64_t dims2[3] = {2, 3, 37};
  check_region("3d f64 (2,3,37) region {0,2} Bluestein", dims2, 3, (int32_t[]){0, 2}, 2, -1,
               FFTL_PRECISION_DOUBLE, 1, 1e-12);
}

static void test_empty_region(void) {
  int64_t dims[2] = {4, 5};
  int64_t nval = 20;
  fftl_plan p = fftl_plan_dft(2, dims, 0, NULL, FFTL_PRECISION_DOUBLE, -1, 1);
  CHECK(p >= 0, "empty region is a plan");
  if (p < 0) return;
  double *x = malloc(nval * sizeof(cpx)), *y = malloc(nval * sizeof(cpx));
  fill(x, nval);
  for (int64_t i = 0; i < nval; i++) y[i * 2] = 1e30;
  CHECK(fftl_plan_execute(p, y, x) == 0, "empty region executes");
  CHECK(memcmp(y, x, nval * sizeof(cpx)) == 0, "empty region copies out of place");
  double before = x[3];
  CHECK(fftl_plan_execute(p, x, x) == 0, "empty region executes in place");
  CHECK(x[3] == before, "empty region is a no-op in place");
  CHECK(fftl_plan_destroy(p) == 0, "destroy empty-region plan");
  free(x); free(y);
}

static void test_threads(void) {
  int64_t big[2] = {33, 1024};
  int64_t nval = 33 * 1024;
  int64_t one[1] = {131072};

  /* threading must give the same result as the single-threaded plan */
  double *x = malloc(nval * sizeof(cpx)), *y1 = malloc(nval * sizeof(cpx)),
         *y4 = malloc(nval * sizeof(cpx));
  fill(x, nval);
  fftl_plan p1 = fftl_plan_dft(2, big, 2, (int32_t[]){0, 1}, FFTL_PRECISION_DOUBLE, -1, 1);
  fftl_plan p4 = fftl_plan_dft(2, big, 2, (int32_t[]){0, 1}, FFTL_PRECISION_DOUBLE, -1, 4);
  CHECK(p1 >= 0 && p4 >= 0, "threads=1 and threads=4 plans");
  if (p1 < 0 || p4 < 0) return;
  CHECK(fftl_plan_execute(p1, y1, x) == 0, "2D t1 executes");
  CHECK(fftl_plan_execute(p4, y4, x) == 0, "2D t4 executes");
  CHECK(relerr(y4, y1, nval) < 1e-13, "2D threads=4 matches threads=1");
  CHECK(fftl_plan_destroy(p1) == 0 && fftl_plan_destroy(p4) == 0, "destroy thread plans");

  /* the threaded plan against the naive reference */
  check_region("2d f64 (33,1024) region {0,1} threads=4", big, 2, (int32_t[]){0, 1}, 2, -1,
               FFTL_PRECISION_DOUBLE, 4, 1e-13);
  check_region("3d f32 (64,64,8) region {0,1} threads=4", (int64_t[]){64, 64, 8}, 3,
               (int32_t[]){0, 1}, 2, -1, FFTL_PRECISION_SINGLE, 4, 1e-5);

  /* long 1D lines on several threads: the four-step path (smooth length >= 2^13) */
  check_roundtrip("1d f64 131072 roundtrip threads=4", one, 1, 1, (int32_t[]){0}, 4, 1e-14);
  check_roundtrip("1d f64 131072 roundtrip threads=1", one, 1, 1, (int32_t[]){0}, 1, 1e-14);
  {
    double *a = malloc(131072 * sizeof(cpx)), *b = malloc(131072 * sizeof(cpx)),
           *c = malloc(131072 * sizeof(cpx));
    fill(a, 131072);
    fftl_plan q1 = fftl_plan_dft(1, one, 1, (int32_t[]){0}, FFTL_PRECISION_DOUBLE, -1, 1);
    fftl_plan q4 = fftl_plan_dft(1, one, 1, (int32_t[]){0}, FFTL_PRECISION_DOUBLE, -1, 4);
    CHECK(q1 >= 0 && q4 >= 0, "1D thread plans");
    CHECK(fftl_plan_execute(q1, b, a) == 0 && fftl_plan_execute(q4, c, a) == 0,
          "1D thread plans execute");
    CHECK(relerr(c, b, 131072) < 1e-13, "1D threads=4 matches threads=1");
    CHECK(fftl_plan_destroy(q1) == 0 && fftl_plan_destroy(q4) == 0, "1D destroys");
    free(a); free(b); free(c);
  }
  check_roundtrip("3d f64 (16,16,16) roundtrip threads=4", (int64_t[]){16, 16, 16}, 3, 3,
                  (int32_t[]){0, 1, 2}, 4, 1e-14);
  check_roundtrip("1d f64 65536 roundtrip threads=1", (int64_t[]){65536}, 1, 1,
                  (int32_t[]){0}, 1, 1e-14);
  free(x); free(y1); free(y4);
}

static void test_errors(void) {
  int64_t dims3[3] = {8, 4, 2};
  int32_t reg2[2] = {0, 1};

  expect_error("precision 0", fftl_plan_dft(1, (int64_t[]){64}, 1, reg2, 0, -1, 1));
  expect_error("precision 3", fftl_plan_dft(1, (int64_t[]){64}, 1, reg2, 3, -1, 1));
  expect_error("sign 0", fftl_plan_dft(1, (int64_t[]){64}, 1, reg2, FFTL_PRECISION_DOUBLE, 0, 1));
  expect_error("sign 2", fftl_plan_dft(1, (int64_t[]){64}, 1, reg2, FFTL_PRECISION_DOUBLE, 2, 1));
  expect_error("rank 0", fftl_plan_dft(0, NULL, 0, NULL, FFTL_PRECISION_DOUBLE, -1, 1));
  expect_error("transform rank -1",
               fftl_plan_dft(1, (int64_t[]){64}, -1, reg2, FFTL_PRECISION_DOUBLE, -1, 1));
  expect_error("transform rank above rank",
               fftl_plan_dft(1, (int64_t[]){64}, 2, (int32_t[]){0, 1}, FFTL_PRECISION_DOUBLE, -1, 1));
  expect_error("size 0", fftl_plan_dft(1, (int64_t[]){0}, 1, reg2, FFTL_PRECISION_DOUBLE, -1, 1));
  expect_error("size -1", fftl_plan_dft(1, (int64_t[]){-1}, 1, reg2, FFTL_PRECISION_DOUBLE, -1, 1));
  expect_error("size 1 of 2d dims 0",
               fftl_plan_dft(2, (int64_t[]){8, 0}, 1, reg2, FFTL_PRECISION_DOUBLE, -1, 1));
  expect_error("region -1",
               fftl_plan_dft(2, (int64_t[]){8, 4}, 1, (int32_t[]){-1}, FFTL_PRECISION_DOUBLE, -1, 1));
  expect_error("region == rank",
               fftl_plan_dft(2, (int64_t[]){8, 4}, 1, (int32_t[]){2}, FFTL_PRECISION_DOUBLE, -1, 1));
  expect_error("region above rank",
               fftl_plan_dft(2, (int64_t[]){8, 4}, 1, (int32_t[]){3}, FFTL_PRECISION_DOUBLE, -1, 1));
  expect_error("region duplicate",
               fftl_plan_dft(3, dims3, 2, (int32_t[]){0, 0}, FFTL_PRECISION_DOUBLE, -1, 1));
  expect_error("region not ascending",
               fftl_plan_dft(3, dims3, 2, (int32_t[]){1, 0}, FFTL_PRECISION_DOUBLE, -1, 1));
  expect_error("region not ascending same ends",
               fftl_plan_dft(3, dims3, 3, (int32_t[]){0, 2, 1}, FFTL_PRECISION_DOUBLE, -1, 1));
  expect_error("threads 0",
               fftl_plan_dft(1, (int64_t[]){64}, 1, reg2, FFTL_PRECISION_DOUBLE, -1, 0));
  expect_error("threads -2",
               fftl_plan_dft(1, (int64_t[]){64}, 1, reg2, FFTL_PRECISION_DOUBLE, -1, -2));

  /* invalid handles */
  CHECK(fftl_plan_execute(-1, NULL, NULL) == -1, "execute handle -1");
  CHECK(fftl_plan_execute(0, NULL, NULL) == -1, "execute handle 0");
  CHECK(fftl_plan_execute((1ll << 32) | 99, NULL, NULL) == -1, "execute uncreated index");
  CHECK(fftl_plan_execute((3ll << 32) | 0, NULL, NULL) == -1, "execute bad precision handle");
  CHECK(fftl_plan_destroy(-1) == -1, "destroy handle -1");
  CHECK(fftl_plan_destroy(0) == -1, "destroy handle 0");

  /* a double destroy fails */
  fftl_plan p = must_plan("plan for double destroy", 1, (int64_t[]){64}, 1, reg2,
                          FFTL_PRECISION_DOUBLE, -1, 1);
  CHECK(fftl_plan_destroy(p) == 0, "first destroy");
  CHECK(fftl_plan_destroy(p) == -1, "second destroy rejected");

  /* the registry holds many plans of both precisions at once */
  fftl_plan plans[64];
  for (int i = 0; i < 64; i++)
    plans[i] = fftl_plan_dft(1, (int64_t[]){256}, 1, reg2,
                             i % 2 ? FFTL_PRECISION_SINGLE : FFTL_PRECISION_DOUBLE, -1, 1);
  for (int i = 0; i < 64; i++) CHECK(plans[i] >= 0, "64 plans created");
  for (int i = 0; i < 64; i++) CHECK(plans[i] < 0 || fftl_plan_destroy(plans[i]) == 0,
                                     "64 plans destroyed");
}

static void test_plan_reuse(void) {
  int64_t d1[1] = {1000};
  int64_t d2[2] = {8, 12};
  int64_t d3[1] = {101};
  fftl_plan ps[3];
  ps[0] = must_plan("reuse 1000", 1, d1, 1, (int32_t[]){0}, FFTL_PRECISION_DOUBLE, -1, 1);
  ps[1] = must_plan("reuse 2d", 2, d2, 2, (int32_t[]){0, 1}, FFTL_PRECISION_DOUBLE, -1, 1);
  ps[2] = must_plan("reuse 101", 1, d3, 1, (int32_t[]){0}, FFTL_PRECISION_SINGLE, 1, 1);
  double *a = malloc(1000 * sizeof(cpx)), *b = malloc(1000 * sizeof(cpx));
  fill(a, 1000);
  CHECK(fftl_plan_execute(ps[0], b, a) == 0, "1000 executes");
  float *f = malloc(101 * 8);
  for (int i = 0; i < 202; i++) f[i] = (float) i;
  float *g = malloc(101 * 8);
  CHECK(fftl_plan_execute(ps[2], g, f) == 0, "101 f32 executes");
  CHECK(fftl_plan_destroy(ps[0]) == 0 && fftl_plan_destroy(ps[1]) == 0 &&
        fftl_plan_destroy(ps[2]) == 0, "reuse destroys");
  free(a); free(b); free(f); free(g);
}

int main(void) {
  test_version_and_error();
  test_1d_double();
  test_1d_single();
  test_2d();
  test_3d();
  test_empty_region();
  test_threads();
  test_errors();
  test_plan_reuse();

  printf("libfftl tests: %d checks, %d failed\n", ntests, nfailed);
  return nfailed == 0 ? 0 : 1;
}
