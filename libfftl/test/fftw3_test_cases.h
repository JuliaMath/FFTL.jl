/* The cases of fftw3_test.c for one precision, included once per precision with
 * PREC (64 or 32), R (double or float), X(name) (fftw_name or fftwf_name), TOL and RELERR. */

#define CAT_(a, b) a##_##b
#define CAT(a, b) CAT_(a, b)
#define FN(name) CAT(name, PREC)
#define CPX X(complex)

/* fills count real values from the generator, keeps a long double copy in ref (if non-NULL) */
static void FN(fill)(R *a, LD *ref, long count) {
  for (long i = 0; i < count; i++) {
    a[i] = (R) urand();
    if (ref) ref[i] = a[i];
  }
}

static void FN(name)(char *buf, size_t len, const char *what, int rank, const int *n, const char *extra) {
  int k = snprintf(buf, len, "%d_%s", PREC, what);
  for (int i = 0; i < rank && k < (int) len; i++) k += snprintf(buf + k, len - k, "_%d", n[i]);
  if (extra && k < (int) len) snprintf(buf + k, len - k, "_%s", extra);
}

static unsigned FN(flags)(int i) {
  static const unsigned f[] = {FFTW_ESTIMATE, FFTW_MEASURE, FFTW_ESTIMATE | FFTW_UNALIGNED,
                               FFTW_MEASURE | FFTW_DESTROY_INPUT, FFTW_PATIENT};
  return f[i % 5];
}

/* ------------------------------------------------------------- complex DFT, basic interface */

static void FN(c2c_basic)(int rank, const int *n, int sign, int inplace, int via) {
  long N = prod(rank, n);
  CPX *in = X(alloc_complex)(N), *out = inplace ? in : X(alloc_complex)(N);
  unsigned fl = FN(flags)(rank + via + inplace);
  X(plan) p = NULL;
  if (via == 0 && rank == 1) p = X(plan_dft_1d)(n[0], in, out, sign, fl);
  else if (via == 0 && rank == 2) p = X(plan_dft_2d)(n[0], n[1], in, out, sign, fl);
  else if (via == 0 && rank == 3) p = X(plan_dft_3d)(n[0], n[1], n[2], in, out, sign, fl);
  else p = X(plan_dft)(rank, n, in, out, sign, fl);
  char nm[256];
  FN(name)(nm, sizeof nm, sign < 0 ? "c2c_fwd" : "c2c_bwd", rank, n, inplace ? (via ? "ip_rank" : "ip") : (via ? "oop_rank" : "oop"));
  CHECK(p != NULL, "%s: plan", nm);
  if (!p) return;
  LD *ref = malloc(2 * N * sizeof(LD));
  FN(fill)((R *) in, ref, 2 * N);           /* after planning: FFTW_MEASURE may overwrite */
  X(execute)(p);
  for (int d = 0; d < rank; d++) ref_dft_dim(ref, rank, n, d, sign);
  double e = RELERR((R *) out, ref, 2 * N);
  CHECK(e < TOL, "%s: rel err %g", nm, e);
  dump(nm, out, 2 * N * sizeof(R));
  X(destroy_plan)(p);
  X(free)(in);
  if (!inplace) X(free)(out);
  free(ref);
}

/* ------------------------------------------------------------- r2c and c2r, basic interface */

static void FN(real_basic)(int rank, const int *n, int inplace, int via) {
  long N = prod(rank, n), last = n[rank - 1], h = last / 2 + 1, outer = N / last;
  long nc = outer * h;                      /* complex values */
  long rphys = inplace ? 2 * h : last;      /* physical length of the real lines */
  R *x = X(alloc_real)(inplace ? 2 * nc : N);
  CPX *y = inplace ? (CPX *) x : X(alloc_complex)(nc);
  unsigned fl = FN(flags)(rank + 2 * inplace + via);
  X(plan) p;
  if (via == 0 && rank == 1) p = X(plan_dft_r2c_1d)(n[0], x, y, fl);
  else if (via == 0 && rank == 2) p = X(plan_dft_r2c_2d)(n[0], n[1], x, y, fl);
  else if (via == 0 && rank == 3) p = X(plan_dft_r2c_3d)(n[0], n[1], n[2], x, y, fl);
  else p = X(plan_dft_r2c)(rank, n, x, y, fl);
  char nm[256];
  FN(name)(nm, sizeof nm, "r2c", rank, n, inplace ? (via ? "ip_rank" : "ip") : (via ? "oop_rank" : "oop"));
  CHECK(p != NULL, "%s: plan", nm);
  if (!p) return;
  LD *xr = malloc(N * sizeof(LD)), *half = malloc(2 * nc * sizeof(LD));
  for (long o = 0; o < outer; o++)
    for (long j = 0; j < last; j++) {
      x[o * rphys + j] = (R) urand();
      xr[o * last + j] = x[o * rphys + j];
    }
  X(execute)(p);
  ref_r2c(xr, rank, n, half);
  double e = RELERR((R *) y, half, 2 * nc);
  CHECK(e < TOL, "%s: rel err %g", nm, e);
  dump(nm, y, 2 * nc * sizeof(R));
  X(destroy_plan)(p);
  /* c2r of the exact half spectrum gives N x */
  FN(name)(nm, sizeof nm, "c2r", rank, n, inplace ? (via ? "ip_rank" : "ip") : (via ? "oop_rank" : "oop"));
  R *xo = inplace ? x : X(alloc_real)(N);
  if (via == 0 && rank == 1) p = X(plan_dft_c2r_1d)(n[0], y, xo, fl);
  else if (via == 0 && rank == 2) p = X(plan_dft_c2r_2d)(n[0], n[1], y, xo, fl);
  else if (via == 0 && rank == 3) p = X(plan_dft_c2r_3d)(n[0], n[1], n[2], y, xo, fl);
  else p = X(plan_dft_c2r)(rank, n, y, xo, fl);
  CHECK(p != NULL, "%s: plan", nm);
  if (p) {
    for (long i = 0; i < 2 * nc; i++) ((R *) y)[i] = (R) half[i];
    X(execute)(p);
    R *compact = malloc(N * sizeof(R));
    LD *want = malloc(N * sizeof(LD));
    for (long o = 0; o < outer; o++)
      for (long j = 0; j < last; j++) {
        compact[o * last + j] = xo[o * rphys + j];
        want[o * last + j] = N * xr[o * last + j];
      }
    e = RELERR(compact, want, N);
    CHECK(e < TOL, "%s: rel err %g", nm, e);
    dump(nm, compact, N * sizeof(R));
    free(compact);
    free(want);
    X(destroy_plan)(p);
  }
  if (!inplace) {
    X(free)(xo);
    X(free)(y);
  }
  X(free)(x);
  free(xr);
  free(half);
}

/* ------------------------------------------------------------------- r2r, basic interface */

static void FN(r2r_basic)(int rank, const int *n, const X(r2r_kind) *kind, int inplace, int via) {
  long N = prod(rank, n);
  R *in = X(alloc_real)(N), *out = inplace ? in : X(alloc_real)(N);
  unsigned fl = FN(flags)(rank + via + 3 * inplace + kind[0]);
  X(plan) p;
  if (via == 0 && rank == 1) p = X(plan_r2r_1d)(n[0], in, out, kind[0], fl);
  else if (via == 0 && rank == 2) p = X(plan_r2r_2d)(n[0], n[1], in, out, kind[0], kind[1], fl);
  else if (via == 0 && rank == 3) p = X(plan_r2r_3d)(n[0], n[1], n[2], in, out, kind[0], kind[1], kind[2], fl);
  else p = X(plan_r2r)(rank, n, in, out, kind, fl);
  char nm[256], ks[64];
  int k = 0;
  for (int i = 0; i < rank; i++) k += snprintf(ks + k, sizeof ks - k, "k%d", (int) kind[i]);
  snprintf(ks + k, sizeof ks - k, "_%s", inplace ? (via ? "ip_rank" : "ip") : (via ? "oop_rank" : "oop"));
  FN(name)(nm, sizeof nm, "r2r", rank, n, ks);
  CHECK(p != NULL, "%s: plan", nm);
  if (!p) return;
  LD *ref = malloc(N * sizeof(LD));
  FN(fill)(in, ref, N);
  X(execute)(p);
  for (int d = 0; d < rank; d++) ref_r2r_dim(ref, rank, n, d, kind[d]);
  double e = RELERR(out, ref, N);
  CHECK(e < TOL, "%s: rel err %g", nm, e);
  dump(nm, out, N * sizeof(R));
  X(destroy_plan)(p);
  X(free)(in);
  if (!inplace) X(free)(out);
  free(ref);
}

/* --------------------------------------------------------------------- advanced interface */

static void FN(many)(void) {
  char nm[256];
  /* c2c: 3 arrays of 5 x 6, one after the other, embeds equal to n */
  {
    int n[2] = {5, 6}, howmany = 3, full[3] = {3, 5, 6};
    long N = 30;
    CPX *in = X(alloc_complex)(howmany * N), *out = X(alloc_complex)(howmany * N);
    X(plan) p = X(plan_many_dft)(2, n, howmany, in, n, 1, N, out, n, 1, N, FFTW_FORWARD, FFTW_MEASURE);
    snprintf(nm, sizeof nm, "%d_many_dft", PREC);
    CHECK(p != NULL, "%s: plan", nm);
    if (p) {
      LD *ref = malloc(2 * howmany * N * sizeof(LD));
      FN(fill)((R *) in, ref, 2 * howmany * N);
      X(execute)(p);
      ref_dft_dim(ref, 3, full, 1, FFTW_FORWARD);
      ref_dft_dim(ref, 3, full, 2, FFTW_FORWARD);
      double e = RELERR((R *) out, ref, 2 * howmany * N);
      CHECK(e < TOL, "%s: rel err %g", nm, e);
      dump(nm, out, 2 * howmany * N * sizeof(R));
      free(ref);
      X(destroy_plan)(p);
    }
    X(free)(in);
    X(free)(out);
  }
  /* the manual's example: the 3 columns of a 10 x 3 row-major array, backward */
  {
    int n[1] = {10}, full[2] = {10, 3};
    CPX *in = X(alloc_complex)(30), *out = X(alloc_complex)(30);
    X(plan) p = X(plan_many_dft)(1, n, 3, in, n, 3, 1, out, n, 3, 1, FFTW_BACKWARD, FFTW_ESTIMATE);
    snprintf(nm, sizeof nm, "%d_many_dft_columns", PREC);
    CHECK(p != NULL, "%s: plan", nm);
    if (p) {
      LD *ref = malloc(60 * sizeof(LD));
      FN(fill)((R *) in, ref, 60);
      X(execute)(p);
      ref_dft_dim(ref, 2, full, 0, FFTW_BACKWARD);
      double e = RELERR((R *) out, ref, 60);
      CHECK(e < TOL, "%s: rel err %g", nm, e);
      dump(nm, out, 60 * sizeof(R));
      free(ref);
      X(destroy_plan)(p);
    }
    X(free)(in);
    X(free)(out);
  }
  /* r2c in place: 4 arrays of 6 x 7 (padded to 6 x 8), NULL embeds */
  {
    int n[2] = {6, 7}, howmany = 4;
    long h = 4, rd = 6 * 2 * h, cd = 6 * h;
    R *x = X(alloc_real)(howmany * rd);
    X(plan) p = X(plan_many_dft_r2c)(2, n, howmany, x, NULL, 1, rd, (CPX *) x, NULL, 1, cd, FFTW_ESTIMATE);
    snprintf(nm, sizeof nm, "%d_many_r2c_ip", PREC);
    CHECK(p != NULL, "%s: plan", nm);
    if (p) {
      LD *xr = malloc(howmany * 42 * sizeof(LD)), *half = malloc(howmany * 2 * cd * sizeof(LD));
      for (long m = 0; m < howmany; m++)
        for (long o = 0; o < 6; o++)
          for (long j = 0; j < 7; j++) {
            x[m * rd + o * 2 * h + j] = (R) urand();
            xr[m * 42 + o * 7 + j] = x[m * rd + o * 2 * h + j];
          }
      X(execute)(p);
      for (long m = 0; m < howmany; m++) ref_r2c(xr + m * 42, 2, n, half + m * 2 * cd);
      double e = RELERR(x, half, howmany * 2 * cd);
      CHECK(e < TOL, "%s: rel err %g", nm, e);
      dump(nm, x, howmany * 2 * cd * sizeof(R));
      free(xr);
      free(half);
      X(destroy_plan)(p);
    }
    X(free)(x);
  }
  /* r2r: 3 lines of 9, DST-III, out of place */
  {
    int n[1] = {9}, full[2] = {3, 9};
    X(r2r_kind) k[1] = {FFTW_RODFT01};
    R *in = X(alloc_real)(27), *out = X(alloc_real)(27);
    X(plan) p = X(plan_many_r2r)(1, n, 3, in, NULL, 1, 9, out, NULL, 1, 9, k, FFTW_MEASURE);
    snprintf(nm, sizeof nm, "%d_many_r2r", PREC);
    CHECK(p != NULL, "%s: plan", nm);
    if (p) {
      LD *ref = malloc(27 * sizeof(LD));
      FN(fill)(in, ref, 27);
      X(execute)(p);
      ref_r2r_dim(ref, 2, full, 1, FFTW_RODFT01);
      double e = RELERR(out, ref, 27);
      CHECK(e < TOL, "%s: rel err %g", nm, e);
      dump(nm, out, 27 * sizeof(R));
      free(ref);
      X(destroy_plan)(p);
    }
    X(free)(in);
    X(free)(out);
  }
}

/* ------------------------------------------------------------------------ guru interface */

static void FN(guru)(void) {
  char nm[256];
  /* a 4 x 5 x 6 row-major array, transformed along dimensions 2 and 0 (in that order), with
   * dimension 1 as the vector loop */
  int full[3] = {4, 5, 6};
  {
    X(iodim) dims[2] = {{6, 1, 1}, {4, 30, 30}}, vec[1] = {{5, 6, 6}};
    CPX *in = X(alloc_complex)(120), *out = X(alloc_complex)(120);
    X(plan) p = X(plan_guru_dft)(2, dims, 1, vec, in, out, FFTW_FORWARD, FFTW_MEASURE);
    snprintf(nm, sizeof nm, "%d_guru_dft", PREC);
    CHECK(p != NULL, "%s: plan", nm);
    if (p) {
      LD *ref = malloc(240 * sizeof(LD));
      FN(fill)((R *) in, ref, 240);
      X(execute)(p);
      ref_dft_dim(ref, 3, full, 0, FFTW_FORWARD);
      ref_dft_dim(ref, 3, full, 2, FFTW_FORWARD);
      double e = RELERR((R *) out, ref, 240);
      CHECK(e < TOL, "%s: rel err %g", nm, e);
      dump(nm, out, 240 * sizeof(R));
      free(ref);
      X(destroy_plan)(p);
    }
    X(free)(in);
    X(free)(out);
  }
  {
    X(iodim64) dims[2] = {{4, 30, 30}, {6, 1, 1}}, vec[1] = {{5, 6, 6}};
    X(r2r_kind) k[2] = {FFTW_REDFT11, FFTW_DHT};
    R *in = X(alloc_real)(120);
    X(plan) p = X(plan_guru64_r2r)(2, dims, 1, vec, in, in, k, FFTW_ESTIMATE);
    snprintf(nm, sizeof nm, "%d_guru64_r2r_ip", PREC);
    CHECK(p != NULL, "%s: plan", nm);
    if (p) {
      LD *ref = malloc(120 * sizeof(LD));
      FN(fill)(in, ref, 120);
      X(execute)(p);
      ref_r2r_dim(ref, 3, full, 0, FFTW_REDFT11);
      ref_r2r_dim(ref, 3, full, 2, FFTW_DHT);
      double e = RELERR(in, ref, 120);
      CHECK(e < TOL, "%s: rel err %g", nm, e);
      dump(nm, in, 120 * sizeof(R));
      free(ref);
      X(destroy_plan)(p);
    }
    X(free)(in);
  }
  /* r2c over dimensions 0 and 2 of a 3 x 4 x 10 real array (halved: dimension 2), dimension 1
   * as the vector loop; complex array 3 x 4 x 6 */
  {
    int rn[3] = {3, 4, 10};
    X(iodim) dims[2] = {{3, 40, 24}, {10, 1, 1}}, vec[1] = {{4, 10, 6}};
    R *x = X(alloc_real)(120);
    CPX *y = X(alloc_complex)(72);
    X(plan) p = X(plan_guru_dft_r2c)(2, dims, 1, vec, x, y, FFTW_ESTIMATE);
    snprintf(nm, sizeof nm, "%d_guru_r2c", PREC);
    CHECK(p != NULL, "%s: plan", nm);
    if (p) {
      LD *a = malloc(240 * sizeof(LD)), *half = malloc(144 * sizeof(LD));
      for (long i = 0; i < 120; i++) {
        x[i] = (R) urand();
        a[2 * i] = x[i];
        a[2 * i + 1] = 0;
      }
      X(execute)(p);
      ref_dft_dim(a, 3, rn, 0, FFTW_FORWARD);
      ref_dft_dim(a, 3, rn, 2, FFTW_FORWARD);
      for (long o = 0; o < 12; o++)
        for (long k = 0; k < 6; k++) {
          half[2 * (o * 6 + k)] = a[2 * (o * 10 + k)];
          half[2 * (o * 6 + k) + 1] = a[2 * (o * 10 + k) + 1];
        }
      double e = RELERR((R *) y, half, 144);
      CHECK(e < TOL, "%s: rel err %g", nm, e);
      dump(nm, y, 144 * sizeof(R));
      free(a);
      free(half);
      X(destroy_plan)(p);
    }
    X(free)(x);
    X(free)(y);
  }
}

/* -------------------------------------------- strided, embedded and split layouts */

/* An array in a user layout: element o (in FFTW's units: complex numbers for interleaved
 * arrays, real numbers for real and split arrays) has its real part at re[mult * o] and its
 * imaginary part at im[mult * o]. Interleaved: im = re + 1, mult 2; split: mult 1; real: im
 * NULL, mult 1. */
typedef struct {
  R *re, *im;
  long mult;
} FN(arr);

/* The parameters of plan_many for a case planned through the advanced interface */
typedef struct {
  int rank, howmany, istride, idist, ostride, odist;
  const int *n, *inembed, *onembed;
} FN(manyp);

/* One transform in a user layout, against the reference in long double. k logical dimensions
 * of sizes n (the real sizes for r2c/c2r), tdim[0..rank-1] the transform dimensions in FFTW's
 * order (the last one is halved for r2c/c2r), the others vector dimensions; is/os the strides
 * per logical dimension. typ 0 c2c (forward), 1 r2c, 2 c2r, 3 r2r. Planned through guru64, or
 * through plan_many if many is not NULL; split arrays through the split planners. */
static void FN(layout_case)(const char *nm, int typ, int k, const int *n, int rank, const int *tdim,
                            const long *is, const long *os, FN(arr) in, FN(arr) out, int split,
                            const X(r2r_kind) *kinds, const FN(manyp) *many) {
  int cin[8], cout[8], hd = rank > 0 ? tdim[rank - 1] : -1;
  for (int d = 0; d < k; d++) {
    int h = (d == hd && (typ == 1 || typ == 2)) ? n[d] / 2 + 1 : n[d];
    cin[d] = typ == 2 ? h : n[d];
    cout[d] = typ == 1 ? h : n[d];
  }
  long N = prod(k, n), Nin = prod(k, cin), Nout = prod(k, cout);
  X(iodim64) td[8], vd[8];
  int nv = 0;
  for (int j = 0; j < rank; j++) td[j] = (X(iodim64)){n[tdim[j]], is[tdim[j]], os[tdim[j]]};
  for (int d = 0; d < k; d++) {
    int t = 0;
    for (int j = 0; j < rank; j++) t |= tdim[j] == d;
    if (!t) vd[nv++] = (X(iodim64)){n[d], is[d], os[d]};
  }
  X(plan) p = NULL;
  void *i1 = in.re, *o1 = out.re;
  if (many) {
    const FN(manyp) *m = many;
    if (typ == 0) p = X(plan_many_dft)(m->rank, m->n, m->howmany, i1, m->inembed, m->istride, m->idist, o1, m->onembed, m->ostride, m->odist, FFTW_FORWARD, FFTW_ESTIMATE);
    else if (typ == 1) p = X(plan_many_dft_r2c)(m->rank, m->n, m->howmany, i1, m->inembed, m->istride, m->idist, o1, m->onembed, m->ostride, m->odist, FFTW_ESTIMATE);
    else if (typ == 2) p = X(plan_many_dft_c2r)(m->rank, m->n, m->howmany, i1, m->inembed, m->istride, m->idist, o1, m->onembed, m->ostride, m->odist, FFTW_ESTIMATE);
    else p = X(plan_many_r2r)(m->rank, m->n, m->howmany, i1, m->inembed, m->istride, m->idist, o1, m->onembed, m->ostride, m->odist, kinds, FFTW_ESTIMATE);
  } else if (split) {
    if (typ == 0) p = X(plan_guru64_split_dft)(rank, td, nv, vd, in.re, in.im, out.re, out.im, FFTW_ESTIMATE);
    else if (typ == 1) p = X(plan_guru64_split_dft_r2c)(rank, td, nv, vd, in.re, out.re, out.im, FFTW_ESTIMATE);
    else p = X(plan_guru64_split_dft_c2r)(rank, td, nv, vd, in.re, in.im, out.re, FFTW_ESTIMATE);
  } else {
    if (typ == 0) p = X(plan_guru64_dft)(rank, td, nv, vd, i1, o1, FFTW_FORWARD, FFTW_ESTIMATE);
    else if (typ == 1) p = X(plan_guru64_dft_r2c)(rank, td, nv, vd, i1, o1, FFTW_ESTIMATE);
    else if (typ == 2) p = X(plan_guru64_dft_c2r)(rank, td, nv, vd, i1, o1, FFTW_ESTIMATE);
    else p = X(plan_guru64_r2r)(rank, td, nv, vd, i1, o1, kinds, FFTW_ESTIMATE);
  }
  CHECK(p != NULL, "%s: plan", nm);
  if (!p) return;
  /* input (after planning), reference */
  LD *ref = malloc(2 * (N > Nout ? N : Nout) * sizeof(LD)), *a = malloc(2 * N * sizeof(LD));
  LD *xr = malloc(N * sizeof(LD));
  for (long i = 0; i < N; i++) xr[i] = (R) urand();
  if (typ == 0) {
    for (long i = 0; i < N; i++) {
      R re = (R) urand(), im = (R) urand();
      long o = in.mult * lay_off(k, cin, is, i);
      in.re[o] = re;
      in.im[o] = im;
      a[2 * i] = re;
      a[2 * i + 1] = im;
    }
    for (int j = 0; j < rank; j++) ref_dft_dim(a, k, n, tdim[j], FFTW_FORWARD);
    memcpy(ref, a, 2 * N * sizeof(LD));
  } else if (typ == 1 || typ == 2) {
    /* the half spectrum of xr along the transform dimensions */
    for (long i = 0; i < N; i++) {
      a[2 * i] = xr[i];
      a[2 * i + 1] = 0;
    }
    for (int j = 0; j < rank; j++) ref_dft_dim(a, k, n, tdim[j], FFTW_FORWARD);
    LD *half = typ == 1 ? ref : a;
    if (typ == 2) {
      half = malloc(2 * Nin * sizeof(LD));
      for (long i = 0; i < Nin; i++) {
        long f = lay_remap(k, cin, n, i);
        half[2 * i] = (R) a[2 * f];
        half[2 * i + 1] = (R) a[2 * f + 1];
        long o = in.mult * lay_off(k, cin, is, i);
        in.re[o] = (R) half[2 * i];
        in.im[o] = (R) half[2 * i + 1];
      }
      /* c2r of the rounded half spectrum: compare with the exact inverse of it */
      LD *b = malloc(2 * N * sizeof(LD));
      for (long i = 0; i < N; i++) b[2 * i] = b[2 * i + 1] = 0;
      for (long i = 0; i < Nin; i++) {
        long f = lay_remap(k, cin, n, i);
        b[2 * f] = half[2 * i];
        b[2 * f + 1] = half[2 * i + 1];
      }
      /* fill the other half by Hermitian symmetry along all transform dimensions: the
       * mirrored index of every transform dimension */
      for (long i = 0; i < N; i++) {
        int dig[8];
        long r = i;
        for (int d = k - 1; d >= 0; d--) {
          dig[d] = r % n[d];
          r /= n[d];
        }
        if (dig[hd] < cin[hd]) continue;
        int mir[8];
        for (int d = 0; d < k; d++) mir[d] = dig[d];
        for (int j = 0; j < rank; j++) mir[tdim[j]] = (n[tdim[j]] - dig[tdim[j]]) % n[tdim[j]];
        long m = 0;
        for (int d = 0; d < k; d++) m = m * n[d] + mir[d];
        b[2 * i] = b[2 * m];
        b[2 * i + 1] = -b[2 * m + 1];
      }
      for (int j = 0; j < rank; j++) ref_dft_dim(b, k, n, tdim[j], FFTW_BACKWARD);
      for (long i = 0; i < N; i++) ref[i] = b[2 * i];
      free(b);
      free(half);
    } else {
      for (long i = 0; i < Nout; i++) {
        long f = lay_remap(k, cout, n, i);
        ref[2 * i] = a[2 * f];
        ref[2 * i + 1] = a[2 * f + 1];
      }
      for (long i = 0; i < N; i++) in.re[in.mult * lay_off(k, cin, is, i)] = (R) xr[i];
    }
  } else {
    for (long i = 0; i < N; i++) {
      in.re[in.mult * lay_off(k, cin, is, i)] = (R) xr[i];
      ref[i] = (R) xr[i];
    }
    for (int j = 0; j < rank; j++) ref_r2r_dim(ref, k, n, tdim[j], kinds[j]);
  }
  X(execute)(p);
  /* the output, gathered from its layout */
  int cplx = typ == 0 || typ == 1;
  R *y = malloc((cplx ? 2 : 1) * Nout * sizeof(R));
  for (long i = 0; i < Nout; i++) {
    long o = out.mult * lay_off(k, cout, os, i);
    if (cplx) {
      y[2 * i] = out.re[o];
      y[2 * i + 1] = out.im[o];
    } else {
      y[i] = out.re[o];
    }
  }
  double e = RELERR(y, ref, (cplx ? 2 : 1) * Nout);
  CHECK(e < TOL, "%s: rel err %g", nm, e);
  dump(nm, y, (cplx ? 2 : 1) * Nout * sizeof(R));
  X(destroy_plan)(p);
  free(y);
  free(ref);
  free(a);
  free(xr);
}

/* Strides for dimensions taken in the order ord (fastest first), with gap extra elements after
 * each: st[ord[0]] = s0, then times (counts + gap). Returns the number of elements spanned. */
static long FN(mkstrides)(int k, const int *counts, const int *ord, const int *gap, long s0, long *st) {
  long acc = s0;
  for (int j = 0; j < k; j++) {
    st[ord[j]] = acc;
    acc *= counts[ord[j]] + gap[j];
  }
  return acc;
}

static FN(arr) FN(inter)(CPX *p) { return (FN(arr)){(R *) p, (R *) p + 1, 2}; }
static FN(arr) FN(realarr)(R *p) { return (FN(arr)){p, NULL, 1}; }
static FN(arr) FN(splitarr)(R *re, R *im) { return (FN(arr)){re, im, 1}; }

static void FN(layouts)(void) {
  char nm[256];
  /* c2c, 4 x 6 x 5, transformed along 2 and 0, 1 a vector dimension; input and output in
   * different orders, both with gaps */
  {
    int n[3] = {4, 6, 5}, td[2] = {2, 0}, oi[3] = {1, 2, 0}, gi[3] = {1, 0, 2}, oo[3] = {0, 1, 2}, go[3] = {0, 2, 1};
    long is[3], os[3];
    long li = FN(mkstrides)(3, n, oi, gi, 1, is), lo = FN(mkstrides)(3, n, oo, go, 1, os);
    CPX *x = X(alloc_complex)(li), *y = X(alloc_complex)(lo);
    memset(y, 0, lo * sizeof(CPX));
    snprintf(nm, sizeof nm, "%d_layout_c2c_gaps_transposed", PREC);
    FN(layout_case)(nm, 0, 3, n, 2, td, is, os, FN(inter)(x), FN(inter)(y), 0, NULL, NULL);
    /* the same with split arrays (strides in real numbers) */
    R *xr = X(alloc_real)(li), *xi = X(alloc_real)(li), *yr = X(alloc_real)(lo), *yi = X(alloc_real)(lo);
    snprintf(nm, sizeof nm, "%d_layout_split_c2c", PREC);
    FN(layout_case)(nm, 0, 3, n, 2, td, is, os, FN(splitarr)(xr, xi), FN(splitarr)(yr, yi), 1, NULL, NULL);
    /* split pointers into interleaved arrays, real and imaginary parts swapped (FFTW's way to a
     * backward transform) */
    long is2[3], os2[3];
    for (int d = 0; d < 3; d++) {
      is2[d] = 2 * is[d];
      os2[d] = 2 * os[d];
    }
    snprintf(nm, sizeof nm, "%d_layout_split_c2c_swapped", PREC);
    FN(layout_case)(nm, 0, 3, n, 2, td, is2, os2, FN(splitarr)((R *) x + 1, (R *) x), FN(splitarr)((R *) y + 1, (R *) y), 1, NULL, NULL);
    X(free)(x);
    X(free)(y);
    X(free)(xr);
    X(free)(xi);
    X(free)(yr);
    X(free)(yi);
  }
  /* c2c in place with gaps: 8 x 3, transformed along 0 */
  {
    int n[2] = {8, 3}, td[1] = {0}, o[2] = {0, 1}, g[2] = {2, 1};
    long st[2];
    long l = FN(mkstrides)(2, n, o, g, 1, st);
    CPX *x = X(alloc_complex)(l);
    snprintf(nm, sizeof nm, "%d_layout_c2c_inplace_gaps", PREC);
    FN(layout_case)(nm, 0, 2, n, 1, td, st, st, FN(inter)(x), FN(inter)(x), 0, NULL, NULL);
    /* split, in place, rank 1: an interleaved array as split arrays */
    int n1[1] = {12}, t1[1] = {0};
    long s1[1] = {2};
    snprintf(nm, sizeof nm, "%d_layout_split_c2c_inplace", PREC);
    FN(layout_case)(nm, 0, 1, n1, 1, t1, s1, s1, FN(splitarr)((R *) x, (R *) x + 1), FN(splitarr)((R *) x, (R *) x + 1), 1, NULL, NULL);
    X(free)(x);
  }
  /* r2c / c2r with the halved dimension the slowest: 6 x 10 x 3, transformed along 1 and 0
   * (0 halved), 2 a vector dimension */
  {
    int n[3] = {6, 10, 3}, h[3] = {4, 10, 3}, td[2] = {1, 0}, oi[3] = {2, 1, 0}, gi[3] = {0, 1, 0}, oo[3] = {1, 2, 0}, go[3] = {1, 0, 2};
    long rs[3], cs[3];
    long lr = FN(mkstrides)(3, n, oi, gi, 1, rs), lc = FN(mkstrides)(3, h, oo, go, 1, cs);
    R *x = X(alloc_real)(lr), *z = X(alloc_real)(lr);
    CPX *y = X(alloc_complex)(lc);
    snprintf(nm, sizeof nm, "%d_layout_r2c_halved_slowest", PREC);
    FN(layout_case)(nm, 1, 3, n, 2, td, rs, cs, FN(realarr)(x), FN(inter)(y), 0, NULL, NULL);
    snprintf(nm, sizeof nm, "%d_layout_c2r_halved_slowest", PREC);
    FN(layout_case)(nm, 2, 3, n, 2, td, cs, rs, FN(inter)(y), FN(realarr)(z), 0, NULL, NULL);
    /* split complex side */
    R *yr = X(alloc_real)(lc), *yi = X(alloc_real)(lc);
    snprintf(nm, sizeof nm, "%d_layout_split_r2c", PREC);
    FN(layout_case)(nm, 1, 3, n, 2, td, rs, cs, FN(realarr)(x), FN(splitarr)(yr, yi), 1, NULL, NULL);
    snprintf(nm, sizeof nm, "%d_layout_split_c2r", PREC);
    FN(layout_case)(nm, 2, 3, n, 2, td, cs, rs, FN(splitarr)(yr, yi), FN(realarr)(z), 1, NULL, NULL);
    X(free)(x);
    X(free)(z);
    X(free)(y);
    X(free)(yr);
    X(free)(yi);
  }
  /* r2c / c2r in place: 5 x 7 (7 halved, padded), a gap of one complex number per row */
  {
    int n[2] = {5, 7}, h[2] = {5, 4}, td[2] = {0, 1}, o[2] = {1, 0}, g[2] = {1, 0};
    long cs[2], rs[2];
    long lc = FN(mkstrides)(2, h, o, g, 1, cs);
    rs[1] = 1;
    rs[0] = 2 * cs[0];
    CPX *b = X(alloc_complex)(lc);
    snprintf(nm, sizeof nm, "%d_layout_r2c_inplace_gaps", PREC);
    FN(layout_case)(nm, 1, 2, n, 2, td, rs, cs, FN(realarr)((R *) b), FN(inter)(b), 0, NULL, NULL);
    snprintf(nm, sizeof nm, "%d_layout_c2r_inplace_gaps", PREC);
    FN(layout_case)(nm, 2, 2, n, 2, td, cs, rs, FN(inter)(b), FN(realarr)((R *) b), 0, NULL, NULL);
    X(free)(b);
  }
  /* r2r: 5 x 6 x 4 along 0 and 2, different layouts; 7 x 4 in place with gaps */
  {
    int n[3] = {5, 6, 4}, td[2] = {0, 2}, oi[3] = {2, 0, 1}, gi[3] = {0, 2, 1}, oo[3] = {1, 2, 0}, go[3] = {1, 1, 0};
    X(r2r_kind) k[2] = {FFTW_REDFT10, FFTW_RODFT11};
    long is[3], os[3];
    long li = FN(mkstrides)(3, n, oi, gi, 1, is), lo = FN(mkstrides)(3, n, oo, go, 1, os);
    R *x = X(alloc_real)(li), *y = X(alloc_real)(lo);
    snprintf(nm, sizeof nm, "%d_layout_r2r_transposed", PREC);
    FN(layout_case)(nm, 3, 3, n, 2, td, is, os, FN(realarr)(x), FN(realarr)(y), 0, k, NULL);
    X(free)(x);
    X(free)(y);
    int m[2] = {7, 4}, t1[1] = {1}, o[2] = {1, 0}, g[2] = {2, 1};
    X(r2r_kind) k1[1] = {FFTW_DHT};
    long st[2];
    long l = FN(mkstrides)(2, m, o, g, 1, st);
    R *z = X(alloc_real)(l);
    snprintf(nm, sizeof nm, "%d_layout_r2r_inplace_gaps", PREC);
    FN(layout_case)(nm, 3, 2, m, 1, t1, st, st, FN(realarr)(z), FN(realarr)(z), 0, k1, NULL);
    X(free)(z);
  }
  /* the advanced interface with embedded arrays: 3 arrays of 6 x 5 in 6 x 7 (stride 2, a gap
   * of 3 between arrays) to 6 x 6 (stride 1, distance 40) */
  {
    int n[3] = {6, 5, 3}, td[2] = {0, 1}, nn[2] = {6, 5}, ie[2] = {6, 7}, oe[2] = {6, 6};
    long is[3] = {2 * 7, 2, 2 * 42 + 3}, os[3] = {6, 1, 40};
    FN(manyp) m = {2, 3, 2, 2 * 42 + 3, 1, 40, nn, ie, oe};
    CPX *x = X(alloc_complex)(3 * is[2]), *y = X(alloc_complex)(120);
    snprintf(nm, sizeof nm, "%d_layout_many_dft_embed", PREC);
    FN(layout_case)(nm, 0, 3, n, 2, td, is, os, FN(inter)(x), FN(inter)(y), 0, NULL, &m);
    X(free)(x);
    X(free)(y);
    /* r2c and c2r: 2 arrays of 4 x 9 real values in 4 x 11 (distance 50), half spectra 4 x 5
     * in 4 x 6 (stride 2, distance 52) */
    int r[3] = {4, 9, 2}, rn[2] = {4, 9}, rie[2] = {4, 11}, coe[2] = {4, 6};
    long rs[3] = {11, 1, 50}, cs[3] = {12, 2, 52};
    FN(manyp) mr = {2, 2, 1, 50, 2, 52, rn, rie, coe};
    FN(manyp) mc = {2, 2, 2, 52, 1, 50, rn, coe, rie};
    R *xr = X(alloc_real)(100), *zr = X(alloc_real)(100);
    CPX *yc = X(alloc_complex)(104);
    snprintf(nm, sizeof nm, "%d_layout_many_r2c_embed", PREC);
    FN(layout_case)(nm, 1, 3, r, 2, td, rs, cs, FN(realarr)(xr), FN(inter)(yc), 0, NULL, &mr);
    snprintf(nm, sizeof nm, "%d_layout_many_c2r_embed", PREC);
    FN(layout_case)(nm, 2, 3, r, 2, td, cs, rs, FN(inter)(yc), FN(realarr)(zr), 0, NULL, &mc);
    X(r2r_kind) k[2] = {FFTW_RODFT00, FFTW_HC2R};
    snprintf(nm, sizeof nm, "%d_layout_many_r2r_embed", PREC);
    FN(layout_case)(nm, 3, 3, r, 2, td, rs, rs, FN(realarr)(xr), FN(realarr)(zr), 0, k, &(FN(manyp)){2, 2, 1, 50, 1, 50, rn, rie, rie});
    X(free)(xr);
    X(free)(zr);
    X(free)(yc);
  }
}

/* ----------------------------------------------------- new arrays and concurrent execution */

struct FN(job) {
  X(plan) p;
  CPX *in, *out;
  const R *want;
  long count;
  int reps, bad;
};

static void *FN(worker)(void *arg) {
  struct FN(job) *j = arg;
  for (int r = 0; r < j->reps; r++) {
    memset(j->out, 0, j->count * sizeof(CPX));
    X(execute_dft)(j->p, j->in, j->out);
    for (long i = 0; i < 2 * j->count; i++)
      if (((R *) j->out)[i] != j->want[i]) {
        j->bad++;
        break;
      }
  }
  return NULL;
}

static void FN(new_arrays)(void) {
  char nm[256];
  int n[2] = {32, 48};
  long N = 32 * 48;
  CPX *a = X(alloc_complex)(N), *b = X(alloc_complex)(N);
  X(plan) p = X(plan_dft)(2, n, a, b, FFTW_BACKWARD, FFTW_MEASURE);
  CHECK(p != NULL, "%d new arrays: plan", PREC);
  if (!p) return;
  /* two other input arrays, each checked against the reference, then executed on two
   * threads at the same time over and over */
  CPX *in[2], *out[2], *seq[2];
  for (int t = 0; t < 2; t++) {
    in[t] = X(alloc_complex)(N);
    out[t] = X(alloc_complex)(N);
    seq[t] = X(alloc_complex)(N);
    LD *ref = malloc(2 * N * sizeof(LD));
    FN(fill)((R *) in[t], ref, 2 * N);
    X(execute_dft)(p, in[t], seq[t]);
    ref_dft_dim(ref, 2, n, 0, FFTW_BACKWARD);
    ref_dft_dim(ref, 2, n, 1, FFTW_BACKWARD);
    double e = RELERR((R *) seq[t], ref, 2 * N);
    CHECK(e < TOL, "%d execute_dft on new arrays %d: rel err %g", PREC, t, e);
    snprintf(nm, sizeof nm, "%d_execute_dft_new_%d", PREC, t);
    dump(nm, seq[t], 2 * N * sizeof(R));
    free(ref);
  }
  struct FN(job) jobs[2];
  pthread_t th[2];
  for (int t = 0; t < 2; t++) {
    jobs[t] = (struct FN(job)){p, in[t], out[t], (const R *) seq[t], N, 200, 0};
    pthread_create(&th[t], NULL, FN(worker), &jobs[t]);
  }
  for (int t = 0; t < 2; t++) pthread_join(th[t], NULL);
  CHECK(jobs[0].bad == 0 && jobs[1].bad == 0,
        "%d one plan on two pthreads at once: %d and %d of 200 runs differ", PREC, jobs[0].bad, jobs[1].bad);
  /* copies outlive the original */
  X(plan) q = X(copy_plan)(p);
  CHECK(q != NULL, "%d copy_plan", PREC);
  X(destroy_plan)(p);
  if (q) {
    X(execute_dft)(q, in[0], out[0]);
    CHECK(memcmp(out[0], seq[0], N * sizeof(CPX)) == 0, "%d copy of a destroyed plan executes", PREC);
    X(destroy_plan)(q);
  }
  for (int t = 0; t < 2; t++) {
    X(free)(in[t]);
    X(free)(out[t]);
    X(free)(seq[t]);
  }
  X(free)(a);
  X(free)(b);

  /* r2c, c2r and r2r on new arrays */
  {
    int m[1] = {30};
    R *x = X(alloc_real)(30), *x2 = X(alloc_real)(30), *z = X(alloc_real)(30);
    CPX *y = X(alloc_complex)(16), *y2 = X(alloc_complex)(16);
    X(plan) pr = X(plan_dft_r2c_1d)(30, x, y, FFTW_ESTIMATE);
    X(plan) pc = X(plan_dft_c2r_1d)(30, y, x, FFTW_ESTIMATE);
    X(plan) pk = X(plan_r2r_1d)(30, x, z, FFTW_REDFT00, FFTW_ESTIMATE);
    LD *xr = malloc(30 * sizeof(LD)), *half = malloc(32 * sizeof(LD)), *rr = malloc(30 * sizeof(LD));
    FN(fill)(x2, xr, 30);
    X(execute_dft_r2c)(pr, x2, y2);
    ref_r2c(xr, 1, m, half);
    double e = RELERR((R *) y2, half, 32);
    CHECK(e < TOL, "%d execute_dft_r2c: rel err %g", PREC, e);
    X(execute_r2r)(pk, x2, z);
    ref_r2r_line(xr, rr, 30, FFTW_REDFT00);
    e = RELERR(z, rr, 30);
    CHECK(e < TOL, "%d execute_r2r: rel err %g", PREC, e);
    X(execute_dft_c2r)(pc, y2, x2);
    for (int i = 0; i < 30; i++) rr[i] = 30 * xr[i];
    e = RELERR(x2, rr, 30);
    CHECK(e < TOL, "%d execute_dft_c2r: rel err %g", PREC, e);
    X(destroy_plan)(pr);
    X(destroy_plan)(pc);
    X(destroy_plan)(pk);
    X(free)(x);
    X(free)(x2);
    X(free)(z);
    X(free)(y);
    X(free)(y2);
    free(xr);
    free(half);
    free(rr);
  }
}

/* ------------------------------- large sizes with threads: round trips, outputs for Julia */

static void FN(large)(void) {
  char nm[256];
  X(plan_with_nthreads)(4);
  CHECK(X(planner_nthreads)() == 4, "%d planner_nthreads", PREC);
  /* complex 3D 40 x 48 x 64 and 1D 2^16 + 1 (no small factor: Bluestein in FFTL) */
  int shapes[2][3] = {{40, 48, 64}, {65537, 0, 0}};
  int ranks[2] = {3, 1};
  for (int s = 0; s < 2; s++) {
    int rank = ranks[s];
    long N = prod(rank, shapes[s]);
    CPX *x = X(alloc_complex)(N), *y = X(alloc_complex)(N), *z = X(alloc_complex)(N);
    X(plan) f = X(plan_dft)(rank, shapes[s], x, y, FFTW_FORWARD, FFTW_ESTIMATE);
    X(plan) g = X(plan_dft)(rank, shapes[s], y, z, FFTW_BACKWARD, FFTW_ESTIMATE);
    FN(name)(nm, sizeof nm, "large_c2c", rank, shapes[s], NULL);
    CHECK(f && g, "%s: plans", nm);
    if (f && g) {
      LD *ref = malloc(2 * N * sizeof(LD));
      FN(fill)((R *) x, NULL, 2 * N);
      X(execute)(f);
      X(execute)(g);
      for (long i = 0; i < 2 * N; i++) ref[i] = (LD) N * ((R *) x)[i];
      double e = RELERR((R *) z, ref, 2 * N);
      CHECK(e < 10 * TOL, "%s: round trip rel err %g", nm, e);
      dump(nm, y, 2 * N * sizeof(R));
      free(ref);
    }
    if (f) X(destroy_plan)(f);
    if (g) X(destroy_plan)(g);
    X(free)(x);
    X(free)(y);
    X(free)(z);
  }
  /* r2c 3D 30 x 64 x 100 out of place, r2r 2D 200 x 150 DCT-II/DST-IV in place */
  {
    int n[3] = {30, 64, 100};
    long N = 30 * 64 * 100, nc = 30 * 64 * 51;
    R *x = X(alloc_real)(N), *xo = X(alloc_real)(N);
    CPX *y = X(alloc_complex)(nc);
    X(plan) f = X(plan_dft_r2c)(3, n, x, y, FFTW_ESTIMATE);
    X(plan) g = X(plan_dft_c2r)(3, n, y, xo, FFTW_ESTIMATE);
    snprintf(nm, sizeof nm, "%d_large_r2c_30_64_100", PREC);
    CHECK(f && g, "%s: plans", nm);
    if (f && g) {
      LD *ref = malloc(N * sizeof(LD));
      FN(fill)(x, NULL, N);
      X(execute)(f);
      dump(nm, y, 2 * nc * sizeof(R));
      X(execute)(g);                         /* destroys y in FFTW */
      for (long i = 0; i < N; i++) ref[i] = (LD) N * x[i];
      double e = RELERR(xo, ref, N);
      CHECK(e < 10 * TOL, "%s: round trip rel err %g", nm, e);
      free(ref);
    }
    if (f) X(destroy_plan)(f);
    if (g) X(destroy_plan)(g);
    X(free)(x);
    X(free)(xo);
    X(free)(y);
  }
  {
    int n[2] = {200, 150};
    long N = 30000;
    X(r2r_kind) k[2] = {FFTW_REDFT10, FFTW_RODFT11}, ki[2] = {FFTW_REDFT01, FFTW_RODFT11};
    R *x = X(alloc_real)(N), *x0 = X(alloc_real)(N);
    X(plan) f = X(plan_r2r)(2, n, x, x, k, FFTW_ESTIMATE);
    X(plan) g = X(plan_r2r)(2, n, x, x, ki, FFTW_ESTIMATE);
    snprintf(nm, sizeof nm, "%d_large_r2r_200_150", PREC);
    CHECK(f && g, "%s: plans", nm);
    if (f && g) {
      LD *ref = malloc(N * sizeof(LD));
      FN(fill)(x, NULL, N);
      memcpy(x0, x, N * sizeof(R));
      X(execute)(f);
      dump(nm, x, N * sizeof(R));
      X(execute)(g);
      for (long i = 0; i < N; i++) ref[i] = (LD) (400 * 300) * x0[i];
      double e = RELERR(x, ref, N);
      CHECK(e < 10 * TOL, "%s: round trip rel err %g", nm, e);
      free(ref);
    }
    if (f) X(destroy_plan)(f);
    if (g) X(destroy_plan)(g);
    X(free)(x);
    X(free)(x0);
  }
  X(plan_with_nthreads)(1);
}

/* ------------------------------------------------ memory, wisdom, flags, information */

struct FN(sbuf) {
  char *s;
  size_t len, pos;
};
static void FN(put)(char c, void *data) {
  struct FN(sbuf) *b = data;
  if (b->len < 65536) b->s[b->len++] = c;
}
static int FN(get)(void *data) {
  struct FN(sbuf) *b = data;
  return b->pos < b->len ? (unsigned char) b->s[b->pos++] : EOF;
}

static void FN(misc)(void) {
  /* memory */
  void *ps[4];
  size_t sizes[4] = {1, 24, 1000, 1 << 20};
  int aligned = 1;
  for (int i = 0; i < 4; i++) {
    ps[i] = X(malloc)(sizes[i]);
    aligned &= ps[i] != NULL && ((uintptr_t) ps[i]) % 16 == 0;
  }
  CHECK(aligned, "%d malloc: 16-byte aligned", PREC);
  R *r1 = X(alloc_real)(100), *r2 = X(alloc_real)(333);
  CPX *c1 = X(alloc_complex)(100);
  CHECK(r1 && r2 && c1, "%d alloc_real/alloc_complex", PREC);
  r1[99] = 1;
  r2[332] = 2;
  ((R *) c1)[199] = 3;
  CHECK(X(alignment_of)(r1) == X(alignment_of)(r2), "%d alignment_of of two allocations", PREC);
  for (int i = 0; i < 4; i++) X(free)(ps[i]);
  X(free)(r1);
  X(free)(r2);
  X(free)(c1);

  /* threads */
  CHECK(X(init_threads)() != 0, "%d init_threads", PREC);
  X(plan_with_nthreads)(3);
  CHECK(X(planner_nthreads)() == 3, "%d planner_nthreads", PREC);
  X(plan_with_nthreads)(1);
  X(make_planner_thread_safe)();

  /* information */
  CPX *a = X(alloc_complex)(64), *b = X(alloc_complex)(64);
  X(plan) p = X(plan_dft_1d)(64, a, b, FFTW_FORWARD, FFTW_ESTIMATE);
  double add = -1, mul = -1, fma = -1;
  X(flops)(p, &add, &mul, &fma);
  CHECK(add > 0 && mul > 0 && fma >= 0, "%d flops: %g %g %g", PREC, add, mul, fma);
  CHECK(X(cost)(p) >= 0, "%d cost", PREC);
  char *s = X(sprint_plan)(p);
  CHECK(s != NULL && strlen(s) > 0, "%d sprint_plan", PREC);
  free(s);
  FILE *tf = tmpfile();
  X(fprint_plan)(p, tf);
  CHECK(tf && ftell(tf) > 0, "%d fprint_plan", PREC);
  if (tf) fclose(tf);
  X(set_timelimit)(FFTW_NO_TIMELIMIT);
  X(set_timelimit)(0.5);
  X(destroy_plan)(p);

  /* wisdom: export and import in all forms */
  char *w = X(export_wisdom_to_string)();
  CHECK(w != NULL && strlen(w) > 0, "%d export_wisdom_to_string", PREC);
  CHECK(w && X(import_wisdom_from_string)(w) == 1, "%d import_wisdom_from_string of the export", PREC);
  free(w);
  char path[1024];
  snprintf(path, sizeof path, "%s/%d_wisdom.txt", outdir, PREC);
  CHECK(X(export_wisdom_to_filename)(path) == 1, "%d export_wisdom_to_filename", PREC);
  CHECK(X(import_wisdom_from_filename)(path) == 1, "%d import_wisdom_from_filename", PREC);
  FILE *wf = fopen(path, "w");
  X(export_wisdom_to_file)(wf);
  fclose(wf);
  wf = fopen(path, "r");
  CHECK(X(import_wisdom_from_file)(wf) == 1, "%d import_wisdom_from_file", PREC);
  fclose(wf);
  remove(path);
  struct FN(sbuf) sb = {malloc(65536), 0, 0};
  X(export_wisdom)(FN(put), &sb);
  CHECK(sb.len > 0, "%d export_wisdom (callback)", PREC);
  CHECK(X(import_wisdom)(FN(get), &sb) == 1, "%d import_wisdom (callback)", PREC);
  free(sb.s);

  /* FFTW_WISDOM_ONLY without wisdom: NULL; the arrays are not touched */
  X(forget_wisdom)();
  for (int i = 0; i < 128; i++) ((R *) a)[i] = 7;
  p = X(plan_dft_1d)(1237, a, b, FFTW_FORWARD, FFTW_WISDOM_ONLY | FFTW_PATIENT);
  CHECK(p == NULL, "%d FFTW_WISDOM_ONLY without wisdom: NULL", PREC);
  if (p) X(destroy_plan)(p);
  p = X(plan_dft_1d)(64, a, b, FFTW_FORWARD, FFTW_ESTIMATE);
  int untouched = 1;
  for (int i = 0; i < 128; i++) untouched &= ((R *) a)[i] == 7;
  CHECK(untouched, "%d FFTW_ESTIMATE planning leaves the input alone", PREC);
  X(destroy_plan)(p);
  X(free)(a);
  X(free)(b);

#ifdef WITH_COMPLEX_H
  /* fftw_complex is the C99 complex type: arithmetic on it */
  CPX *z = X(alloc_complex)(8);
  CPX *zo = X(alloc_complex)(8);
  X(plan) pz = X(plan_dft_1d)(8, z, zo, FFTW_FORWARD, FFTW_ESTIMATE);
  for (int i = 0; i < 8; i++) z[i] = (i == 1) ? 3 + 4 * I : 0;
  X(execute)(pz);
  /* DFT of 3+4i at index 1: zo[k] = (3+4i) exp(-2 pi i k / 8) */
  double err = 0;
  for (int k = 0; k < 8; k++) {
    double complex want = (3 + 4 * I) * cexp(-2 * M_PI * I * k / 8);
    double d = cabs((double complex) zo[k] - want);
    if (d > err) err = d;
  }
  CHECK(err < 1e-5, "%d complex.h arithmetic on fftw_complex: err %g", PREC, err);
  X(destroy_plan)(pz);
  X(free)(z);
  X(free)(zo);
#endif
  X(cleanup)();
}

/* ----------------------------------------------------------------------------- driver */

static void FN(run)(void) {
  rng_seed(PREC);
  static const int sizes1[] = {1, 2, 7, 8, 12, 17, 64, 97, 100};
  static const int sizes2[][2] = {{5, 6}, {8, 8}, {7, 13}, {1, 9}, {16, 3}};
  static const int sizes3[][3] = {{4, 6, 5}, {3, 7, 2}, {8, 1, 9}};
  for (int sign = -1; sign <= 1; sign += 2)
    for (int ip = 0; ip <= 1; ip++)
      for (int via = 0; via <= 1; via++) {
        for (unsigned i = 0; i < sizeof sizes1 / sizeof *sizes1; i++) FN(c2c_basic)(1, &sizes1[i], sign, ip, via);
        for (unsigned i = 0; i < sizeof sizes2 / sizeof *sizes2; i++) FN(c2c_basic)(2, sizes2[i], sign, ip, via);
        for (unsigned i = 0; i < sizeof sizes3 / sizeof *sizes3; i++) FN(c2c_basic)(3, sizes3[i], sign, ip, via);
      }
  static const int rs1[] = {1, 2, 7, 8, 17, 100};
  static const int rs2[][2] = {{5, 6}, {8, 9}, {7, 1}};
  static const int rs3[][3] = {{3, 4, 5}, {4, 5, 6}};
  for (int ip = 0; ip <= 1; ip++)
    for (int via = 0; via <= 1; via++) {
      for (unsigned i = 0; i < sizeof rs1 / sizeof *rs1; i++) FN(real_basic)(1, &rs1[i], ip, via);
      for (unsigned i = 0; i < sizeof rs2 / sizeof *rs2; i++) FN(real_basic)(2, rs2[i], ip, via);
      for (unsigned i = 0; i < sizeof rs3 / sizeof *rs3; i++) FN(real_basic)(3, rs3[i], ip, via);
    }
  static const int ks[] = {2, 5, 8, 9, 16, 17};
  for (int kind = 0; kind <= 10; kind++)
    for (unsigned i = 0; i < sizeof ks / sizeof *ks; i++)
      for (int ip = 0; ip <= 1; ip++) {
        X(r2r_kind) k = (X(r2r_kind)) kind;
        FN(r2r_basic)(1, &ks[i], &k, ip, (int) i % 2);
      }
  {
    int n2[2] = {5, 8}, n3[3] = {4, 6, 5};
    X(r2r_kind) k2[2] = {FFTW_RODFT00, FFTW_HC2R}, k3[3] = {FFTW_REDFT00, FFTW_R2HC, FFTW_RODFT10};
    for (int ip = 0; ip <= 1; ip++)
      for (int via = 0; via <= 1; via++) {
        FN(r2r_basic)(2, n2, k2, ip, via);
        FN(r2r_basic)(3, n3, k3, ip, via);
      }
  }
  FN(many)();
  FN(guru)();
  FN(layouts)();
  FN(new_arrays)();
  FN(large)();
  FN(misc)();
}

#undef CAT_
#undef CAT
#undef FN
#undef CPX
