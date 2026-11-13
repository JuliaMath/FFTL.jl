// C ABI around ducc0's public FFT interface. The only ducc0 header included is ducc0/fft/fft.h.
//
// 1-D: plan objects (pocketfft_c, plan once and execute many times) and the free c2c call.
// N-D: the free functions c2c, r2c, c2r and the real-to-real transforms. Arrays are passed in C order: the
//      Julia side reverses its shape, so the last C axis is Julia's first (fastest) axis.
#include <complex>
#include <cstddef>
#include <numeric>
#include <vector>
#include "ducc0/fft/fft.h"

namespace {

using ducc0::cfmav;
using ducc0::vfmav;
using ducc0::pocketfft_c;
using ducc0::pocketfft_r;
using shape_t = std::vector<size_t>;
using stride_t = std::vector<ptrdiff_t>;

// C-order element strides of a contiguous array of the given shape.
stride_t cstrides(const shape_t &s) {
  stride_t st(s.size());
  ptrdiff_t acc = 1;
  for (size_t i = s.size(); i-- > 0;) {
    st[i] = acc;
    acc *= static_cast<ptrdiff_t>(s[i]);
  }
  return st;
}

shape_t all_axes(size_t nd) {
  shape_t ax(nd);
  std::iota(ax.begin(), ax.end(), size_t(0));
  return ax;
}

shape_t halved(shape_t s) {
  s.back() = s.back() / 2 + 1;
  return s;
}

template <typename T>
void c2c_nd(const void *in, void *out, const size_t *shape, size_t nd, T fct, bool fwd, size_t nt) {
  shape_t s(shape, shape + nd);
  stride_t st = cstrides(s);
  auto i = cfmav<std::complex<T>>(static_cast<const std::complex<T> *>(in), s, st);
  auto o = vfmav<std::complex<T>>(static_cast<std::complex<T> *>(out), s, st);
  ducc0::c2c(i, o, all_axes(nd), fwd, fct, nt);
}

template <typename T>
void r2c_nd(const void *in, void *out, const size_t *shape, size_t nd, T fct, size_t nt) {
  shape_t s(shape, shape + nd), so = halved(s);
  auto i = cfmav<T>(static_cast<const T *>(in), s, cstrides(s));
  auto o = vfmav<std::complex<T>>(static_cast<std::complex<T> *>(out), so, cstrides(so));
  ducc0::r2c(i, o, all_axes(nd), true, fct, nt);
}

// `shape` is the real output shape; the complex input has the halved last axis.
template <typename T>
void c2r_nd(const void *in, void *out, const size_t *shape, size_t nd, T fct, size_t nt) {
  shape_t s(shape, shape + nd), si = halved(s);
  auto i = cfmav<std::complex<T>>(static_cast<const std::complex<T> *>(in), si, cstrides(si));
  auto o = vfmav<T>(static_cast<T *>(out), s, cstrides(s));
  ducc0::c2r(i, o, all_axes(nd), false, fct, nt);
}


// --- real 1-D plans (pocketfft_r, FFTPACK halfcomplex: r0, r1, i1, r2, i2, ...) -------------
// r2c and c2r convert between the halfcomplex result and the complex layout of the transform.
// Both use the plan in place on the buffer they are given: the caller copies the input first.
template <typename T>
void r2c_plan(pocketfft_r<T> *p, T *in, T *buf, std::complex<T> *out, size_t n, size_t nt) {
  T *res = p->exec(in, buf, T(1), true, nt);
  out[0] = {res[0], T(0)};
  for (size_t k = 1; k < (n + 1) / 2; ++k) out[k] = {res[2 * k - 1], res[2 * k]};
  if (n % 2 == 0) out[n / 2] = {res[n - 1], T(0)};
}

template <typename T>
void c2r_plan(pocketfft_r<T> *p, const std::complex<T> *in, T *out, T *buf, size_t n, size_t nt) {
  out[0] = in[0].real();
  for (size_t k = 1; k < (n + 1) / 2; ++k) {
    out[2 * k - 1] = in[k].real();
    out[2 * k] = in[k].imag();
  }
  if (n % 2 == 0) out[n - 1] = in[n / 2].real();
  T *res = p->exec(out, buf, T(1), false, nt);
  if (res != out) for (size_t i = 0; i < n; ++i) out[i] = res[i];
}

// --- real-to-real by FFTW kind code, through ducc0's public transforms ----------------------
// Codes as in FFTW.jl (R2HC 0, HC2R 1, DHT 2, REDFT00 3, REDFT01 4, REDFT10 5, REDFT11 6,
// RODFT00 7, RODFT01 8, RODFT10 9, RODFT11 10).
template <typename T>
void r2r_nd(int code, const size_t *shape, size_t nd, const T *in, T *out, size_t nt) {
  shape_t s(shape, shape + nd);
  auto i = cfmav<T>(in, s, cstrides(s));
  auto o = vfmav<T>(out, s, cstrides(s));
  auto ax = all_axes(nd);
  switch (code) {
    case 0: ducc0::r2r_fftw(i, o, ax, true, T(1), nt); break;
    case 1: ducc0::r2r_fftw(i, o, ax, false, T(1), nt); break;
    case 2: ducc0::r2r_separable_fht(i, o, ax, T(1), nt); break;   // FFTW's DHT: separable, nD
    case 3: ducc0::dct(i, o, ax, 1, T(1), false, nt); break;
    case 4: ducc0::dct(i, o, ax, 3, T(1), false, nt); break;
    case 5: ducc0::dct(i, o, ax, 2, T(1), false, nt); break;
    case 6: ducc0::dct(i, o, ax, 4, T(1), false, nt); break;
    case 7: ducc0::dst(i, o, ax, 1, T(1), false, nt); break;
    case 8: ducc0::dst(i, o, ax, 3, T(1), false, nt); break;
    case 9: ducc0::dst(i, o, ax, 2, T(1), false, nt); break;
    case 10: ducc0::dst(i, o, ax, 4, T(1), false, nt); break;
    default: break;
  }
}

}  // namespace

extern "C" {

// --- 1-D plans -------------------------------------------------------------------------------
void *d1_plan_f64(size_t n) { return new pocketfft_c<double>(n); }
void d1_free_f64(void *p) { delete static_cast<pocketfft_c<double> *>(p); }
size_t d1_bufsize_f64(void *p) { return static_cast<pocketfft_c<double> *>(p)->bufsize(); }
// Returns where the result is: `in` or `buf` (at an offset), depending on the passes of the plan.
void *d1_exec_f64(void *p, void *in, void *buf, double fct, int fwd, size_t nt) {
  return static_cast<pocketfft_c<double> *>(p)->exec(static_cast<ducc0::Cmplx<double> *>(in),
                                                     static_cast<ducc0::Cmplx<double> *>(buf), fct, fwd != 0, nt);
}

void *d1_plan_f32(size_t n) { return new pocketfft_c<float>(n); }
void d1_free_f32(void *p) { delete static_cast<pocketfft_c<float> *>(p); }
size_t d1_bufsize_f32(void *p) { return static_cast<pocketfft_c<float> *>(p)->bufsize(); }
void *d1_exec_f32(void *p, void *in, void *buf, float fct, int fwd, size_t nt) {
  return static_cast<pocketfft_c<float> *>(p)->exec(static_cast<ducc0::Cmplx<float> *>(in),
                                                    static_cast<ducc0::Cmplx<float> *>(buf), fct, fwd != 0, nt);
}

// --- free calls ------------------------------------------------------------------------------
void d1_free_c2c_f64(void *in, void *out, size_t n, double fct, int fwd, size_t nt) {
  c2c_nd<double>(in, out, &n, 1, fct, fwd != 0, nt);
}
void d1_free_c2c_f32(void *in, void *out, size_t n, float fct, int fwd, size_t nt) {
  c2c_nd<float>(in, out, &n, 1, fct, fwd != 0, nt);
}

void dfft_c2c_f64(void *in, void *out, const size_t *shape, size_t nd, double fct, int fwd, size_t nt) {
  c2c_nd<double>(in, out, shape, nd, fct, fwd != 0, nt);
}
void dfft_c2c_f32(void *in, void *out, const size_t *shape, size_t nd, float fct, int fwd, size_t nt) {
  c2c_nd<float>(in, out, shape, nd, fct, fwd != 0, nt);
}
void dfft_r2c_f64(void *in, void *out, const size_t *shape, size_t nd, double fct, size_t nt) {
  r2c_nd<double>(in, out, shape, nd, fct, nt);
}
void dfft_r2c_f32(void *in, void *out, const size_t *shape, size_t nd, float fct, size_t nt) {
  r2c_nd<float>(in, out, shape, nd, fct, nt);
}
void dfft_c2r_f64(void *in, void *out, const size_t *shape, size_t nd, double fct, size_t nt) {
  c2r_nd<double>(in, out, shape, nd, fct, nt);
}
void dfft_c2r_f32(void *in, void *out, const size_t *shape, size_t nd, float fct, size_t nt) {
  c2r_nd<float>(in, out, shape, nd, fct, nt);
}


// real 1-D plans
void *d1r_plan_f64(size_t n) { return new pocketfft_r<double>(n); }
void d1r_free_f64(void *p) { delete static_cast<pocketfft_r<double> *>(p); }
size_t d1r_bufsize_f64(void *p) { return static_cast<pocketfft_r<double> *>(p)->bufsize(); }
void d1r_r2c_f64(void *p, void *in, void *buf, void *out, size_t n, size_t nt) {
  r2c_plan(static_cast<pocketfft_r<double> *>(p), static_cast<double *>(in), static_cast<double *>(buf),
           static_cast<std::complex<double> *>(out), n, nt);
}
void d1r_c2r_f64(void *p, void *in, void *out, void *buf, size_t n, size_t nt) {
  c2r_plan(static_cast<pocketfft_r<double> *>(p), static_cast<const std::complex<double> *>(in),
           static_cast<double *>(out), static_cast<double *>(buf), n, nt);
}
void *d1r_plan_f32(size_t n) { return new pocketfft_r<float>(n); }
void d1r_free_f32(void *p) { delete static_cast<pocketfft_r<float> *>(p); }
size_t d1r_bufsize_f32(void *p) { return static_cast<pocketfft_r<float> *>(p)->bufsize(); }
void d1r_r2c_f32(void *p, void *in, void *buf, void *out, size_t n, size_t nt) {
  r2c_plan(static_cast<pocketfft_r<float> *>(p), static_cast<float *>(in), static_cast<float *>(buf),
           static_cast<std::complex<float> *>(out), n, nt);
}
void d1r_c2r_f32(void *p, void *in, void *out, void *buf, size_t n, size_t nt) {
  c2r_plan(static_cast<pocketfft_r<float> *>(p), static_cast<const std::complex<float> *>(in),
           static_cast<float *>(out), static_cast<float *>(buf), n, nt);
}

// r2r through the public transforms; `shape` in C order
void dfft_r2r_f64(int code, const size_t *shape, size_t nd, void *in, void *out, size_t nt) {
  r2r_nd<double>(code, shape, nd, static_cast<const double *>(in), static_cast<double *>(out), nt);
}
void dfft_r2r_f32(int code, const size_t *shape, size_t nd, void *in, void *out, size_t nt) {
  r2r_nd<float>(code, shape, nd, static_cast<const float *>(in), static_cast<float *>(out), nt);
}

}  // extern "C"
