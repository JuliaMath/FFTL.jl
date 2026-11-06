/* fftw3.h for libfftl: the FFTW3 C interface, implemented by FFTL.
 *
 * Programs written for FFTW3 compile against this header and link against libfftl (or its
 * links libfftw3 and libfftw3f, which are the same library) without source changes, in double
 * (fftw_*) and single (fftwf_*) precision. Written from the FFTW manual
 * (https://www.fftw.org/fftw3_doc/); the constant values are those of FFTW.jl (src/fft.jl).
 * This is not FFTW's header and contains no FFTW code.
 *
 * What differs from FFTW:
 * - Layouts: every layout the manual describes is accepted, including strides with gaps,
 *   embedded arrays, negative strides, different input and output layouts and split complex
 *   arrays. When the transform and vector dimensions (in any order) form one contiguous array
 *   without gaps, with the same layout for input and output (r2c/c2r: FFTW's real and complex
 *   layouts of that array, padded in place, the halved dimension the fastest-varying of the
 *   transform dimensions), the transform runs on the arrays themselves. Any other layout, and
 *   every split array, is copied into a contiguous buffer of the plan and back, which costs one
 *   pass over the input and one over the output. In-place split r2c/c2r of rank above 1 (NULL
 *   in FFTW) are planned as well. Long double, quad precision and MPI are not provided.
 * - Failed plans return NULL with the reason in fftl_last_error (fftl.h).
 * - Planner flags are accepted and change nothing, except FFTW_WISDOM_ONLY: there is never
 *   wisdom, so such a plan is NULL. Planning never reads or writes the arrays; c2r keeps
 *   its input out of place.
 * - Wisdom export writes a placeholder, import accepts anything and returns 1 (0 if a named
 *   file cannot be opened); fftw_import_system_wisdom returns 0 (there is no system wisdom).
 * - fftw_cost and fftw_flops return an estimate (5 N log2 N for a complex DFT of N points,
 *   half of it for real transforms), not a count of the operations FFTL performs.
 * - Any alignment is fine; fftw_alignment_of returns 0 for every pointer. fftw_malloc
 *   aligns to 64 bytes.
 * - The worker pool of the library is fixed when it is built (fftl.h, Threading);
 *   fftw_plan_with_nthreads sets how many of its threads later plans use, and
 *   fftw_threads_set_callback is accepted and ignored.
 * - Plans can be created, executed and destroyed from any thread; the same plan may execute
 *   on several threads at once through the new-array execute functions, as in FFTW.
 * - Not provided: the variables fftw_version, fftw_cc and fftw_codelet_optim, and
 *   fftw_estimate_cost (not described in the manual).
 */
#ifndef FFTL_FFTW3_H
#define FFTL_FFTW3_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Complex numbers: interleaved (re, im) pairs. The C99 complex type if <complex.h> was
 * included before this header (binary compatible with the arrays). */
#if !defined(__cplusplus) && defined(_Complex_I)
typedef double _Complex fftw_complex;
typedef float _Complex fftwf_complex;
#else
typedef double fftw_complex[2];
typedef float fftwf_complex[2];
#endif

/* Opaque plans; NULL is a failed plan. */
typedef struct fftl_fftw_plan_s *fftw_plan;
typedef struct fftl_fftwf_plan_s *fftwf_plan;

/* Guru dimensions: size, input stride, output stride (in elements of the array they refer to:
 * complex numbers for interleaved complex arrays, real numbers for real arrays). */
typedef struct {
  int n;
  int is;
  int os;
} fftw_iodim;
typedef struct {
  ptrdiff_t n;
  ptrdiff_t is;
  ptrdiff_t os;
} fftw_iodim64;
typedef fftw_iodim fftwf_iodim;
typedef fftw_iodim64 fftwf_iodim64;

/* Real-to-real kinds */
typedef enum {
  FFTW_R2HC = 0,
  FFTW_HC2R = 1,
  FFTW_DHT = 2,
  FFTW_REDFT00 = 3,
  FFTW_REDFT01 = 4,
  FFTW_REDFT10 = 5,
  FFTW_REDFT11 = 6,
  FFTW_RODFT00 = 7,
  FFTW_RODFT01 = 8,
  FFTW_RODFT10 = 9,
  FFTW_RODFT11 = 10
} fftw_r2r_kind;
typedef fftw_r2r_kind fftwf_r2r_kind;

/* Sign of the exponent */
#define FFTW_FORWARD (-1)
#define FFTW_BACKWARD (+1)

/* Planner flags (all accepted; only FFTW_WISDOM_ONLY changes the result) */
#define FFTW_MEASURE (0U)
#define FFTW_DESTROY_INPUT (1U << 0)
#define FFTW_UNALIGNED (1U << 1)
#define FFTW_CONSERVE_MEMORY (1U << 2)
#define FFTW_EXHAUSTIVE (1U << 3)
#define FFTW_PRESERVE_INPUT (1U << 4)
#define FFTW_PATIENT (1U << 5)
#define FFTW_ESTIMATE (1U << 6)
#define FFTW_NO_SIMD (1U << 17)
#define FFTW_WISDOM_ONLY (1U << 21)

/* fftw_set_timelimit: no limit (the default; the limit has no effect in FFTL) */
#define FFTW_NO_TIMELIMIT (-1.0)

/* Planner: basic interface */
fftw_plan fftw_plan_dft_1d(int n0, fftw_complex *in, fftw_complex *out, int sign, unsigned flags);
fftw_plan fftw_plan_dft_2d(int n0, int n1, fftw_complex *in, fftw_complex *out, int sign, unsigned flags);
fftw_plan fftw_plan_dft_3d(int n0, int n1, int n2, fftw_complex *in, fftw_complex *out, int sign, unsigned flags);
fftw_plan fftw_plan_dft(int rank, const int *n, fftw_complex *in, fftw_complex *out, int sign, unsigned flags);
fftw_plan fftw_plan_dft_r2c_1d(int n0, double *in, fftw_complex *out, unsigned flags);
fftw_plan fftw_plan_dft_r2c_2d(int n0, int n1, double *in, fftw_complex *out, unsigned flags);
fftw_plan fftw_plan_dft_r2c_3d(int n0, int n1, int n2, double *in, fftw_complex *out, unsigned flags);
fftw_plan fftw_plan_dft_r2c(int rank, const int *n, double *in, fftw_complex *out, unsigned flags);
fftw_plan fftw_plan_dft_c2r_1d(int n0, fftw_complex *in, double *out, unsigned flags);
fftw_plan fftw_plan_dft_c2r_2d(int n0, int n1, fftw_complex *in, double *out, unsigned flags);
fftw_plan fftw_plan_dft_c2r_3d(int n0, int n1, int n2, fftw_complex *in, double *out, unsigned flags);
fftw_plan fftw_plan_dft_c2r(int rank, const int *n, fftw_complex *in, double *out, unsigned flags);
fftw_plan fftw_plan_r2r_1d(int n, double *in, double *out, fftw_r2r_kind kind, unsigned flags);
fftw_plan fftw_plan_r2r_2d(int n0, int n1, double *in, double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, unsigned flags);
fftw_plan fftw_plan_r2r_3d(int n0, int n1, int n2, double *in, double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flags);
fftw_plan fftw_plan_r2r(int rank, const int *n, double *in, double *out, const fftw_r2r_kind *kind, unsigned flags);

/* Planner: advanced interface */
fftw_plan fftw_plan_many_dft(int rank, const int *n, int howmany, fftw_complex *in, const int *inembed, int istride, int idist, fftw_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned flags);
fftw_plan fftw_plan_many_dft_r2c(int rank, const int *n, int howmany, double *in, const int *inembed, int istride, int idist, fftw_complex *out, const int *onembed, int ostride, int odist, unsigned flags);
fftw_plan fftw_plan_many_dft_c2r(int rank, const int *n, int howmany, fftw_complex *in, const int *inembed, int istride, int idist, double *out, const int *onembed, int ostride, int odist, unsigned flags);
fftw_plan fftw_plan_many_r2r(int rank, const int *n, int howmany, double *in, const int *inembed, int istride, int idist, double *out, const int *onembed, int ostride, int odist, const fftw_r2r_kind *kind, unsigned flags);

/* Planner: guru interface */
fftw_plan fftw_plan_guru_dft(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, fftw_complex *in, fftw_complex *out, int sign, unsigned flags);
fftw_plan fftw_plan_guru_split_dft(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *ri, double *ii, double *ro, double *io, unsigned flags);
fftw_plan fftw_plan_guru_dft_r2c(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *in, fftw_complex *out, unsigned flags);
fftw_plan fftw_plan_guru_split_dft_r2c(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *in, double *ro, double *io, unsigned flags);
fftw_plan fftw_plan_guru_dft_c2r(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, fftw_complex *in, double *out, unsigned flags);
fftw_plan fftw_plan_guru_split_dft_c2r(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *ri, double *ii, double *out, unsigned flags);
fftw_plan fftw_plan_guru_r2r(int rank, const fftw_iodim *dims, int howmany_rank, const fftw_iodim *howmany_dims, double *in, double *out, const fftw_r2r_kind *kind, unsigned flags);

/* Planner: 64-bit guru interface */
fftw_plan fftw_plan_guru64_dft(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, fftw_complex *in, fftw_complex *out, int sign, unsigned flags);
fftw_plan fftw_plan_guru64_split_dft(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *ri, double *ii, double *ro, double *io, unsigned flags);
fftw_plan fftw_plan_guru64_dft_r2c(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *in, fftw_complex *out, unsigned flags);
fftw_plan fftw_plan_guru64_split_dft_r2c(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *in, double *ro, double *io, unsigned flags);
fftw_plan fftw_plan_guru64_dft_c2r(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, fftw_complex *in, double *out, unsigned flags);
fftw_plan fftw_plan_guru64_split_dft_c2r(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *ri, double *ii, double *out, unsigned flags);
fftw_plan fftw_plan_guru64_r2r(int rank, const fftw_iodim64 *dims, int howmany_rank, const fftw_iodim64 *howmany_dims, double *in, double *out, const fftw_r2r_kind *kind, unsigned flags);

/* Execution */
void fftw_execute(const fftw_plan plan);
void fftw_execute_dft(const fftw_plan p, fftw_complex *in, fftw_complex *out);
void fftw_execute_split_dft(const fftw_plan p, double *ri, double *ii, double *ro, double *io);
void fftw_execute_dft_r2c(const fftw_plan p, double *in, fftw_complex *out);
void fftw_execute_split_dft_r2c(const fftw_plan p, double *in, double *ro, double *io);
void fftw_execute_dft_c2r(const fftw_plan p, fftw_complex *in, double *out);
void fftw_execute_split_dft_c2r(const fftw_plan p, double *ri, double *ii, double *out);
void fftw_execute_r2r(const fftw_plan p, double *in, double *out);

/* Plans */
void fftw_destroy_plan(fftw_plan plan);
fftw_plan fftw_copy_plan(fftw_plan plan);
void fftw_cleanup(void);
void fftw_set_timelimit(double seconds);
double fftw_cost(const fftw_plan plan);
void fftw_flops(const fftw_plan plan, double *add, double *mul, double *fma);
void fftw_fprint_plan(const fftw_plan plan, FILE *output_file);
void fftw_print_plan(const fftw_plan plan);
char *fftw_sprint_plan(const fftw_plan plan);

/* Memory */
void *fftw_malloc(size_t n);
void fftw_free(void *p);
double *fftw_alloc_real(size_t n);
fftw_complex *fftw_alloc_complex(size_t n);
int fftw_alignment_of(double *p);

/* Wisdom */
int fftw_export_wisdom_to_filename(const char *filename);
void fftw_export_wisdom_to_file(FILE *output_file);
char *fftw_export_wisdom_to_string(void);
void fftw_export_wisdom(void (*write_char)(char c, void *), void *data);
int fftw_import_system_wisdom(void);
int fftw_import_wisdom_from_filename(const char *filename);
int fftw_import_wisdom_from_file(FILE *input_file);
int fftw_import_wisdom_from_string(const char *input_string);
int fftw_import_wisdom(int (*read_char)(void *), void *data);
void fftw_forget_wisdom(void);

/* Threads */
int fftw_init_threads(void);
void fftw_plan_with_nthreads(int nthreads);
int fftw_planner_nthreads(void);
void fftw_cleanup_threads(void);
void fftw_make_planner_thread_safe(void);
void fftw_threads_set_callback(void (*parallel_loop)(void *(*work)(void *), char *jobdata, size_t elsize, int njobs, void *data), void *data);

/* Planner: basic interface */
fftwf_plan fftwf_plan_dft_1d(int n0, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags);
fftwf_plan fftwf_plan_dft_2d(int n0, int n1, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags);
fftwf_plan fftwf_plan_dft_3d(int n0, int n1, int n2, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags);
fftwf_plan fftwf_plan_dft(int rank, const int *n, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags);
fftwf_plan fftwf_plan_dft_r2c_1d(int n0, float *in, fftwf_complex *out, unsigned flags);
fftwf_plan fftwf_plan_dft_r2c_2d(int n0, int n1, float *in, fftwf_complex *out, unsigned flags);
fftwf_plan fftwf_plan_dft_r2c_3d(int n0, int n1, int n2, float *in, fftwf_complex *out, unsigned flags);
fftwf_plan fftwf_plan_dft_r2c(int rank, const int *n, float *in, fftwf_complex *out, unsigned flags);
fftwf_plan fftwf_plan_dft_c2r_1d(int n0, fftwf_complex *in, float *out, unsigned flags);
fftwf_plan fftwf_plan_dft_c2r_2d(int n0, int n1, fftwf_complex *in, float *out, unsigned flags);
fftwf_plan fftwf_plan_dft_c2r_3d(int n0, int n1, int n2, fftwf_complex *in, float *out, unsigned flags);
fftwf_plan fftwf_plan_dft_c2r(int rank, const int *n, fftwf_complex *in, float *out, unsigned flags);
fftwf_plan fftwf_plan_r2r_1d(int n, float *in, float *out, fftwf_r2r_kind kind, unsigned flags);
fftwf_plan fftwf_plan_r2r_2d(int n0, int n1, float *in, float *out, fftwf_r2r_kind kind0, fftwf_r2r_kind kind1, unsigned flags);
fftwf_plan fftwf_plan_r2r_3d(int n0, int n1, int n2, float *in, float *out, fftwf_r2r_kind kind0, fftwf_r2r_kind kind1, fftwf_r2r_kind kind2, unsigned flags);
fftwf_plan fftwf_plan_r2r(int rank, const int *n, float *in, float *out, const fftwf_r2r_kind *kind, unsigned flags);

/* Planner: advanced interface */
fftwf_plan fftwf_plan_many_dft(int rank, const int *n, int howmany, fftwf_complex *in, const int *inembed, int istride, int idist, fftwf_complex *out, const int *onembed, int ostride, int odist, int sign, unsigned flags);
fftwf_plan fftwf_plan_many_dft_r2c(int rank, const int *n, int howmany, float *in, const int *inembed, int istride, int idist, fftwf_complex *out, const int *onembed, int ostride, int odist, unsigned flags);
fftwf_plan fftwf_plan_many_dft_c2r(int rank, const int *n, int howmany, fftwf_complex *in, const int *inembed, int istride, int idist, float *out, const int *onembed, int ostride, int odist, unsigned flags);
fftwf_plan fftwf_plan_many_r2r(int rank, const int *n, int howmany, float *in, const int *inembed, int istride, int idist, float *out, const int *onembed, int ostride, int odist, const fftwf_r2r_kind *kind, unsigned flags);

/* Planner: guru interface */
fftwf_plan fftwf_plan_guru_dft(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags);
fftwf_plan fftwf_plan_guru_split_dft(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *ri, float *ii, float *ro, float *io, unsigned flags);
fftwf_plan fftwf_plan_guru_dft_r2c(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *in, fftwf_complex *out, unsigned flags);
fftwf_plan fftwf_plan_guru_split_dft_r2c(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *in, float *ro, float *io, unsigned flags);
fftwf_plan fftwf_plan_guru_dft_c2r(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, fftwf_complex *in, float *out, unsigned flags);
fftwf_plan fftwf_plan_guru_split_dft_c2r(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *ri, float *ii, float *out, unsigned flags);
fftwf_plan fftwf_plan_guru_r2r(int rank, const fftwf_iodim *dims, int howmany_rank, const fftwf_iodim *howmany_dims, float *in, float *out, const fftwf_r2r_kind *kind, unsigned flags);

/* Planner: 64-bit guru interface */
fftwf_plan fftwf_plan_guru64_dft(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags);
fftwf_plan fftwf_plan_guru64_split_dft(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *ri, float *ii, float *ro, float *io, unsigned flags);
fftwf_plan fftwf_plan_guru64_dft_r2c(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *in, fftwf_complex *out, unsigned flags);
fftwf_plan fftwf_plan_guru64_split_dft_r2c(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *in, float *ro, float *io, unsigned flags);
fftwf_plan fftwf_plan_guru64_dft_c2r(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, fftwf_complex *in, float *out, unsigned flags);
fftwf_plan fftwf_plan_guru64_split_dft_c2r(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *ri, float *ii, float *out, unsigned flags);
fftwf_plan fftwf_plan_guru64_r2r(int rank, const fftwf_iodim64 *dims, int howmany_rank, const fftwf_iodim64 *howmany_dims, float *in, float *out, const fftwf_r2r_kind *kind, unsigned flags);

/* Execution */
void fftwf_execute(const fftwf_plan plan);
void fftwf_execute_dft(const fftwf_plan p, fftwf_complex *in, fftwf_complex *out);
void fftwf_execute_split_dft(const fftwf_plan p, float *ri, float *ii, float *ro, float *io);
void fftwf_execute_dft_r2c(const fftwf_plan p, float *in, fftwf_complex *out);
void fftwf_execute_split_dft_r2c(const fftwf_plan p, float *in, float *ro, float *io);
void fftwf_execute_dft_c2r(const fftwf_plan p, fftwf_complex *in, float *out);
void fftwf_execute_split_dft_c2r(const fftwf_plan p, float *ri, float *ii, float *out);
void fftwf_execute_r2r(const fftwf_plan p, float *in, float *out);

/* Plans */
void fftwf_destroy_plan(fftwf_plan plan);
fftwf_plan fftwf_copy_plan(fftwf_plan plan);
void fftwf_cleanup(void);
void fftwf_set_timelimit(double seconds);
double fftwf_cost(const fftwf_plan plan);
void fftwf_flops(const fftwf_plan plan, double *add, double *mul, double *fma);
void fftwf_fprint_plan(const fftwf_plan plan, FILE *output_file);
void fftwf_print_plan(const fftwf_plan plan);
char *fftwf_sprint_plan(const fftwf_plan plan);

/* Memory */
void *fftwf_malloc(size_t n);
void fftwf_free(void *p);
float *fftwf_alloc_real(size_t n);
fftwf_complex *fftwf_alloc_complex(size_t n);
int fftwf_alignment_of(float *p);

/* Wisdom */
int fftwf_export_wisdom_to_filename(const char *filename);
void fftwf_export_wisdom_to_file(FILE *output_file);
char *fftwf_export_wisdom_to_string(void);
void fftwf_export_wisdom(void (*write_char)(char c, void *), void *data);
int fftwf_import_system_wisdom(void);
int fftwf_import_wisdom_from_filename(const char *filename);
int fftwf_import_wisdom_from_file(FILE *input_file);
int fftwf_import_wisdom_from_string(const char *input_string);
int fftwf_import_wisdom(int (*read_char)(void *), void *data);
void fftwf_forget_wisdom(void);

/* Threads */
int fftwf_init_threads(void);
void fftwf_plan_with_nthreads(int nthreads);
int fftwf_planner_nthreads(void);
void fftwf_cleanup_threads(void);
void fftwf_make_planner_thread_safe(void);
void fftwf_threads_set_callback(void (*parallel_loop)(void *(*work)(void *), char *jobdata, size_t elsize, int njobs, void *data), void *data);

#ifdef __cplusplus
}
#endif

#endif /* FFTL_FFTW3_H */
