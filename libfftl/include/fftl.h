/* libfftl: FFTL.jl as a C library.
 *
 * Plans by opaque handle for the complex DFT, the real-input DFT (r2c) and its inverse (c2r),
 * and real-to-real transforms (DCT, DST, Hartley, halfcomplex): any rank, any lengths, both
 * precisions, single- or multi-threaded execution. The plan is built, applied
 * and freed through the four calls below; everything else about it is run-time data inside the
 * library (radix sequence, algorithm and work buffers are chosen and owned by FFTL).
 *
 * Data layout: complex values are interleaved pairs (re, im), stored contiguously in
 * column-major order: in a dims[0] x ... x dims[rank-1] array the first index varies fastest,
 * so the value (i1, ..., ir) sits at offset i1 + dims[0] * (i2 + dims[1] * (...)). Buffers need
 * no alignment beyond what malloc returns, and out-of-place execution (out != in) may not
 * overlap partially.
 *
 * Precision: the element type (8 or 16 bytes per complex value) is fixed by the plan; buffer
 * pointers are passed as void * and must point to the matching layout.
 *
 * Normalization: nothing is normalised. The backward transform is the plain unnormalised
 * inverse map: applying the sign +1 plan after the sign -1 plan multiplies the input by the
 * number of transformed elements.
 */
#ifndef FFTL_H
#define FFTL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  FFTL_PRECISION_DOUBLE = 1,   /* Complex{Float64}, 16 bytes per value */
  FFTL_PRECISION_SINGLE = 2   /* Complex{Float32},  8 bytes per value */
};

/* Real-to-real kinds (fftl_plan_r2r), unnormalised, of a line x_0 … x_(n-1):
 *   FFTL_REDFT00  DCT-I    y_k = x_0 + (-1)^k x_(n-1) + 2 Σ_(j=1)^(n-2) x_j cos(πjk/(n-1)), n ≥ 2
 *   FFTL_REDFT10  DCT-II   y_k = 2 Σ x_j cos(π(j+½)k/n)
 *   FFTL_REDFT01  DCT-III  y_k = x_0 + 2 Σ_(j≥1) x_j cos(πj(k+½)/n)
 *   FFTL_REDFT11  DCT-IV   y_k = 2 Σ x_j cos(π(j+½)(k+½)/n)
 *   FFTL_RODFT00  DST-I    y_k = 2 Σ x_j sin(π(j+1)(k+1)/(n+1))
 *   FFTL_RODFT10  DST-II   y_k = 2 Σ x_j sin(π(j+½)(k+1)/n)
 *   FFTL_RODFT01  DST-III  y_k = (-1)^k x_(n-1) + 2 Σ_(j≤n-2) x_j sin(π(j+1)(k+½)/n)
 *   FFTL_RODFT11  DST-IV   y_k = 2 Σ x_j sin(π(j+½)(k+½)/n)
 *   FFTL_DHT      Hartley  y_k = Σ x_j (cos(2πjk/n) + sin(2πjk/n))
 *   FFTL_R2HC     DFT F_k = Σ x_j e^(-2πijk/n) in halfcomplex order: y_k = Re F_k for
 *                 0 ≤ k ≤ n/2, y_(n-k) = Im F_k for 0 < k < n/2
 *   FFTL_HC2R     inverse of FFTL_R2HC: y_j = Σ_k F_k e^(2πijk/n), F in halfcomplex order
 * Inverse pairs (up to the factor 2n, 2(n-1) for DCT-I, 2(n+1) for DST-I, n for DHT and the
 * halfcomplex pair): DCT-II/III, DST-II/III; DCT-I, DCT-IV, DST-I, DST-IV, DHT are their own
 * inverses. */
enum {
  FFTL_REDFT00 = 0, FFTL_REDFT10 = 1, FFTL_REDFT01 = 2, FFTL_REDFT11 = 3,
  FFTL_RODFT00 = 4, FFTL_RODFT10 = 5, FFTL_RODFT01 = 6, FFTL_RODFT11 = 7,
  FFTL_DHT = 8, FFTL_R2HC = 9, FFTL_HC2R = 10
};

/* Plans are opaque handles. A negative handle is an error and fftl_last_error explains;
 * the only valid operations on an invalid handle are fftl_last_error and passing it to
 * fftl_plan_destroy and fftl_plan_execute, which reject it. */
typedef int64_t fftl_plan;

/* size_t fftl_version(char *buf, size_t buf_len)
 *
 * Writes the library version as a zero-terminated "major.minor.patch" string into buf (at
 * most buf_len bytes including the terminator) and returns the full length of the version
 * string, like snprintf. With buf == NULL it only returns the length.
 */
size_t fftl_version(char *buf, size_t buf_len);

/* fftl_plan fftl_plan_dft(int32_t rank, const int64_t *dims, int32_t transform_rank,
 *                         const int32_t *region, int32_t precision, int32_t sign,
 *                         int32_t threads)
 *
 * Plan for the complex DFT of a rank-dimensional array of dims[0] x ... x dims[rank-1]
 * complex values along the dimensions region[0] ... region[transform_rank-1], with the given
 * precision (FFTL_PRECISION_*) and sign (-1 forward, +1 backward, unnormalised).
 *
 * rank                 >= 1: the rank of the array the buffers point to; dims has this
 *                      length. There is no upper limit; memory and the 64-bit element
 *                      offsets bound the product of the dimensions in practice.
 * dims                 the array dimensions, each >= 1; any length is planned (lengths with
 *                      large prime factors go through Bluestein's convolution algorithm,
 *                      smooth lengths through radix stages).
 * transform_rank       0 to rank: the length of region. Zero transforms nothing; executing
 *                      such a plan just copies input to output (in place: no-op).
 * region               transformed dimensions as indices into dims: 0-based (0 is the
 *                      fastest-varying index), strictly ascending, each within 0 ... rank-1.
 *                      Duplicates and out-of-range indices are errors, as are region arrays
 *                      that are not strictly ascending.
 * sign                 -1 or +1.
 * threads              number of worker threads for execution, >= 1. The plan stores it;
 *                      execution uses the plan's value, bounded by what the transform size
 *                      allows (small transforms run on one thread regardless).
 *
 * The thread count of the library's own worker pool is fixed when the library is built (see
 * "Threading" at the end of this file); a plan with more threads than that queues its work
 * on fewer workers, which gives no speedup.
 *
 * Returns an fftl_plan, or -1 with the reason in fftl_last_error: unsupported precision or
 * sign, rank or transform_rank or region invalid, an array dimension below one, or out of
 * memory.
 */
fftl_plan fftl_plan_dft(int32_t rank, const int64_t *dims, int32_t transform_rank,
                        const int32_t *region, int32_t precision, int32_t sign,
                        int32_t threads);

/* fftl_plan fftl_plan_dft_r2c(int32_t rank, const int64_t *dims, int32_t transform_rank,
 *                             const int32_t *region, int32_t precision, int32_t threads)
 * fftl_plan fftl_plan_dft_c2r(int32_t rank, const int64_t *dims, int32_t transform_rank,
 *                             const int32_t *region, int32_t precision, int32_t threads)
 *
 * Real-input forward DFT (r2c, sign -1) and its unnormalised inverse (c2r, sign +1) of a real
 * array of dims[0] x ... x dims[rank-1] values along region (as in fftl_plan_dft;
 * transform_rank >= 1). dims are the sizes of the REAL array in both cases. The complex array
 * holds the half spectrum: along the first transformed dimension d = region[0] it has
 * dims[d] / 2 + 1 values, the other dimensions as in dims. r2c reads the real array and
 * writes the complex one; c2r reads the complex array (and leaves it unchanged) and writes
 * the real one; c2r after r2c multiplies by the number of transformed elements.
 *
 * In place (out == in, as in FFTW): the real array is padded along d to 2 * (dims[d] / 2 + 1)
 * values, so that it occupies exactly the memory of the complex half spectrum. Its values at
 * positions dims[d] and above along d are ignored by r2c and undefined after c2r; the in-place
 * c2r overwrites its input.
 */
fftl_plan fftl_plan_dft_r2c(int32_t rank, const int64_t *dims, int32_t transform_rank,
                            const int32_t *region, int32_t precision, int32_t threads);
fftl_plan fftl_plan_dft_c2r(int32_t rank, const int64_t *dims, int32_t transform_rank,
                            const int32_t *region, int32_t precision, int32_t threads);

/* fftl_plan fftl_plan_r2r(int32_t rank, const int64_t *dims, int32_t transform_rank,
 *                         const int32_t *region, const int32_t *kinds, int32_t precision,
 *                         int32_t threads)
 *
 * Real-to-real transforms of a real array along region (as in fftl_plan_dft), kinds[i]
 * (FFTL_REDFT00 … FFTL_HC2R) along dimension region[i]. Input and output are real arrays of
 * size dims; in place (out == in) or out of place.
 */
fftl_plan fftl_plan_r2r(int32_t rank, const int64_t *dims, int32_t transform_rank,
                        const int32_t *region, const int32_t *kinds, int32_t precision,
                        int32_t threads);

/* int32_t fftl_plan_execute(fftl_plan p, void *out, const void *in)
 *
 * Executes the plan of any of the four kinds: transforms the elements of in according to the
 * plan's region and stores the result in out. out and in point to separate arrays (out of
 * place) or the same pointer (in place): fftl_plan_execute(p, buf, buf); out and in may not
 * partially overlap. r2c/c2r plans in place use the padded real layout described at
 * fftl_plan_dft_r2c. The buffers hold the
 * arrays described at the plan's creation (complex or real values in the plan's precision).
 *
 * Returns 0, or -1 if the handle is invalid (the buffers are untouched) or the execution failed
 * (out of memory; out is then undefined); fftl_last_error has the reason.
 */
int32_t fftl_plan_execute(fftl_plan p, void *out, const void *in);

/* int32_t fftl_plan_destroy(fftl_plan p)
 *
 * Frees the plan and its tables and buffers. The handle must not be used afterwards; passing
 * it again is rejected as an invalid handle. Destroying is not synchronised with a concurrent
 * execution of the same plan (see "Threading" below).
 */
int32_t fftl_plan_destroy(fftl_plan p);

/* size_t fftl_last_error(char *buf, size_t buf_len)
 *
 * Writes the message of the last failed call of this thread as a zero-terminated string into
 * buf (at most buf_len bytes including the terminator) and returns the full length of the
 * message, like snprintf. An empty message means no error so far. With buf == NULL it only
 * returns the length. Successful calls do not clear the message.
 */
size_t fftl_last_error(char *buf, size_t buf_len);

/* ---------------------------------------------------------------------------
 * Threading
 *
 * libfftl embeds the Julia runtime, whose worker pool is fixed when the library is BUILT and
 * cannot change afterwards: build.sh bakes the number through juliac's --jl-option threads
 * (its FFTL_THREADS variable: a number, or "auto" for one worker per CPU of the machine the
 * library is LOADED on, counted at load time from the process's CPU affinity; the default is
 * "auto", on macOS the number of performance cores of the build machine). JULIA_NUM_THREADS
 * in the environment of the loading process has no effect: the baked option takes precedence.
 * The Julia runtime cannot grow its pool after it starts, so changing the count needs a
 * rebuild.
 *
 * A plan's threads value above the baked worker count is allowed and correct: execution
 * queues that plan's tasks on the available workers, which gives no speedup - and neither
 * does a baked count above the number of cores the process may use. The thread that loads
 * the library is one of the workers: a runtime baked with 4 runs a threads = 4 plan on all
 * 4, while the calling thread waits for them inside fftl_plan_execute.
 *
 * All entry points may be called from any thread of the host program, including threads it
 * created itself, and from several threads at once (the Julia runtime adopts a foreign
 * thread on its first call). Plans are created and destroyed under the library's
 * registry lock; a single plan must not execute on two threads at the same time; different
 * plans can run in parallel. Destroying a plan while it executes is not allowed (the
 * executing thread keeps its own reference until fftl_plan_execute returns, but the handle
 * becomes invalid).
 * ------------------------------------------------------------------------- */


#ifdef __cplusplus
}
#endif

#endif /* FFTL_H */
