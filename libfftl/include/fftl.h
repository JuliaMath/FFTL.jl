/* libfftl: FFTL.jl as a C library.
 *
 * Complex DFT plans by opaque handle: any rank, any lengths, both precisions, both signs,
 * in place or out of place, single- or multi-threaded execution. The plan is built, applied
 * and freed through the four calls below; everything else about it is run-time data inside the
 * library (FFT sorts radices, chooses algorithms, owns its work buffers).
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

/* int32_t fftl_plan_execute(fftl_plan p, void *out, const void *in)
 *
 * Executes the plan: transforms the elements of in according to the plan's region and stores
 * the result in out. out and in point to separate arrays (out of place) or are the same
 * pointer (in place): fftl_plan_execute(p, buf, buf); out and in may not partially overlap.
 * The buffers hold the product of dims complex values in the plan's precision.
 *
 * Returns 0, or -1 if the handle is invalid (then fftl_last_error has the reason; the buffers
 * are untouched).
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
 * (its FFTL_THREADS variable: a number - this build uses 4 - or "auto", which bakes one
 * worker per CPU of the machine the library is LOADED on, counted at load time from the
 * process's CPU affinity). The option is applied by the generated loader before the runtime
 * starts; setting JULIA_NUM_THREADS in the environment of the loading process has NO effect
 * on a libfftl built this way (measured both against the baked number and "auto"; the baked
 * option takes precedence, as with the julia command line's -t flag). Rebuild with a
 * different FFTL_THREADS to change it; calling fftl_init-style run-time configuration does
 * not exist because the Julia runtime cannot grow its pool after it starts.
 *
 * A plan's threads value above the baked worker count is allowed and correct: execution
 * queues that plan's tasks on the available workers, which gives no speedup - and neither
 * does a baked count above the number of cores the process may use. The thread that loads
 * the library is one of the workers: a runtime baked with 4 runs a threads = 4 plan on all
 * 4, while the calling thread waits for them inside fftl_plan_execute.
 *
 * All entry points may be called from the loading thread (and, with the usual care for
 * embedded runtimes, from other host threads that the Julia runtime knows about; the plain
 * case is a single caller thread). Plans are created and destroyed under the library's
 * registry lock; a single plan must not execute on two threads at the same time; different
 * plans can run in parallel. Destroying a plan while it executes is not allowed (the
 * executing thread keeps its own reference until fftl_plan_execute returns, but the handle
 * becomes invalid).
 * ------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------------
 * Planned extensions (not implemented yet; the signatures below are sketches
 * for the C interface of the package features named in each section). These
 * exist so that the entry points above stay free of the parameters the
 * extensions need.
 *
 * Pass plans along one dimension with pruning, for callers that transform
 * strided boxes with zero inputs outside given ranges or need only given
 * output ranges (maps FFTL.plan_pass: dimension, in and out strides, nonzero
 * and needed index ranges). Planned shapes:
 *
 *   fftl_plan fftl_plan_pass_dft(int32_t rank, const int64_t *dims,
 *                                const int64_t *in_strides, const int64_t *out_strides,
 *                                int32_t dim, int32_t precision, int32_t sign,
 *                                int32_t threads,
 *                                int32_t n_nonzero, const int64_t *nonzero,
 *                                int32_t n_needed, const int64_t *needed);
 *
 * Real transforms: r2c for a full complex spectrum of half storage and c2r
 * for its (unnormalised) inverse, along a subset of dimensions (maps
 * FFTL.plan_rfft / plan_brfft):
 *
 *   fftl_plan fftl_plan_r2c(...);
 *   fftl_plan fftl_plan_c2r(...);
 * ------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif

#endif /* FFTL_H */
