/* libfftl: FFTL.jl as a C library.
 *
 * The target API (section "general plans") covers the complex DFT of 1D to 3D arrays, both
 * precisions, both signs, single- or multi-threaded execution. It maps onto FFTL.RPlan and
 * will be implemented when the boundary moves to RPlan (phase 3); until then this section is
 * a header-only draft for review. The phase-0 prototype below it is implemented and tested:
 * plans by handle for one-dimensional DFTs of a fixed set of sizes.
 *
 * The description below applies to both sections.
 *
 * Data layout: complex values are interleaved pairs (re, im), stored contiguously in
 * column-major order: in a d1 × … × dr array the first index d1 varies fastest, so the value
 * (i1, …, ir) sits at offset i1 + d1 * (i2 + d2 * (…)). Buffers need no alignment beyond
 * what malloc returns, and out-of-place execution (out != in) may not overlap partially.
 *
 * Precision: the element type (8 or 16 bytes per complex value) is fixed by the plan; buffer
 * pointers are passed as void * and must point to the matching layout.
 *
 * Normalization: nothing is normalised. The backward transform is the plain (unnormalised)
 * inverse map: applying the sign +1 plan after the sign -1 plan multiplies the input by the
 * number of transformed elements.
 *
 * Threads: plans are created and destroyed with the library's registry lock held, safe from
 * any thread. A single plan must not run on two threads at the same time; different plans
 * can run in parallel. Execution uses min(threads, useful) worker threads of the library's
 * own pool.
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
  FFTL_PRECISION_SINGLE = 2    /* Complex{Float32},  8 bytes per value */
};

/* Plans are opaque handles. A negative handle is an error and fftl_last_error explains;
 * the only valid operations on an invalid handle are fftl_last_error and passing it to
 * fftl_plan_destroy, which rejects it. */
typedef int64_t fftl_plan;

/* size_t fftl_version(char *buf, size_t buf_len)
 *
 * Writes the library version as a zero-terminated "major.minor.patch" string into buf (at
 * most buf_len bytes including the terminator) and returns the full length of the version
 * string, like snprintf. With buf == NULL it only returns the length.
 */
size_t fftl_version(char *buf, size_t buf_len);

/* fftl_plan fftl_plan_dft(...)
 *
 * Plan for the complex DFT of a rank-dimensional array of dims[0] × … × dims[rank-1] complex
 * values along the dimensions region[0] … region[transform_rank-1], with the given precision
 * (FFTL_PRECISION_*) and sign (-1 forward, +1 backward, unnormalised).
 *
 * rank                 1 to 3: the rank of the array the buffers point to, dims has this length.
 * dims                 the array dimensions, each >= 1. Currently no prime factor of a
 *                      transformed length may exceed FFTL's maximum radix prime, except that
 *                      any length is allowed if it costs only Bluestein's algorithm; the exact
 *                      rule follows FFTL's planning and will be documented with the RPlan port.
 * transform_rank       0 to rank: the length of region. Zero transforms nothing; executing
 *                      such a plan just copies input to output (in place: no-op).
 * region               transformed dimensions as indices into dims: 0-based (0 is the
 *                      fastest-varying index), strictly ascending, each within 0 .. rank-1.
 * sign                 -1 or +1.
 * threads              number of worker threads for execution, >= 1. The plan stores it;
 *                      execution uses the plan's value, min with what the size allows.
 *
 * Returns an fftl_plan or a negative error code; fftl_last_error has the reason (unsupported
 * precision or sign, rank or region out of range, size not supported, out of memory).
 */
fftl_plan fftl_plan_dft(int32_t rank, const int64_t *dims, int32_t transform_rank,
                        const int32_t *region, int32_t precision, int32_t sign,
                        int32_t threads);

/* int32_t fftl_plan_execute(fftl_plan p, void *out, const void *in)
 *
 * Executes the plan: transforms the elements of in according to the plan's region and stores
 * the result in out. out and in point to separate arrays (out of place) or are the same
 * pointer (in place):  fftl_plan_execute(p, buf, buf); out and in may not partially overlap.
 * The buffers hold prod(dims) complex values in the plan's precision.
 *
 * Returns 0, or -1 if the plan is invalid (then fftl_last_error has the reason; the buffers
 * are untouched). The same signature and semantics serve the phase-0 prototype below; its
 * plans are one-dimensional, so the buffers hold size complex values.
 */
int32_t fftl_plan_execute(fftl_plan p, void *out, const void *in);

/* int32_t fftl_plan_destroy(fftl_plan p)
 *
 * Frees the plan and its tables and buffers. Invalid handles are rejected with -1 and an
 * entry in fftl_last_error. Destroyed handles must not be used again.
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
 * Phase-0 prototype (implemented, tested; will be removed with the RPlan port
 * above, which supersedes it). One-dimensional complex DFTs of a fixed set of
 * sizes (256, 1000, 65536, 1048576, 101), single thread, unnormalised. The
 * executor and the error and destroy entry points are shared with the target
 * API above; the creator below covers only the one fixed sizes and will be
 * replaced by fftl_plan_dft.
 * ------------------------------------------------------------------------- */

int64_t fftl_plan_create(int32_t precision, int64_t size, int32_t sign);
int32_t fftl_plan_execute(fftl_plan p, void *out, const void *in);
int32_t fftl_plan_destroy(fftl_plan p);
size_t  fftl_last_error(char *buf, size_t buf_len);

#ifdef __cplusplus
}
#endif

#endif /* FFTL_H */
