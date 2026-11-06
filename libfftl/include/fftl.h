/* libfftl: FFTL.jl as a C library.
 *
 * The transforms are FFTW3's C interface: include fftw3.h (next to this file), link with
 * -lfftl, or with -lfftw3 / -lfftw3f, which are links to the same library. fftw3.h lists what
 * differs from FFTW. This header declares what FFTW has no counterpart for: the message of
 * the last failed call, the library version, and the rules of the worker pool baked into the
 * library (Threading, below).
 */
#ifndef FFTL_H
#define FFTL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* size_t fftl_last_error(char *buf, size_t buf_len)
 *
 * Writes the message of the last failed call of this thread as a zero-terminated string into
 * buf (at most buf_len bytes including the terminator) and returns the full length of the
 * message, like snprintf. An empty message means no error so far. With buf == NULL it only
 * returns the length. Successful calls do not clear the message.
 *
 * Every NULL a planner of fftw3.h returns leaves its reason here, as do execute functions
 * called with an invalid plan or with arrays the plan cannot take (FFTW returns nothing from
 * them), fftw_destroy_plan and fftw_copy_plan of an invalid plan, and failed allocations.
 */
size_t fftl_last_error(char *buf, size_t buf_len);

/* size_t fftl_version(char *buf, size_t buf_len)
 *
 * Writes the library version as a zero-terminated "major.minor.patch" string into buf (at
 * most buf_len bytes including the terminator) and returns the full length of the version
 * string, like snprintf. With buf == NULL it only returns the length.
 */
size_t fftl_version(char *buf, size_t buf_len);

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
 * fftw_plan_with_nthreads sets how many threads later plans use. A count above the baked
 * worker count is allowed and correct: execution queues that plan's tasks on the available
 * workers, which gives no speedup - and neither does a baked count above the number of cores
 * the process may use. The thread that loads the library is one of the workers: a runtime
 * baked with 4 runs a plan with 4 threads on all 4, while the calling thread waits for them
 * inside the execute function. Small transforms run on one thread regardless.
 *
 * All entry points may be called from any thread of the host program, including threads it
 * created itself, and from several threads at once (the Julia runtime adopts a foreign
 * thread on its first call). Plans are created and destroyed under the library's registry
 * lock. One plan may execute on several threads at once through the new-array execute
 * functions, as in FFTW: each execution takes its own copy of the plan's work buffers.
 * Destroying a plan while it executes is not allowed.
 * ------------------------------------------------------------------------- */

#ifdef __cplusplus
}
#endif

#endif /* FFTL_H */
