/* libfftl: FFTL.jl as a C library, phase 0 probe.
 *
 * Plans by handle for one-dimensional complex DFTs of a fixed set of sizes, single thread.
 * precision: 1 = double, 2 = single. sign: -1 forward, +1 backward (unnormalised).
 * out and in are size complex values (interleaved re, im), column major; they may alias.
 */
#ifndef FFTL_H
#define FFTL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int64_t fftl_plan_create(int32_t precision, int64_t size, int32_t sign);
int32_t fftl_plan_dft(int64_t plan, void *out, const void *in);
int32_t fftl_plan_destroy(int64_t plan);
size_t  fftl_last_error(char *buf, size_t buf_len);

#ifdef __cplusplus
}
#endif

#endif /* FFTL_H */
