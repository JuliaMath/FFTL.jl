#!/bin/bash
# Builds build/lib/libfftl with juliac --trim (JuliaC.jl in its own environment jc/), compiles
# the C test test/test against it. Needs Julia >= 1.12. FFTL_TRIM=off skips the trim verifier
# (plain --output-lib).
#
# The embedded Julia runtime's worker pool is fixed when the library is built (baked through
# --jl-option; JULIA_NUM_THREADS in the loading process's environment is ignored - the baked
# option takes precedence, including for "auto"). FFTL_THREADS sets the baked count: a number,
# or "auto" for one worker per CPU of the machine the library loads on, counted at load time
# from the process's CPU affinity. Plans may request more threads than are baked; they run
# correctly on fewer workers, which gives no speedup. See include/fftl.h, section Threading.
set -e
cd "$(dirname "$0")"
julia --project=jc -e 'using Pkg; Pkg.instantiate()'
julia --project=jc -e 'using Pkg; print("juliac ", Pkg.dependencies()[Base.UUID("acedd4c2-ced6-4a15-accc-2607eb759ba2")].version)'
julia -e 'println(", Julia ", VERSION)'
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
# One CPU target: a library built for several targets (Julia's default) hangs at the first garbage
# collection triggered by a second host thread, https://github.com/JuliaLang/JuliaC.jl/issues/190.
if [ -z "$JULIA_CPU_TARGET" ]; then
  case "$(uname -m)" in
    arm64|aarch64) [ "$(uname -s)" = Darwin ] && export JULIA_CPU_TARGET=apple-m1 || export JULIA_CPU_TARGET=generic ;;
    x86_64) export JULIA_CPU_TARGET=x86-64-v3 ;;
    *) export JULIA_CPU_TARGET=generic ;;
  esac
fi
# Default: one worker per CPU ("auto"), on macOS one per performance core - workers on the
# efficiency cores hold up the threaded passes (64³ with 8 threads: 300 instead of 214 µs on the
# M1 Max). The macOS count is the build machine's.
if [ -z "$FFTL_THREADS" ]; then
  FFTL_THREADS=auto
  [ "$(uname -s)" = Darwin ] && FFTL_THREADS=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo auto)
fi
# FFTL_TRIM=safe (default): trimmed image, the verifier must resolve every statically reachable
#   call - the state the final library ships in.
# FFTL_TRIM=off: plain --output-lib (AOT entry points, full method tables) - usable for running
#   the C matrix while construction-path idioms still fail the trim verifier.
FFTL_TRIM="${FFTL_TRIM:-safe}"
trim_args=()
[ "$FFTL_TRIM" = "off" ] || trim_args=(--trim="$FFTL_TRIM")
rm -rf build
julia --project=jc -m JuliaC --output-lib build/libfftl --project . --bundle build \
  "${trim_args[@]}" --compile-ccallable --experimental \
  --jl-option threads=$FFTL_THREADS --jl-option handle-signals=yes "$@" src/libfftl.jl
# Libraries of the Julia installation that the library never loads (checked on the Mac with
# DYLD_PRINT_LIBRARIES, as with libnfftjl; re-checked on Linux with ldd after this build):
# BLAS (src/libfftl.jl disables its init), networking, git, C++.
# libLLVM + libjulia-codegen (JIT only, 204 MB of the bundle): measured on 2026-11-01 - the
# full C matrix (build with FFTL_TRIM=off, threads=4, 378 checks: smooth + Bluestein + threaded
# + error paths, both precisions) passes with both moved out of the bundle, so nothing the tests
# execute reaches codegen; a trimmed image is even stricter (the trim verifier guarantees
# everything is AOT-compiled). A missing JIT is also loud: the runtime aborts at the first
# compile, it does not silently misbehave.
for lib in LLVM julia-codegen openblas gfortran quadmath gomp crypto ssl git2 ssh2 curl nghttp2 stdc++ ssp gmpxx dSFMT atomic; do
  rm -f build/lib/julia/lib${lib}*.{so,dylib,dll} build/lib/julia/lib${lib}*.so.* build/bin/lib${lib}*.dll
done
cc -O2 -o test/test test/test.c -Iinclude -Lbuild/lib -lfftl -lm -Wl,-rpath,"$(pwd)/build/lib"
