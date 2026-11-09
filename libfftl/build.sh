#!/bin/bash
# Builds build/lib/libfftl with juliac --trim (JuliaC.jl in its own environment jc/) and the
# FFTW library names as links to it. Needs Julia >= 1.12. FFTL_TRIM=off skips the trim verifier
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
# FFTL_TRIM=off: plain --output-lib (AOT entry points, full method tables), for diagnosing a
#   failing trim build with the C tests.
FFTL_TRIM="${FFTL_TRIM:-safe}"
trim_args=()
[ "$FFTL_TRIM" = "off" ] || trim_args=(--trim="$FFTL_TRIM")
rm -rf build
# juliac compiles nothing at run time, so every radix codelet must be compiled with the package.
# The library is therefore built from a copy of the package in which LAZY_RADICES is empty
# (build/fftl-trim), in an environment that points to that copy (build/trim-env).
mkdir -p build/fftl-trim build/trim-env
cp -R ../Project.toml ../src ../ext build/fftl-trim/
sed -i.bak 's/^const LAZY_RADICES = .*/const LAZY_RADICES = ()/' build/fftl-trim/src/rchain.jl
rm -f build/fftl-trim/src/rchain.jl.bak
grep -q '^const LAZY_RADICES = ()$' build/fftl-trim/src/rchain.jl || { echo "LAZY_RADICES substitution failed"; exit 1; }
sed 's|path = ".."|path = "../fftl-trim"|' Project.toml > build/trim-env/Project.toml
sed 's|path = ".."|path = "../fftl-trim"|' Manifest.toml > build/trim-env/Manifest.toml
cp LocalPreferences.toml build/trim-env/
julia --project=build/trim-env -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
julia --project=jc -m JuliaC --output-lib build/libfftl --project build/trim-env --bundle build \
  "${trim_args[@]}" --compile-ccallable --experimental \
  --jl-option threads=$FFTL_THREADS --jl-option handle-signals=yes "$@" src/libfftl.jl
# Libraries of the Julia installation that the library never loads (checked on the Mac with
# DYLD_PRINT_LIBRARIES, as with libnfftjl; re-checked on Linux with ldd after this build):
# BLAS (src/libfftl.jl disables its init), networking, git, C++.
# libLLVM and libjulia-codegen serve the JIT only (≈ 200 MB): the trimmed image compiles nothing
# at run time, and the C tests pass without them; a missing JIT would abort at the first compile.
for lib in LLVM julia-codegen openblas gfortran quadmath gomp crypto ssl git2 ssh2 curl nghttp2 stdc++ ssp gmpxx dSFMT atomic; do
  rm -f build/lib/julia/lib${lib}*.{so,dylib,dll} build/lib/julia/lib${lib}*.so.* build/bin/lib${lib}*.dll
done
# The FFTW3 interface (include/fftw3.h): libfftl exports fftw_* and fftwf_*, so the FFTW library
# names are links to it. Programs link with -lfftw3 / -lfftw3f (and -lfftw3_threads or
# -lfftw3_omp, which FFTW keeps separate) or load libfftw3.so.3 / libfftw3.3.dylib by name.
if [ "$(uname -s)" = Darwin ]; then
  for name in fftw3 fftw3f fftw3_threads fftw3f_threads fftw3_omp fftw3f_omp; do
    ln -sf libfftl.dylib build/lib/lib$name.dylib
    ln -sf libfftl.dylib build/lib/lib$name.3.dylib
  done
else
  for name in fftw3 fftw3f fftw3_threads fftw3f_threads fftw3_omp fftw3f_omp; do
    ln -sf libfftl.so build/lib/lib$name.so
    ln -sf libfftl.so build/lib/lib$name.so.3
  done
fi
