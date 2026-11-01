#!/bin/bash
# Trim check of FFTL's public plan/execute calls (T1 in notes/coordination.md): builds the
# entry points in entry.jl with juliac --trim=safe (the same JuliaC environment and flags as
# libfftl/build.sh; juliac exits nonzero while the verifier finds unresolved calls) and
# reports the number of verifier errors per entry point. Baseline of 2026-11-01 and the
# interpretation (construction and run! layer unresolved today, stage codelets resolve,
# phase-2 target 0 errors through RPlan): notes/libfftl.md.
# Usage: ./run.sh   (from this directory)
set -e
cd "$(dirname "$0")"
julia --project=../jc -e 'using Pkg; Pkg.instantiate()' >/dev/null
julia --project=. -e 'using Pkg; Pkg.instantiate()' >/dev/null
if [ -z "$JULIA_CPU_TARGET" ]; then
  # One CPU target: a library built for several targets hangs at the first garbage collection
  # triggered by a second host thread, https://github.com/JuliaLang/JuliaC.jl/issues/190
  case "$(uname -m)" in
    arm64|aarch64) [ "$(uname -s)" = Darwin ] && export JULIA_CPU_TARGET=apple-m1 || export JULIA_CPU_TARGET=generic ;;
    x86_64) export JULIA_CPU_TARGET=x86-64-v3 ;;
    *) export JULIA_CPU_TARGET=generic ;;
  esac
fi
rm -rf build
mkdir -p build
status=0
julia --project=../jc -m JuliaC --output-lib build/libtrimcheck --project . \
  --trim=safe --compile-ccallable --experimental \
  --jl-option threads=1 --jl-option handle-signals=yes "$@" entry.jl >build/trimcheck.log 2>&1 || status=$?
julia --project=. count.jl build/trimcheck.log
if [ "$status" != 0 ]; then
  echo "juliac exit = $status (expected while unresolved calls remain)"
fi
