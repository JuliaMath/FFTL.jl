#!/bin/bash
# Builds the trim probe of the run-time pass plans (passes.jl) with juliac --trim=safe, in the
# environment of libfftl. Fails on any trim verifier error.
set -e
cd "$(dirname "$0")/.."
julia --project=jc -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
if [ -z "$JULIA_CPU_TARGET" ]; then
  case "$(uname -m)" in
    arm64|aarch64) [ "$(uname -s)" = Darwin ] && export JULIA_CPU_TARGET=apple-m1 || export JULIA_CPU_TARGET=generic ;;
    x86_64) export JULIA_CPU_TARGET=x86-64-v3 ;;
    *) export JULIA_CPU_TARGET=generic ;;
  esac
fi
rm -rf probe/build
julia --project=jc -m JuliaC --output-lib probe/build/libfftlprobe --project . --bundle probe/build \
  --trim=safe --compile-ccallable --experimental probe/passes.jl
