#!/bin/bash
# FFTW and libfftl through the same FFTW3 calls (bench.c, linked against each), and FFTL in
# Julia (jbench.jl), alternating, [rounds] rounds (default 2). Further arguments go to bench
# (number of cases, largest thread count). Needs a built library (../build.sh) and the test
# environment (../test, for FFTW_jll). Julia's first plan includes compilation; the library's
# does not. FFTW plans with FFTW_MEASURE (BENCH_ESTIMATE=1: FFTW_ESTIMATE).
set -e
cd "$(dirname "$0")"
LIB="$(cd .. && pwd)/build/lib"
FFTW_D=$(julia --project=../test -e 'using FFTW_jll; print(FFTW_jll.libfftw3_path)')
cc -O2 -o ../build/bench_fftw bench.c -I../include "$FFTW_D" -Wl,-rpath,"$(dirname "$FFTW_D")"
cc -O2 -DLINKED_FFTL -o ../build/bench_fftl bench.c -I../include -L"$LIB" -lfftw3 -Wl,-rpath,"$LIB"
nt=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || nproc)
rounds="${1:-2}"
shift || true
for r in $(seq "$rounds"); do
  ../build/bench_fftw "$@"
  ../build/bench_fftl "$@"
  julia --startup-file=no --project=.. -t "$nt" jbench.jl "$@"
done | sort -k3,3n -k5,5n -k7,7n -k1,1
