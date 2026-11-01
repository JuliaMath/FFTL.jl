#!/bin/bash
# libfftl from C against FFTL in Julia, alternating, [rounds] rounds (default 2). Needs a built
# library (../build.sh). Julia's first plan includes compilation; the library's does not.
set -e
cd "$(dirname "$0")"
cc -O2 -o ../build/bench bench.c -I../include -L../build/lib -lfftl -Wl,-rpath,"$(cd .. && pwd)/build/lib"
nt=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || nproc)
for r in $(seq "${1:-2}"); do
  ../build/bench
  julia --startup-file=no --project=.. -t "$nt" jbench.jl
done | sort -k3,3n -k5,5n -k7,7n -k1,1
