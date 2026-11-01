#!/bin/bash
# Interleaved timing of libfftl from C against FFTL in Julia, variant by variant: for every
# variant one C measurement (one process, the precompiled library) alternates with one Julia
# measurement (fresh process per variant and round, because expanding run! for many chain
# variants in one Julia process crashed the GC on this shared machine), so machine drift
# cancels and each side stays in a small, short-lived process. Requires the built library
# (../build.sh; JULIA_CPU_TARGET as set there). Waits for enough free memory, retries a
# killed/broken measurement once.
# Usage: run.sh [rounds]   (default 5)
set -e
cd "$(dirname "$0")"
[ -f ../build/lib/libfftl.dylib ] || [ -f ../build/lib/libfftl.so ] ||
  { echo "no library in ../build/lib, run ../build.sh first"; exit 1; }
ROUNDS=${1:-5}
cc -O2 -o ../build/bench bench.c -I../include -L../build/lib -lfftl -lm \
  -Wl,-rpath,"$(cd .. && pwd)/build/lib"

freepct() { memory_pressure -Q 2>/dev/null | awk '/free percentage/ {gsub(/%/, ""); print $NF}'; }
freerss() {  # foreign Julia processes with > 300 MB, in GB
  ps aux | awk 'NR>1 && $11 ~ /julia/ && $0 !~ /jbench/ && $6 > 300000 {s += $6} END {printf "%d", s / 1048576}'
}
waitmem() {
  while [ "$(freepct)" -lt 55 ] || [ "$(freerss)" -gt 5 ]; do
    echo "busy machine: free=$(freepct)%, foreign julia $(freerss) GB, waiting 60 s" >&2
    sleep 60
  done
}

OUT=../build/bench.txt
rm -f "$OUT"
for r in $(seq "$ROUNDS"); do
  for prec in 1 2; do
    for n in 256 1000 65536 1048576 101; do
      for sign in -1 1; do
        waitmem
        ../build/bench "$prec" "$n" "$sign" >>"$OUT"
        waitmem
        if ! julia --project=.. jbench.jl "$prec" "$n" "$sign" >>"$OUT"; then
          echo "jbench $prec $n $sign failed, retrying once" >&2
          julia --project=.. jbench.jl "$prec" "$n" "$sign" >>"$OUT"
        fi
      done
    done
  done
done
julia --project=.. aggreg.jl "$OUT"
