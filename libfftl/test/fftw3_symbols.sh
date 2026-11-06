#!/bin/bash
# Checks that the built library exports every function of test/fftw3_symbols.txt in both
# precisions (fftw_ and fftwf_), so that a missing entry point fails here and not when some
# program links. Run after build.sh, from anywhere.
set -e
cd "$(dirname "$0")/.."
if [ "$(uname -s)" = Darwin ]; then
  syms=$(nm -gU build/lib/libfftl.dylib | awk '{print $3}' | sed 's/^_//')
else
  syms=$(nm -D --defined-only build/lib/libfftl.so | awk '{print $3}')
fi
missing=0
total=0
for name in $(grep -v '^#' test/fftw3_symbols.txt); do
  for prefix in fftw_ fftwf_; do
    total=$((total + 1))
    if ! grep -qx "$prefix$name" <<< "$syms"; then
      echo "missing: $prefix$name"
      missing=$((missing + 1))
    fi
  done
done
echo "fftw3_symbols: $total functions checked, $missing missing"
[ $missing -eq 0 ]
