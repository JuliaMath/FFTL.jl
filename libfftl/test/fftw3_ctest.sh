#!/bin/bash
# The differential C test of the FFTW3 interface. test/fftw3_test.c is compiled against
# include/fftw3.h four times: linked against FFTW (FFTW_jll, from the test environment) or
# against libfftl's libfftw3/libfftw3f links (build/lib, made by build.sh), each with
# fftw_complex as double[2] and as the C99 complex type. Every binary checks its results
# against direct sums; test/fftw3_compare.jl then compares the outputs of the FFTW and libfftl
# builds. The FFTW-linked binary also runs unchanged on libfftl (library path pointing to
# build/lib, where libfftw3.so.3 / libfftw3.3.dylib are links to libfftl). Run after build.sh,
# from anywhere.
set -e
cd "$(dirname "$0")/.."
julia --project=test -e 'using Pkg; Pkg.instantiate()'
FFTW_D=$(julia --project=test -e 'using FFTW_jll; print(FFTW_jll.libfftw3_path)')
FFTW_F=$(julia --project=test -e 'using FFTW_jll; print(FFTW_jll.libfftw3f_path)')
FFTW_DIR=$(dirname "$FFTW_D")
LIB="$(pwd)/build/lib"
# RUNPATH rather than RPATH on Linux, so that LD_LIBRARY_PATH takes precedence (drop-in run below)
rpath_flags=()
[ "$(uname -s)" = Darwin ] || rpath_flags=(-Wl,--enable-new-dtags)
OUT=build/fftw3_test
rm -rf "$OUT"
for variant in plain complex; do
  flags=()
  [ $variant = complex ] && flags=(-DWITH_COMPLEX_H)
  mkdir -p "$OUT/fftw-$variant" "$OUT/fftl-$variant"
  cc -O2 "${flags[@]}" -Iinclude -o "$OUT/fftw3_test_fftw_$variant" test/fftw3_test.c \
    "$FFTW_D" "$FFTW_F" -lm -lpthread "${rpath_flags[@]}" -Wl,-rpath,"$FFTW_DIR"
  cc -O2 "${flags[@]}" -DLINKED_FFTL -Iinclude -o "$OUT/fftw3_test_fftl_$variant" test/fftw3_test.c \
    -L"$LIB" -lfftw3 -lfftw3f -lm -lpthread "${rpath_flags[@]}" -Wl,-rpath,"$LIB"
done
status=0
for variant in plain complex; do
  for lib in fftw fftl; do
    "$OUT/fftw3_test_${lib}_$variant" "$OUT/$lib-$variant" || status=1
  done
done
mkdir -p "$OUT/dropin-plain"
if [ "$(uname -s)" = Darwin ]; then
  DYLD_LIBRARY_PATH="$LIB" "$OUT/fftw3_test_fftw_plain" "$OUT/dropin-plain" || status=1
else
  LD_LIBRARY_PATH="$LIB" "$OUT/fftw3_test_fftw_plain" "$OUT/dropin-plain" || status=1
fi
julia --project=test test/fftw3_compare.jl "$OUT" || status=1
exit $status
