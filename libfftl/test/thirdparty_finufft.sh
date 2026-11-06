#!/bin/bash
# An unmodified third-party program on libfftl: FINUFFT (https://github.com/flatironinstitute/finufft,
# Apache-2.0), a C++ library for nonuniform FFTs that plans its FFTs through FFTW3's C
# interface. Its own test suite (make test: the transforms against direct sums) is built twice
# from the same checkout and against libfftl's fftw3.h: linked against libfftl's
# libfftw3/libfftw3f links and against FFTW (FFTW_jll). Both must pass, with the same summary.
# Single-threaded build (OMP=OFF), so that no OpenMP runtime is needed. Run after build.sh.
set -e
cd "$(dirname "$0")/.."
FINUFFT_COMMIT=679d9ae59fe0146c50da360e5a3fee70ae3aa646   # tag v2.5.1
julia --project=test -e 'using Pkg; Pkg.instantiate()'
FFTW_D=$(julia --project=test -e 'using FFTW_jll; print(FFTW_jll.libfftw3_path)')
FFTW_DIR=$(dirname "$FFTW_D")
LIB="$(pwd)/build/lib"
INC="$(pwd)/include"
OUT="$(pwd)/build/thirdparty"
rm -rf "$OUT"
mkdir -p "$OUT"
git clone -q https://github.com/flatironinstitute/finufft.git "$OUT/src"
git -C "$OUT/src" checkout -q "$FINUFFT_COMMIT"
status=0
for lib in fftl fftw; do
  cp -R "$OUT/src" "$OUT/$lib"
  dir=$([ $lib = fftl ] && echo "$LIB" || echo "$FFTW_DIR")
  (cd "$OUT/$lib" && make test OMP=OFF LTO=OFF CXX=c++ CC=cc CFLAGS="-O2 -I$INC" \
     LIBS="-lm -L$dir -Wl,-rpath,$dir" > "$OUT/$lib.log" 2>&1) || status=1
  echo "FINUFFT linked against $lib:"
  grep -E "fails out of|segfaults out of" "$OUT/$lib.log" || { tail -20 "$OUT/$lib.log"; status=1; }
done
diff <(grep -E "fails out of|segfaults out of" "$OUT/fftl.log") <(grep -E "fails out of|segfaults out of" "$OUT/fftw.log") || status=1
exit $status
