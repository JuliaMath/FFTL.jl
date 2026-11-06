#!/bin/bash
# The Fortran test of the FFTW3 interface (test/fortran): the legacy interface (dfftw_*,
# sfftw_* with fftw3.f) and the Fortran 2003 interface (fftw3.f03) in both precisions,
# against direct sums. Built twice with gfortran (FC overrides it), against FFTW (FFTW_jll,
# whose libraries include FFTW's legacy Fortran wrappers) and against libfftl's
# libfftw3/libfftw3f links (build/lib, made by build.sh). Run after build.sh, from anywhere.
set -e
cd "$(dirname "$0")/.."
FC="${FC:-gfortran}"
# Homebrew's gfortran on macOS finds the system libraries only through the SDK
if [ "$(uname -s)" = Darwin ] && [ -z "$SDKROOT" ]; then
  SDKROOT=$(xcrun --show-sdk-path 2>/dev/null || true)
  export SDKROOT
fi
julia --project=test -e 'using Pkg; Pkg.instantiate()'
FFTW_D=$(julia --project=test -e 'using FFTW_jll; print(FFTW_jll.libfftw3_path)')
FFTW_F=$(julia --project=test -e 'using FFTW_jll; print(FFTW_jll.libfftw3f_path)')
FFTW_DIR=$(dirname "$FFTW_D")
LIB="$(pwd)/build/lib"
OUT=build/fftw3_ftest
rm -rf "$OUT"
mkdir -p "$OUT"
# one source per precision from the templates
sed -e 's/Xfftw_/dfftw_/g; s/(WP)/(8)/g; s/, WP)/, 8)/g; s/PREC/d/g; s/TOLV/1d-11/g' test/fortran/legacy.f90.in > "$OUT/legacy_d.f90"
sed -e 's/Xfftw_/sfftw_/g; s/(WP)/(4)/g; s/, WP)/, 4)/g; s/PREC/s/g; s/TOLV/2d-5/g' test/fortran/legacy.f90.in > "$OUT/legacy_s.f90"
sed -e 's/fftwY_/fftw_/g; s/RKIND/C_DOUBLE/g; s/CKIND/C_DOUBLE_COMPLEX/g; s/IODIM/fftw_iodim/g; s/PREC/d/g; s/TOLV/1d-11/g' test/fortran/f03.f90.in > "$OUT/f03_d.f90"
sed -e 's/fftwY_/fftwf_/g; s/RKIND/C_FLOAT/g; s/CKIND/C_FLOAT_COMPLEX/g; s/IODIM/fftwf_iodim/g; s/PREC/s/g; s/TOLV/2d-5/g' test/fortran/f03.f90.in > "$OUT/f03_s.f90"
srcs=(test/fortran/ref.f90 test/fortran/wisdom_char.f90 "$OUT/legacy_d.f90" "$OUT/legacy_s.f90" test/fortran/mangling.f90
      "$OUT/f03_d.f90" "$OUT/f03_s.f90" test/fortran/main.f90)
rpath_flags=()
[ "$(uname -s)" = Darwin ] || rpath_flags=(-Wl,--enable-new-dtags)
"$FC" -O1 -Iinclude -J"$OUT" -o "$OUT/ftest_fftw" "${srcs[@]}" "$FFTW_D" "$FFTW_F" "${rpath_flags[@]}" -Wl,-rpath,"$FFTW_DIR"
"$FC" -O1 -Iinclude -J"$OUT" -o "$OUT/ftest_fftl" "${srcs[@]}" -L"$LIB" -lfftw3 -lfftw3f "${rpath_flags[@]}" -Wl,-rpath,"$LIB"
status=0
for lib in fftw fftl; do
  echo "linked against $lib:"
  (cd "$OUT" && "./ftest_$lib") || status=1
done
exit $status
