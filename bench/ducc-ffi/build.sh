#!/bin/bash
# Builds libducc_ffi against the ducc0 source tree, with native flags. Only ducc0/fft/fft.h is
# included by ducc_ffi.cc; the instantiation units and infrastructure files are needed to link.
#   bench/ducc-ffi/build.sh <ducc-src-dir (contains ducc0/)>
set -euo pipefail
src=$1
here=$(cd "$(dirname "$0")" && pwd)
arch=-march=native; [ "$(uname -m)" = arm64 ] && arch=-mcpu=native
ext=so; [ "$(uname)" = Darwin ] && ext=dylib
${CXX:-g++} -O3 $arch -std=c++17 -ffp-contract=fast -fPIC -shared -pthread -I"$src" \
  "$here/ducc_ffi.cc" "$src/ducc0/fft/fft_inst1.cc" "$src/ducc0/fft/fft_inst2.cc" \
  "$src/ducc0/infra/mav.cc" "$src/ducc0/infra/threading.cc" "$src/ducc0/infra/string_utils.cc" \
  -o "$here/libducc_ffi.$ext"
echo "built $here/libducc_ffi.$ext"
