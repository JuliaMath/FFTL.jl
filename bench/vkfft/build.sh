#!/bin/bash
# Builds vkfft_bench against VkFFT (main branch, MIT; a clone of it anywhere, passed as the first
# argument; only its public header and metal-cpp are used).
#   bench/vkfft/build.sh [vkfft-dir]
# macOS: Metal backend; Linux: CUDA backend (CUDA toolkit with NVRTC in /usr/local/cuda).
set -euo pipefail
cd "$(dirname "$0")"
V=${1:-../../refs/VkFFT}
if [ "$(uname)" = Darwin ]; then
  clang++ -O2 -std=c++17 -DVKFFT_BACKEND=5 -I"$V/vkFFT" -I"$V/metal-cpp" -w vkfft_bench.cpp -o vkfft_bench \
    -framework Metal -framework Foundation -framework QuartzCore
else
  CU=${CUDA_HOME:-/usr/local/cuda}
  g++ -O2 -std=c++17 -DVKFFT_BACKEND=1 -I"$V/vkFFT" -I"$CU/include" -w vkfft_bench.cpp -o vkfft_bench \
    -L"$CU/lib64" -lcuda -lcudart -lnvrtc -Wl,-rpath,"$CU/lib64"
fi
echo "built $(pwd)/vkfft_bench"
