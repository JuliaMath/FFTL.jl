#!/bin/bash
# Builds cufft_bench against the CUDA toolkit's cuFFT (CUDA_HOME, default /usr/local/cuda).
#   bench/cufft/build.sh
set -euo pipefail
cd "$(dirname "$0")"
CU=${CUDA_HOME:-/usr/local/cuda}
g++ -O2 -std=c++17 -I"$CU/include" cufft_bench.cpp -o cufft_bench -L"$CU/lib64" -lcufft -lcudart -Wl,-rpath,"$CU/lib64"
echo "built $(pwd)/cufft_bench"
