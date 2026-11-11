// cuFFT benchmark harness, the counterpart of bench/vkfft/vkfft_bench.cpp: one transform per
// process, data through files, timings on stdout; the plan is made once and reused.
//
//   cufft_bench <kind> <f32|f64> <in.bin> <out.bin> <reps> <samples> n1 [n2 [n3]] [bB]
//
// kind: c2c (forward), bc2c (backward, unnormalised), r2c, c2r (unnormalised). Arrays are column
// major (n1 contiguous, as in Julia), so cuFFT gets the dimensions in reverse order; r2c/c2r use
// the unpadded layout (n1 real, n1/2+1 complex values per line). A last argument bB transforms B
// arrays one after another (batch). All transforms are out of place; cuFFT's c2r may overwrite
// its input for more than one dimension, so the input is reloaded before each sample (outside the
// timing), as in the VkFFT harness.
//
// Output, one line: plan_ms first_ms exec_us; plan_ms is cufftPlanMany (including cuFFT's
// kernel setup), first_ms the first execution including its synchronisation, exec_us the median
// over <samples> of the mean time per transform of <reps> back-to-back executions with one
// synchronisation.
//
// Build: bench/cufft/build.sh
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }
static void die(const char* what, long code) {
  fprintf(stderr, "cufft_bench: %s failed (%ld)\n", what, code);
  exit(1);
}
#define CHECK(call, what) do { long r_ = (long)(call); if (r_ != 0) die(what, r_); } while (0)

int main(int argc, char** argv) {
  if (argc < 8) { fprintf(stderr, "usage: cufft_bench kind f32|f64 in out reps samples n1 [n2 [n3]] [bB]\n"); return 2; }
  std::string kind = argv[1];
  bool dbl = std::string(argv[2]) == "f64";
  const char* fin = argv[3]; const char* fout = argv[4];
  int reps = atoi(argv[5]), samples = atoi(argv[6]);
  int dim = argc - 7;
  long long batch = 1;
  if (argv[argc - 1][0] == 'b') { batch = atoll(argv[argc - 1] + 1); dim--; }
  if (dim < 1 || dim > 3) die("dimension", dim);
  long long n[3] = {1, 1, 1};
  for (int d = 0; d < dim; d++) n[d] = atoll(argv[7 + d]);
  long long per = n[0] * n[1] * n[2], h = n[0] / 2 + 1, perh = h * (per / n[0]);
  size_t es = dbl ? 8 : 4;
  bool r2c = kind == "r2c", c2r = kind == "c2r";
  if (!(r2c || c2r || kind == "c2c" || kind == "bc2c")) die("kind", 0);
  size_t bytes_c = 2 * es * per * batch, bytes_r = es * per * batch, bytes_h = 2 * es * perh * batch;
  size_t in_bytes = r2c ? bytes_r : c2r ? bytes_h : bytes_c;
  size_t out_bytes = r2c ? bytes_h : c2r ? bytes_r : bytes_c;

  std::vector<char> hin(in_bytes), hout(out_bytes);
  FILE* f = fopen(fin, "rb"); if (!f || fread(hin.data(), 1, in_bytes, f) != in_bytes) die("reading input", 0); fclose(f);
  CHECK(cudaSetDevice(0), "cudaSetDevice");
  void *din, *dout;
  CHECK(cudaMalloc(&din, in_bytes), "cudaMalloc"); CHECK(cudaMalloc(&dout, out_bytes), "cudaMalloc");
  CHECK(cudaMemcpy(din, hin.data(), in_bytes, cudaMemcpyHostToDevice), "HtoD");
  CHECK(cudaDeviceSynchronize(), "sync");

  // cuFFT dimensions: slowest first (row-major), i.e. the column-major sizes reversed
  long long nn[3];
  for (int d = 0; d < dim; d++) nn[d] = n[dim - 1 - d];
  cufftType type = r2c ? (dbl ? CUFFT_D2Z : CUFFT_R2C) : c2r ? (dbl ? CUFFT_Z2D : CUFFT_C2R) : (dbl ? CUFFT_Z2Z : CUFFT_C2C);
  long long idist = r2c ? per : c2r ? perh : per, odist = r2c ? perh : c2r ? per : per;
  cufftHandle plan;
  size_t work = 0;
  auto t0 = clk::now();
  CHECK(cufftCreate(&plan), "cufftCreate");
  CHECK(cufftMakePlanMany64(plan, dim, nn, nullptr, 1, idist, nullptr, 1, odist, type, batch, &work), "cufftMakePlanMany64");
  double plan_ms = ms_since(t0);
  int dir = kind == "bc2c" ? CUFFT_INVERSE : CUFFT_FORWARD;
  auto run = [&]() {
    if (type == CUFFT_C2C) CHECK(cufftExecC2C(plan, (cufftComplex*)din, (cufftComplex*)dout, dir), "exec");
    else if (type == CUFFT_Z2Z) CHECK(cufftExecZ2Z(plan, (cufftDoubleComplex*)din, (cufftDoubleComplex*)dout, dir), "exec");
    else if (type == CUFFT_R2C) CHECK(cufftExecR2C(plan, (cufftReal*)din, (cufftComplex*)dout), "exec");
    else if (type == CUFFT_D2Z) CHECK(cufftExecD2Z(plan, (cufftDoubleReal*)din, (cufftDoubleComplex*)dout), "exec");
    else if (type == CUFFT_C2R) CHECK(cufftExecC2R(plan, (cufftComplex*)din, (cufftReal*)dout), "exec");
    else CHECK(cufftExecZ2D(plan, (cufftDoubleComplex*)din, (cufftDoubleReal*)dout), "exec");
  };
  t0 = clk::now();
  run(); CHECK(cudaDeviceSynchronize(), "sync");
  double first_ms = ms_since(t0);
  CHECK(cudaMemcpy(hout.data(), dout, out_bytes, cudaMemcpyDeviceToHost), "DtoH");
  f = fopen(fout, "wb"); fwrite(hout.data(), 1, out_bytes, f); fclose(f);

  std::vector<double> ts;
  for (int s = 0; s < samples; s++) {
    if (c2r) CHECK(cudaMemcpy(din, hin.data(), in_bytes, cudaMemcpyHostToDevice), "HtoD");
    run(); CHECK(cudaDeviceSynchronize(), "sync");
    t0 = clk::now();
    for (int r = 0; r < reps; r++) run();
    CHECK(cudaDeviceSynchronize(), "sync");
    ts.push_back(ms_since(t0) * 1e3 / reps);
  }
  std::sort(ts.begin(), ts.end());
  printf("%.3f %.3f %.3f\n", plan_ms, first_ms, ts[ts.size() / 2]);
  cufftDestroy(plan); cudaFree(din); cudaFree(dout);
  return 0;
}
