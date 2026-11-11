// VkFFT benchmark harness: one transform per process, data through files, timings on stdout.
//
//   vkfft_bench <kind> <f32|f64> <in.bin> <out.bin> <reps> <samples> <cache> n1 [n2 [n3]] [bB]
//
// A last argument bB transforms B arrays of size n1×n2×n3 stored one after another (batch).
// kind: c2c (forward), bc2c (backward, unnormalised), r2c, c2r (unnormalised), dct1…dct4,
// dst1…dst4. Arrays are column major (n1 contiguous, as in Julia); r2c/c2r use the unpadded
// layout of FFTL and FFTW (n1 real, n1/2+1 complex values per line), every transform is out of
// place and leaves its input unchanged except c2r with more than one dimension (see below).
// cache: a file for VkFFT's compiled kernels ("-" for none; CUDA only, the Metal backend has no
// binary cache): loaded when it exists, written otherwise.
//
// Output, one line: plan_ms first_ms exec_us cached batch_us; plan_ms is initializeVkFFT (kernel
// generation and run-time compilation, or loading the cache), first_ms the first execution
// including its synchronisation, exec_us the median over <samples> of the mean time per
// transform of <reps> back-to-back transforms with one synchronisation (on Metal one command
// buffer per transform, as a library call issues it), batch_us the same with all <reps>
// transforms in one command buffer and encoder (Metal; equal to exec_us on CUDA).
//
// Build: bench/vkfft/build.sh (backend 1 = CUDA, 5 = Metal).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if VKFFT_BACKEND == 1
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#endif
#include "vkFFT.h"

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

static void die(const char* what, long code) {
  fprintf(stderr, "vkfft_bench: %s failed (%ld)\n", what, code);
  exit(1);
}
#define CHECK(call, what) do { long r_ = (long)(call); if (r_ != 0) die(what, r_); } while (0)

// device memory of one backend
struct Dev {
#if VKFFT_BACKEND == 1
  CUdevice dev;
  CUcontext ctx;
  void init() {
    CHECK(cuInit(0), "cuInit");
    CHECK(cuDeviceGet(&dev, 0), "cuDeviceGet");
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    CHECK(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
  }
  typedef void* Buf;   // holds a CUdeviceptr
  Buf alloc(size_t n) { CUdeviceptr p; CHECK(cuMemAlloc(&p, n), "cuMemAlloc"); CHECK(cuMemsetD8(p, 0, n), "cuMemset"); return (void*)p; }
  void put(Buf b, const void* h, size_t n) { CHECK(cuMemcpyHtoD((CUdeviceptr)b, h, n), "HtoD"); }
  void get(void* h, Buf b, size_t n) { CHECK(cuMemcpyDtoH(h, (CUdeviceptr)b, n), "DtoH"); }
  void sync() { CHECK(cuCtxSynchronize(), "sync"); }
  void config(VkFFTConfiguration& c) { c.device = &dev; }
#elif VKFFT_BACKEND == 5
  MTL::Device* dev;
  MTL::CommandQueue* queue;
  MTL::CommandBuffer* cb = nullptr;
  void init() {
    NS::Array* devices = MTL::CopyAllDevices();
    dev = (MTL::Device*)devices->object(0);
    queue = dev->newCommandQueue();
  }
  typedef MTL::Buffer* Buf;
  Buf alloc(size_t n) { Buf b = dev->newBuffer(n, MTL::ResourceStorageModeShared); memset(b->contents(), 0, n); return b; }
  void put(Buf b, const void* h, size_t n) { memcpy(b->contents(), h, n); }
  void get(void* h, Buf b, size_t n) { memcpy(h, b->contents(), n); }
  void sync() { if (cb) { cb->waitUntilCompleted(); cb = nullptr; } }
  void config(VkFFTConfiguration& c) { c.device = dev; c.queue = queue; }
#endif
};

int main(int argc, char** argv) {
  if (argc < 9) { fprintf(stderr, "usage: vkfft_bench kind f32|f64 in out reps samples cache n1 [n2 [n3]] [bB]\n"); return 2; }
  std::string kind = argv[1];
  bool dbl = std::string(argv[2]) == "f64";
  const char* fin = argv[3]; const char* fout = argv[4];
  int reps = atoi(argv[5]), samples = atoi(argv[6]);
  std::string cache = argv[7];
  int dim = argc - 8;
  uint64_t batch = 1;
  if (argv[argc - 1][0] == 'b') { batch = strtoull(argv[argc - 1] + 1, nullptr, 10); dim--; }
  uint64_t n[3] = {1, 1, 1};
  for (int d = 0; d < dim; d++) n[d] = strtoull(argv[8 + d], nullptr, 10);
  uint64_t total = n[0] * n[1] * n[2] * batch;
  size_t es = dbl ? 8 : 4;                       // bytes per real value
  uint64_t h = n[0] / 2 + 1, rest = total / n[0];

  bool r2c = kind == "r2c", c2r = kind == "c2r", real = r2c || c2r;
  bool r2r = kind.rfind("dct", 0) == 0 || kind.rfind("dst", 0) == 0;
  int inverse = (kind == "bc2c" || c2r) ? 1 : -1;
  size_t bytes_c = 2 * es * total;               // complex array
  size_t bytes_r = es * total;                   // real array
  size_t bytes_h = 2 * es * h * rest;            // half spectrum
  size_t in_bytes = r2r ? bytes_r : r2c ? bytes_r : c2r ? bytes_h : bytes_c;
  size_t out_bytes = r2r ? bytes_r : r2c ? bytes_h : c2r ? bytes_r : bytes_c;

#if VKFFT_BACKEND == 5
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
#endif
  Dev D; D.init();

  std::vector<char> hin(in_bytes), hout(out_bytes);
  FILE* f = fopen(fin, "rb"); if (!f || fread(hin.data(), 1, in_bytes, f) != in_bytes) die("reading input", 0); fclose(f);

  VkFFTConfiguration c = {};
  VkFFTApplication app = {};
  c.FFTdim = dim;
  c.numberBatches = batch;
  for (int d = 0; d < dim; d++) c.size[d] = n[d];
  c.doublePrecision = dbl;
  D.config(c);
  if (r2r) {
    int t = kind[3] - '0';
    if (kind[1] == 'c') c.performDCT = t; else c.performDST = t;
  }
  // Buffers: out of place throughout. r2c reads the real input buffer (stride n1) and writes the
  // half spectrum to the main buffer (stride n1/2+1). c2r runs as the inverse of that plan with
  // the result returned to the real input buffer; it reads the half spectrum from the main
  // buffer, which it may overwrite for more than one dimension (it is reloaded before each
  // sample, outside the timing).
  Dev::Buf bin, bmain;
  uint64_t szin, szmain;
  if (real) {
    c.performR2C = 1;
    c.isInputFormatted = 1;
    c.inputBufferStride[0] = n[0];
    for (int d = 1; d < dim; d++) c.inputBufferStride[d] = c.inputBufferStride[d - 1] * n[d];
    c.bufferStride[0] = h;
    for (int d = 1; d < dim; d++) c.bufferStride[d] = c.bufferStride[d - 1] * n[d];
    szin = bytes_r; szmain = bytes_h;
    if (c2r) { c.inverseReturnToInputBuffer = 1; c.makeInversePlanOnly = 1; } else c.makeForwardPlanOnly = 1;
  } else {
    c.isInputFormatted = 1;
    szin = in_bytes; szmain = out_bytes;
    if (inverse == 1) c.makeInversePlanOnly = 1; else c.makeForwardPlanOnly = 1;
  }
  bin = D.alloc(szin); bmain = D.alloc(szmain);
  c.inputBufferSize = &szin; c.bufferSize = &szmain;
#if VKFFT_BACKEND == 1
  c.inputBuffer = &bin; c.buffer = &bmain;
#else
  c.inputBuffer = &bin; c.buffer = &bmain;
#endif
  // the transform's input: the input buffer, except for c2r (main buffer)
  Dev::Buf src = c2r ? bmain : bin, dst = c2r ? bin : bmain;
  D.put(src, hin.data(), in_bytes);

  // kernel cache (CUDA)
  std::vector<char> blob;
  bool cached = false;
#if VKFFT_BACKEND == 1
  if (cache != "-") {
    FILE* cf = fopen(cache.c_str(), "rb");
    if (cf) {
      fseek(cf, 0, SEEK_END); long sz = ftell(cf); fseek(cf, 0, SEEK_SET);
      blob.resize(sz);
      if (fread(blob.data(), 1, sz, cf) != (size_t)sz) die("reading cache", 0);
      fclose(cf);
      c.loadApplicationFromString = 1; c.loadApplicationString = blob.data(); cached = true;
    } else c.saveApplicationToString = 1;
  }
#endif
  auto t0 = clk::now();
  CHECK(initializeVkFFT(&app, c), "initializeVkFFT");
  double plan_ms = ms_since(t0);
#if VKFFT_BACKEND == 1
  if (c.saveApplicationToString) {
    FILE* cf = fopen(cache.c_str(), "wb");
    if (cf) { fwrite(app.saveApplicationString, 1, app.applicationStringSize, cf); fclose(cf); }
  }
#endif

  VkFFTLaunchParams lp = {};
  lp.inputBuffer = &bin; lp.buffer = &bmain;
  // one transform per command buffer on Metal (as a library call would do), back to back
  auto run = [&]() {
#if VKFFT_BACKEND == 5
    MTL::CommandBuffer* cb = D.queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    lp.commandBuffer = cb; lp.commandEncoder = enc;
    CHECK(VkFFTAppend(&app, inverse, &lp), "VkFFTAppend");
    enc->endEncoding();
    cb->commit();
    D.cb = cb;
#else
    CHECK(VkFFTAppend(&app, inverse, &lp), "VkFFTAppend");
#endif
  };

  t0 = clk::now();
  run(); D.sync();
  double first_ms = ms_since(t0);
  D.get(hout.data(), dst, out_bytes);
  f = fopen(fout, "wb"); fwrite(hout.data(), 1, out_bytes, f); fclose(f);

  std::vector<double> ts;
  for (int s = 0; s < samples; s++) {
    if (c2r) { D.put(src, hin.data(), in_bytes); D.sync(); }
    run(); D.sync();
    t0 = clk::now();
    for (int r = 0; r < reps; r++) run();
    D.sync();
    ts.push_back(ms_since(t0) * 1e3 / reps);
  }
  std::sort(ts.begin(), ts.end());
  double exec_us = ts[ts.size() / 2];
#if VKFFT_BACKEND == 5
  // all <reps> transforms in one command buffer and encoder (VkFFT's own benchmark mode)
  std::vector<double> tb;
  for (int s = 0; s < samples; s++) {
    if (c2r) D.put(src, hin.data(), in_bytes);
    MTL::CommandBuffer* cb = D.queue->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    lp.commandBuffer = cb; lp.commandEncoder = enc;
    for (int r = 0; r < reps; r++) CHECK(VkFFTAppend(&app, inverse, &lp), "VkFFTAppend");
    enc->endEncoding();
    t0 = clk::now();
    cb->commit(); cb->waitUntilCompleted();
    tb.push_back(ms_since(t0) * 1e3 / reps);
  }
  std::sort(tb.begin(), tb.end());
  double batch_us = tb[tb.size() / 2];
#else
  double batch_us = exec_us;
#endif
  printf("%.3f %.3f %.3f %d %.3f\n", plan_ms, first_ms, exec_us, cached ? 1 : 0, batch_us);
  fflush(stdout);
  // The process ends here without tearing down VkFFT and the Metal objects (their release
  // order under the autorelease pool crashes; the OS frees everything at exit).
  std::_Exit(0);
}
