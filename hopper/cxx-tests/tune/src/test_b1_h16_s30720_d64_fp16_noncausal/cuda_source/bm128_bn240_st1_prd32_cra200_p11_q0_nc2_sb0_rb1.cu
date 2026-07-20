#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

#include <cute/tensor.hpp>

// Keep this example focused on the ordinary FA3 forward path.
#define FLASHATTENTION_DISABLE_SPLIT
#define FLASHATTENTION_DISABLE_PAGEDKV
#define FLASHATTENTION_DISABLE_SOFTCAP
#define FLASHATTENTION_DISABLE_APPENDKV
#define FLASHATTENTION_DISABLE_PACKGQA
#define FLASHATTENTION_DISABLE_SM8x
#define FLASHATTENTION_DISABLE_LOCAL

#define ENABLE_CUSTOM_FWD_LAUNCH_TEMPLATE_REPORT 0
#define USE_MMA_SOFTMAX 0
#define USE_MIX_WGMMA 1

#include "custom_api.cuh"

namespace {

struct TConfig {
  int kBlockM = 128;
  int kBlockN = 240;
  int kStage = 1;
  uint32_t producer_reg_dealloc = 32;
  uint32_t consumer_reg_alloc = 200;
  int p_smem_k_tiles = 11;
  int q_reg_k_tiles = 0;
  int num_consumer = 2;
  int use_scheduler_barrier = 0;
  int rescale_o_before_gemm = 1;
};

constexpr int kBatch = 1;
constexpr int kSeqlen = 30720;
constexpr int kNumHeads = 16;
constexpr int kHeadDim = 64;

constexpr TConfig cfg{};

struct RunOptions {
  bool verify = false;
  uint64_t seed = 1234;
  double atol = -1.0;
  double rtol = -1.0;
};

void print_usage(const char* program) {
  std::fprintf(stderr,
               "usage: %s [--verify] [--seed=N] [--atol=X] [--rtol=X]\n",
               program);
}

bool parse_options(int argc, char** argv, RunOptions& options) {
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--verify") == 0) {
      options.verify = true;
    } else if (std::strncmp(arg, "--seed=", 7) == 0) {
      char* end = nullptr;
      options.seed = std::strtoull(arg + 7, &end, 10);
      if (end == arg + 7 || *end != '\0') return false;
    } else if (std::strncmp(arg, "--atol=", 7) == 0) {
      char* end = nullptr;
      options.atol = std::strtod(arg + 7, &end);
      if (end == arg + 7 || *end != '\0' || options.atol < 0.0) return false;
    } else if (std::strncmp(arg, "--rtol=", 7) == 0) {
      char* end = nullptr;
      options.rtol = std::strtod(arg + 7, &end);
      if (end == arg + 7 || *end != '\0' || options.rtol < 0.0) return false;
    } else {
      return false;
    }
  }
  return true;
}

template <typename Element, at::ScalarType InputType, bool IsCausal>
void benchmark(const char* dtype_name, const RunOptions& options) {
  const auto fp32 = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  const auto shape = std::initializer_list<int64_t>{kBatch, kSeqlen, kNumHeads, kHeadDim};

  // FA3 expects [batch, sequence, heads, head_dim]. FP8 output is BF16.
  at::manual_seed(options.seed);
  auto q_fp32 = at::rand(shape, fp32) - 0.5f;
  auto k_fp32 = at::rand(shape, fp32) - 0.5f;
  auto v_fp32 = at::rand(shape, fp32) * 2.0f - 1.0f;
  auto q = q_fp32.to(InputType);
  auto k = k_fp32.to(InputType);
  auto v = v_fp32.to(InputType);
  const auto output_type = InputType == at::kFloat8_e4m3fn ? at::kBFloat16 : at::kHalf;
  auto out = at::empty(shape, fp32.dtype(output_type));
  std::optional<at::Tensor> out_opt = out;
  const float softmax_scale = 1.0f / std::sqrt(float(kHeadDim));

  auto run = [&] {
    if constexpr (IsCausal) {
      custom_mha_fwd_causal<cfg, Element, kHeadDim, kHeadDim>(
          q, k, v, out_opt, softmax_scale);
    } else {
      custom_mha_fwd_noncausal<cfg, Element, kHeadDim, kHeadDim>(
          q, k, v, out_opt, softmax_scale);
    }
  };

  if (options.verify) {
    run();

    // ATen SDPA expects [batch, heads, sequence, head_dim].  FP8 SDPA is not
    // universally available, so use the exact quantized FP8 values converted
    // to FP16 as its independent reference input.
    constexpr bool IsFP8 = InputType == at::kFloat8_e4m3fn;
    constexpr at::ScalarType ReferenceType = IsFP8 ? at::kHalf : InputType;
    auto q_ref = q.to(ReferenceType).permute({0, 2, 1, 3});
    auto k_ref = k.to(ReferenceType).permute({0, 2, 1, 3});
    auto v_ref = v.to(ReferenceType).permute({0, 2, 1, 3});
    auto reference = at::scaled_dot_product_attention(
        q_ref, k_ref, v_ref, std::nullopt, 0.0, IsCausal,
        static_cast<double>(softmax_scale), false)
                         .permute({0, 2, 1, 3})
                         .contiguous()
                         .to(at::kFloat);
    auto actual = out.to(at::kFloat);

    const double atol = options.atol >= 0.0 ? options.atol : (IsFP8 ? 0.1 : 0.02);
    const double rtol = options.rtol >= 0.0 ? options.rtol : (IsFP8 ? 0.1 : 0.02);
    auto finite = at::isfinite(actual) & at::isfinite(reference);
    auto abs_diff = (actual - reference).abs();
    auto allowed = atol + rtol * reference.abs();
    auto mismatch = (~finite) | (abs_diff > allowed);
    const int64_t mismatch_count = mismatch.sum().item<int64_t>();
    const int64_t finite_count = finite.sum().item<int64_t>();
    auto reported_diff = at::where(
        finite, abs_diff, at::full_like(abs_diff, INFINITY));
    const int64_t max_index = reported_diff.reshape({-1}).argmax().item<int64_t>();
    const double max_abs = reported_diff.reshape({-1})[max_index].item<double>();
    const double mean_abs = finite_count == 0
                                ? INFINITY
                                : at::where(finite, abs_diff, at::zeros_like(abs_diff))
                                          .sum()
                                          .item<double>() /
                                      finite_count;

    int64_t index = max_index;
    const int64_t d = index % kHeadDim;
    index /= kHeadDim;
    const int64_t h = index % kNumHeads;
    index /= kNumHeads;
    const int64_t s = index % kSeqlen;
    const int64_t b = index / kSeqlen;
    const double actual_at_max = actual.reshape({-1})[max_index].item<double>();
    const double reference_at_max = reference.reshape({-1})[max_index].item<double>();

    std::printf(
        "VERIFY %s %s seed=%llu: mismatches=%lld/%lld, "
        "max_abs=%.8g, mean_abs=%.8g, atol=%.4g, rtol=%.4g\n",
        dtype_name, IsCausal ? "causal" : "non-causal",
        static_cast<unsigned long long>(options.seed),
        static_cast<long long>(mismatch_count),
        static_cast<long long>(actual.numel()), max_abs, mean_abs, atol, rtol);
    std::printf(
        "  max error at [b=%lld, s=%lld, h=%lld, d=%lld]: "
        "actual=%.8g, reference=%.8g\n",
        static_cast<long long>(b), static_cast<long long>(s),
        static_cast<long long>(h), static_cast<long long>(d), actual_at_max,
        reference_at_max);
    if (mismatch_count != 0) {
      std::fprintf(stderr, "VERIFY FAILED\n");
      std::exit(EXIT_FAILURE);
    }
    std::printf("VERIFY PASSED\n");
    return;
  }

  for (int i = 0; i < 50; ++i) run();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, at::cuda::getCurrentCUDAStream());
  for (int i = 0; i < 250; ++i) run();
  cudaEventRecord(stop, at::cuda::getCurrentCUDAStream());
  cudaEventSynchronize(stop);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  const double time_ms = total_ms / 250;
  // QK^T and P*V each cost 2*B*H*S*S*D FLOPs. Causal computes half.
  double flops = 4.0 * kBatch * kNumHeads * kSeqlen * kSeqlen * kHeadDim;
  if constexpr (IsCausal) flops *= 0.5;
  const double tflops = flops / (time_ms * 1.0e9);

  std::printf("%-4s  %-10s  time = %8.3f ms,  throughput = %8.2f TFLOPS\n",
              dtype_name, IsCausal ? "causal" : "non-causal", time_ms,
              tflops);
}

}  // namespace

int main(int argc, char** argv) {
  RunOptions options;
  if (!parse_options(argc, argv, options)) {
    print_usage(argv[0]);
    return 2;
  }
  std::printf("FA3 forward: B=%d, S=%d, H=%d, D=%d\n",
              kBatch, kSeqlen, kNumHeads, kHeadDim);

  benchmark<cutlass::half_t, at::kHalf, false>("FP16", options);

  return 0;
}
