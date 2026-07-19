#include <ATen/ATen.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cuda_profiler_api.h>

#include <cute/tensor.hpp>

#include "utils.cuh"

// simply disable everything that is irrelevant to our focus
#define FLASHATTENTION_DISABLE_SPLIT
#define FLASHATTENTION_DISABLE_PAGEDKV
#define FLASHATTENTION_DISABLE_SOFTCAP
#define FLASHATTENTION_DISABLE_APPENDKV
#define FLASHATTENTION_DISABLE_PACKGQA
#define FLASHATTENTION_DISABLE_SM8x
#define FLASHATTENTION_DISABLE_LOCAL


// custom macros
#define ENABLE_CUSTOM_FWD_LAUNCH_TEMPLATE_REPORT      0
#define USE_MMA_SOFTMAX                               1 // bool
#define USE_REUSE_KV                                  0 // bool

#include "custom_api.cuh"

using namespace cute;

#ifndef FLASHATTENTION_BENCH_FP16
#define FLASHATTENTION_BENCH_FP16 0
#endif

#if FLASHATTENTION_BENCH_FP16
static constexpr at::ScalarType kTargetTorchType = at::kHalf;
using target_type = cutlass::half_t;
static constexpr const char* kBenchDataType = "FP16-FP32";
#else
static constexpr at::ScalarType kTargetTorchType = at::kFloat8_e4m3fn;
using target_type = cute::float_e4m3_t;
static constexpr const char* kBenchDataType = "FP8-FP32";
#endif

bool has_arg(int argc, char* argv[], std::string const& name) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == name) {
      return true;
    }
  }
  return false;
}

std::string get_string_arg(int argc, char* argv[], std::string const& name, std::string default_value = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return argv[i + 1];
    }
  }
  return default_value;
}

int get_int_arg(int argc, char* argv[], std::string const& name, int default_value) {
  std::string value = get_string_arg(argc, argv, name, "");
  return value.empty() ? default_value : std::stoi(value);
}

template <bool IsCausal, typename CausalFn, typename NonCausalFn>
void run_selected(CausalFn &&f_run_causal, NonCausalFn &&f_run_noncausal) {
  if constexpr (IsCausal) {
    f_run_causal();
  } else {
    f_run_noncausal();
  }
}

template <int HeadDim, bool IsCausal>
auto bench_fwd(
  int batch_size,
  int num_heads,
  int seqlen,
  int iter,
  int warmup
){
  auto q_tensor = at::rand({batch_size, seqlen, num_heads, HeadDim}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  auto k_tensor = at::rand({batch_size, seqlen, num_heads, HeadDim}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  auto v_tensor = at::rand({batch_size, seqlen, num_heads, HeadDim}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  auto out_tensor = at::zeros({batch_size, seqlen, num_heads, HeadDim}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));

  auto [q_tensor_typed, k_tensor_typed, v_tensor_typed, out_tensor_typed] = convert_tensor_dtype(
    q_tensor, k_tensor, v_tensor, out_tensor, kTargetTorchType
  );

  [[maybe_unused]] c10::optional<bool> pack_gqa_opt = c10::nullopt;
  c10::optional<at::Tensor> out_opt = out_tensor_typed;

  float softmax_scale = 1.0 / sqrtf(HeadDim);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto f_run_causal = [&](){
    custom_mha_fwd_causal<target_type, HeadDim, HeadDim>(
      q_tensor_typed, k_tensor_typed, v_tensor_typed, out_opt, softmax_scale
    );
  };
  auto f_run_noncausal = [&](){
    custom_mha_fwd_noncausal<target_type, HeadDim, HeadDim>(
      q_tensor_typed, k_tensor_typed, v_tensor_typed, out_opt, softmax_scale
    );
  };
  
  for (int i = 0; i < warmup; ++i) {
    run_selected<IsCausal>(f_run_causal, f_run_noncausal);
  }
  
  cudaEventRecord(start);
  for (int iter_idx = 0; iter_idx < iter; ++iter_idx) {
    run_selected<IsCausal>(f_run_causal, f_run_noncausal);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  
  float total_elapsed_time_ms;
  cudaEventElapsedTime(&total_elapsed_time_ms, start, stop);
  float avg_time_ms = total_elapsed_time_ms / iter;

  return avg_time_ms;

}

template <int HeadDim>
float bench_fwd_single(
  int batch_size,
  int num_heads,
  int seqlen,
  bool is_causal,
  int iter,
  int warmup
){
  if (is_causal) {
    return bench_fwd<HeadDim, true>(batch_size, num_heads, seqlen, iter, warmup);
  }
  return bench_fwd<HeadDim, false>(batch_size, num_heads, seqlen, iter, warmup);
}

float bench_fwd_single_dispatch(
  int batch_size,
  int num_heads,
  int seqlen,
  int head_dim,
  bool is_causal,
  int iter,
  int warmup
){
  if (head_dim == 64) {
    return bench_fwd_single<64>(batch_size, num_heads, seqlen, is_causal, iter, warmup);
  }
  if (head_dim == 128) {
    return bench_fwd_single<128>(batch_size, num_heads, seqlen, is_causal, iter, warmup);
  }
  throw std::runtime_error("single-shape mode supports head_dim 64 or 128");
}

void bench_fwd_single_to_csv(int argc, char* argv[]) {
  std::string csv_filename = get_string_arg(argc, argv, "--csv", "");
  int batch_size = get_int_arg(argc, argv, "--batch-size", 1);
  int num_heads = get_int_arg(argc, argv, "--num-heads", 32);
  int seqlen = get_int_arg(argc, argv, "--seq-len", 1024);
  int head_dim = get_int_arg(argc, argv, "--head-dim", 64);
  bool is_causal = get_int_arg(argc, argv, "--causal", 0) != 0;
  int iter = get_int_arg(argc, argv, "--iter", 1000);
  int warmup = get_int_arg(argc, argv, "--warmup", 200);

  float avg_time_ms = bench_fwd_single_dispatch(
    batch_size, num_heads, seqlen, head_dim, is_causal, iter, warmup
  );

  printf("batch_size: %-10d, seqlen: %-10d, num_heads: %-10d, head_dim: %-10d, causal: %-10d, avg_time_ms: %.6f\n",
    batch_size, seqlen, num_heads, head_dim, is_causal, avg_time_ms);

  if (!csv_filename.empty()) {
    std::vector<std::string> csv_header = {
      "DataType",
      "Comment",
      "batchsize",
      "nheads",
      "seqlen",
      "headdim",
      "is_causal",
      "time_ms"
    };
    std::vector<std::vector<std::string>> csv_data = {{
      std::string(kBenchDataType),
      std::string("fa-t"),
      std::to_string(batch_size),
      std::to_string(num_heads),
      std::to_string(seqlen),
      std::to_string(head_dim),
      std::to_string(is_causal),
      std::to_string(avg_time_ms)
    }};

    add_write_result_to_csv(csv_filename, csv_header, csv_data);
  }
}

void bench_fwd_all(std::string const& csv_filename = ""){
  constexpr int total_tokens = 16384;
  constexpr auto seqlens = cute::make_tuple(Int<128>{}, Int<256>{}, Int<512>{}, Int<1024>{}, Int<2048>{}, Int<4096>{}, Int<8192>{}, Int<16384>{});
  constexpr auto batch_sizes = cute::transform(seqlens, [&](auto seqlen){
    constexpr int seqlen_v = CUTE_STATIC_V(seqlen);
    return Int<total_tokens / seqlen_v>{};
  });
  constexpr int hid_dim = 2048;

  // constexpr auto head_dims = cute::make_tuple(Int<32>{}, Int<64>{}, Int<128>{});
  // skip head dim 32
  constexpr auto head_dims = cute::make_tuple(Int<64>{}, Int<128>{});
  constexpr auto num_heads = cute::transform(head_dims, [&](auto head_dim){
    constexpr int head_dim_v = CUTE_STATIC_V(head_dim);
    return Int<hid_dim / head_dim_v>{};
  });

  constexpr auto causal_list = cute::make_tuple(cute::false_type{}, cute::true_type{});

  constexpr auto causal_list_len = cute::rank(causal_list);
  constexpr int seqlens_len = cute::rank(seqlens);
  constexpr int head_dims_len = cute::rank(head_dims);

  std::vector<std::string> csv_header = {
    "DataType",
    "Comment",
    "batchsize",
    "nheads",
    "seqlen",
    "headdim",
    "is_causal",
    "time_ms"
  };
  std::vector<std::vector<std::string>> csv_data;
  
  for_each(make_int_sequence<causal_list_len>{}, [&](auto causal_idx){
    auto is_causal = cute::get<causal_idx>(causal_list);
    constexpr bool is_causal_v = CUTE_STATIC_V(is_causal);
    for_each(make_int_sequence<head_dims_len>{}, [&](auto num_head_head_dim_idx){
      auto head_dim = cute::get<num_head_head_dim_idx>(head_dims);
      auto num_head = cute::get<num_head_head_dim_idx>(num_heads);
      constexpr int head_dim_v = CUTE_STATIC_V(head_dim);
      constexpr int num_head_v = CUTE_STATIC_V(num_head);
      for_each(make_int_sequence<seqlens_len>{}, [&](auto batchsize_seqlen_idx){
        auto seqlen = cute::get<batchsize_seqlen_idx>(seqlens);
        auto batch_size = cute::get<batchsize_seqlen_idx>(batch_sizes);
        constexpr int seqlen_v = CUTE_STATIC_V(seqlen);
        constexpr int batch_size_v = CUTE_STATIC_V(batch_size);
        constexpr int iter = 1000;
        constexpr int warmup = 200;
        float avg_time_ms = bench_fwd<head_dim_v, is_causal_v>(
          batch_size_v,
          num_head_v,
          seqlen_v,
          iter,
          warmup
        );
        csv_data.push_back({
          std::string(kBenchDataType),
          std::string("fa-t"),
          std::to_string(batch_size_v),
          std::to_string(num_head_v),
          std::to_string(seqlen_v),
          std::to_string(head_dim_v),
          std::to_string(is_causal_v),
          std::to_string(avg_time_ms)
        });
        printf("batch_size: %-10d, seqlen: %-10d, num_heads: %-10d, head_dim: %-10d, causal: %-10d, avg_time_ms: %.6f\n",
          batch_size_v, seqlen_v, num_head_v, head_dim_v, is_causal_v, avg_time_ms);
      });
    });
  });
  
  if (!csv_filename.empty()) {
    write_result_to_csv(csv_filename, csv_header, csv_data);
  }
  
}

int main(int argc, char* argv[]){
  if (has_arg(argc, argv, "--single")) {
    bench_fwd_single_to_csv(argc, argv);
    return 0;
  }
  std::string csv_filename = parse_filename_arg(argc, argv);
  bench_fwd_all(csv_filename);
}
