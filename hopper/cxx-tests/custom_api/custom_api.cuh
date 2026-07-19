#pragma once

// custom macro control

#if USE_MMA_SOFTMAX

// custom config

#define USE_DEFAULT_MAX                               0
#define USE_BINARY_TREE_MAX                           1

#define USE_DEFAULT_SUM                               1
#define USE_BINARY_TREE_SUM                           0

#define USE_INVERSE_WGSYNC                            1

#define ENABLE_CUSTOM_SM90_SCHED_BARRIER_OVERRIDE     1 // bool
#define CUSTOM_OVERRIDE_SCHED_BARRIER                 1 // bool

#else

// default FA3 config

#define USE_DEFAULT_MAX                               1
#define USE_BINARY_TREE_MAX                           0

#define USE_DEFAULT_SUM                               1
#define USE_BINARY_TREE_SUM                           0

#define USE_INVERSE_WGSYNC                            0

#define ENABLE_CUSTOM_SM90_SCHED_BARRIER_OVERRIDE     0 // bool
#define CUSTOM_OVERRIDE_SCHED_BARRIER                 0 // bool

#endif

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn/functional.h>

#include "flash.h"
#include "static_switch.h"
#include "tile_size.h"
#include "heuristics.h"
#include "cuda_check.h"

#include "flash_fwd_launch_template.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                                    \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}),                  \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


void set_params_fprop(Flash_fwd_params &params,
  // sizes
  const size_t b,
  const size_t seqlen_q,
  const size_t seqlen_k,
  const size_t seqlen_q_rounded,
  const size_t seqlen_k_rounded,
  const size_t h,
  const size_t h_k,
  const size_t d,
  const size_t d_rounded,
  // device pointers
  const at::Tensor q,
  const at::Tensor k,
  const at::Tensor v,
  at::Tensor out,
  void *cu_seqlens_q_d,
  void *cu_seqlens_k_d,
  void *seqused_q,
  void *seqused_k,
  void *softmax_lse_d,
  float p_dropout,
  float softmax_scale,
  int window_size_left,
  int window_size_right,
  const float softcap=0.f,
  const int sm_margin=0) {

  // Reset the parameters
  params = {};

  params.is_bf16 = q.dtype() == torch::kBFloat16;
  params.is_e4m3 = q.dtype() == torch::kFloat8_e4m3fn;

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  // All stride are in elements, not bytes.
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.v_dim_stride = v.stride(-1);
  params.o_ptr = out.data_ptr();
  params.o_row_stride = out.stride(-3);
  params.o_head_stride = out.stride(-2);

  if (cu_seqlens_q_d == nullptr) {
  params.q_batch_stride = q.stride(0);
  params.o_batch_stride = out.stride(0);
  }
  if (cu_seqlens_k_d == nullptr) {
  params.k_batch_stride = k.stride(0);
  params.v_batch_stride = v.stride(0);
  }

  params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
  params.seqused_q = static_cast<int *>(seqused_q);
  params.seqused_k = static_cast<int *>(seqused_k);

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.d_rounded = d_rounded;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.softcap = softcap;

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to float to compare.
  // [Minor] We want to round down since when we do the comparison we use <= instead of <
  // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
  // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  TORCH_CHECK(p_dropout < 1.f);
  #ifdef FLASHATTENTION_DISABLE_DROPOUT
  TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
  #endif

  // Causal is the special case where window_size_right == 0 and window_size_left < 0.
  // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = window_size_left < 0 && window_size_right == 0;
  params.is_local = (window_size_left >= 0 || window_size_right >= 0) && !params.is_causal;

  // TODO: check this
  if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k - 1; }
  if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_q - 1; }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

  params.arch = at::cuda::getCurrentDeviceProperties()->major * 10 + at::cuda::getCurrentDeviceProperties()->minor;
  params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin;

  #ifdef FLASHATTENTION_DISABLE_LOCAL
  TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");
  #endif
}

inline int round_up_headdim(int head_size) {
  #ifndef FLASHATTENTION_DISABLE_HDIM64
  if (head_size <= 64) { return 64; }
  #endif
  #ifndef FLASHATTENTION_DISABLE_HDIM96
  if (head_size <= 96) { return 96; }
  #endif
  #ifndef FLASHATTENTION_DISABLE_HDIM128
  if (head_size <= 128) { return 128; }
  #endif
  #ifndef FLASHATTENTION_DISABLE_HDIM192
  if (head_size <= 192) { return 192; }
  #endif
  #ifndef FLASHATTENTION_DISABLE_HDIM256
  if (head_size <= 256) { return 256; }
  #endif
  return 256;
}

inline int round_up_headdimv(int head_size) {
  if (head_size <= 64) { return 64; }
  if (head_size <= 96) { return 96; }
  if (head_size <= 128) { return 128; }
  if (head_size <= 192) { return 192; }
  if (head_size <= 256) { return 256; }
  return 512;
}

inline bool get_pagedkv_tma(Flash_fwd_params const& params) {
  if (params.arch < 90 || !params.page_table || params.leftpad_k || params.knew_ptr) { return false; }
  // This needs to match the kernel configs
  auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, false /*paged_kv_non_TMA*/, params.softcap > 0.f);
  int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
  int const kBlockN = std::get<1>(kBlockMN_kernel_args_sm90);
  // Heuristic: when seqlen_q <= kBlockM, we're not compute bound, and somehow using TMA is slower,
  // at least for MLA.
  return params.page_size % kBlockN == 0 && params.seqlen_q * (params.h / params.h_k) > kBlockM;
}

inline bool get_pack_gqa(Flash_fwd_params const& params) {
  // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation and binary size.
  // Has little effect on speed.
  if (params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1) { return true; }
  #ifdef FLASHATTENTION_DISABLE_PACKGQA
  return false;
  #else
  // params.page_table must already be set
  if (params.h == params.h_k) { return false; }
  // This needs to match the kernel configs
  auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
  int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
  return should_pack_gqa(params.cu_seqlens_q || params.seqused_q, params.seqlen_q, params.h / params.h_k, kBlockM);
  #endif
}

inline int get_num_splits(Flash_fwd_params const& params) {
  #ifdef FLASHATTENTION_DISABLE_SPLIT
  return 1;
  #else
  // Always enable PackGQA for Split
  // params.page_table must already be set
  // This needs to match the kernel configs
  bool varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k;
  auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
  // Strictly speaking we need to pass in (varlen && params.num_splits > 1) but num_splits
  // has not been set here. It's OK though because we might just underestimate kBlockN a bit
  auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, varlen, params.softcap > 0.f, params.knew_ptr);
  int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
  int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
  int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
  // If is_local, we're not going to load all of seqlen_k
  int const seqlen_k_loaded = !params.is_local
      ? params.seqlen_k
      : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + kBlockM));
  int const num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
  int const num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) / kBlockM;
  int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * (params.is_e4m3 ? 1 : 2);
  // Always enable PackGQA for Split
  // If varlen, we use dynamic split, so this heuristic just needs to get an upper bound on num_splits.
  // We assume the case where there's 1 long sequence and the rest are short, i.e. pretending
  // that batch = 1.
  int total_mblocks = (params.num_splits_dynamic_ptr ? 1 : params.b) * params.h_k * num_m_blocks;
  return num_splits_heuristic(total_mblocks, params.num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, params.is_causal || params.is_local, 128);
  #endif
}

#if USE_MIX_WGMMA
template <auto Config, typename Dtype, int HeadDim, int HeadDimV, bool IsCausal>
#else
template <typename Dtype, int HeadDim, int HeadDimV, bool IsCausal>
#endif
std::vector<at::Tensor> custom_mha_fwd_template_core(
    at::Tensor &q, // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const at::Tensor &k, // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
            // or (num_pages, page_size, h_k, d) if there is page_table.
    const at::Tensor & v, // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k
           // or (num_pages, page_size, h_k, dv) if there is page_table.
    std::optional<at::Tensor> &out_, // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    float const softmax_scale,
    int window_size_left,
    int window_size_right,
    float const softcap,
    int num_splits,
    std::optional<bool> pack_gqa_,
    int const sm_margin
  ) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm8x = dprops->major >= 8;
  TORCH_CHECK(is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
  bool is_sm90 = dprops->major >= 9;
  TORCH_CHECK(is_sm90, "Custom FlashAttention only supports Hopper GPUs or newer.");

  auto q_type = q.scalar_type();
  TORCH_CHECK(
      q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16 ||
          q_type == at::ScalarType::Float8_e4m3fn,
      "FlashAttention only supports fp16, bf16, and fp8_e4m3 data type");
  if (dprops->major < 9) {
    TORCH_CHECK(q_type == at::ScalarType::Half ||
                    q_type == at::ScalarType::BFloat16,
                "FlashAttention on Ampere/Ada cards only supports fp16 and "
                "bf16 data type");
  }
  TORCH_CHECK(k.scalar_type() == q_type,
              "query and key must have the same dtype");
  TORCH_CHECK(v.scalar_type() == q_type,
              "query and value must have the same dtype");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);

  TORCH_CHECK(q.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1,
              "Input tensor must have contiguous last dimension");

  bool is_causal = IsCausal;

  // useless tensors
  at::Tensor page_table;
  constexpr bool paged_KV = false;
  at::Tensor cu_seqlens_q;
  constexpr bool is_varlen_q = false;
  at::Tensor cu_seqlens_k;
  constexpr bool is_varlen_k = false;

  if (paged_KV) {
    // page_table = page_table_.value();
    // CHECK_DEVICE(page_table);
    // TORCH_CHECK(page_table.dtype() == torch::kInt32, "page_table must have dtype torch.int32");
    // TORCH_CHECK(page_table.stride(-1) == 1, "page_table must have contiguous last dimension");
  }

  if (is_varlen_k) {
    // cu_seqlens_k = cu_seqlens_k_.value();
    // CHECK_DEVICE(cu_seqlens_k); CHECK_CONTIGUOUS(cu_seqlens_k);
    // TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype torch.int32");
    // TORCH_CHECK(max_seqlen_k_.has_value(), "max_seqlen_k must be provided if cu_seqlens_k is provided");
    // TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, then page table is not supported");
    // TORCH_CHECK(!kv_batch_idx_.has_value(), "If cu_seqlens_k is passed in, then page table is not supported");
  }


  auto const sizes = q.sizes();
  const int batch_size = !is_varlen_q ? sizes[0] : cu_seqlens_q.size(0) - 1;
  int seqlen_q = sizes[1];
  int total_q = batch_size * sizes[1];
  int num_heads = q.size(-2);
  int const head_size = q.size(-1);
  int const head_size_v = v.size(-1);
  int const max_num_pages_per_seq = !paged_KV ? 0 : page_table.size(1);
  int const num_pages = !paged_KV ? 0 : k.size(0);
  int const page_size = !paged_KV ? 1 : k.size(1);
  int const seqlen_k = k.size(1);
  int const total_k = !is_varlen_k ? batch_size * k.size(1) : k.size(0);
  int const num_heads_k = k.size(-2);
  int const batch_size_k =
      !paged_KV ? (!is_varlen_k ? k.size(0) : cu_seqlens_k.size(0) - 1)
                : page_table.size(0);

  if (head_size_v != head_size) {
    TORCH_CHECK((head_size > 128 && head_size <= 192 && head_size_v > 96 &&
                 head_size_v <= 128) ||
                    (head_size <= 64 && head_size_v <= 512),
                "If V headdim is different from Q/K dim, we only support Q/K "
                "headdim in (128, 192] and V headdim in (96, 128], "
                "or (Q/K <= 64 and V <= 512).");
    TORCH_CHECK(dprops->major == 9, "Only Hopper supports different V headdim");
    if (head_size_v > 256) {
      TORCH_CHECK(q_type == at::ScalarType::Half ||
                      q_type == at::ScalarType::BFloat16,
                  "HeaddimV > 256 requires fp16 and bf16 data type");
    }
  }

  // This needs to go before kBlockM & kBlockN since we rely on the correct
  // window_size and is_causal to set kBlockM
  // TODO: check this
  if (window_size_left >= seqlen_k - 1) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_q - 1) {
    window_size_right = -1;
  }
  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1 && window_size_left == -1 && window_size_right == -1) {
    // Special case of hdim 128 where we want causal to have kBlockN=128, better
    // for pagedKV and TMA
    if ((head_size <= 64 || head_size > 128) || !paged_KV) {
      is_causal = false;
    }
  }
  if (is_causal) {
    window_size_right = 0;
  }
  // There's a case where is_causal=false, window_size=(-1, 0). Then
  // set_params_fprop will set params.is_causal=true. If we don't have is_causal
  // here matching params.is_causal, we might get the wrong kBlockM.
  is_causal = window_size_left < 0 && window_size_right == 0;

  if (!is_varlen_q) {
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  } else {
    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  }
  if (!paged_KV) {
    if (!is_varlen_k) {
      CHECK_SHAPE(k, batch_size_k, seqlen_k, num_heads_k, head_size);
      CHECK_SHAPE(v, batch_size_k, seqlen_k, num_heads_k, head_size_v);
    } else {
      CHECK_SHAPE(k, total_k, num_heads_k, head_size);
      CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
      CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    }
  } else {
    CHECK_SHAPE(k, num_pages, page_size, num_heads_k, head_size);
    CHECK_SHAPE(v, num_pages, page_size, num_heads_k, head_size_v);
    CHECK_SHAPE(page_table, batch_size_k, max_num_pages_per_seq);
  }

  if (/*seqused_q_.has_value()*/false){
    // auto seqused_q = seqused_q_.value();
    // TORCH_CHECK(seqused_q.dtype() == torch::kInt32, "seqused_q must have dtype int32");
    // CHECK_DEVICE(seqused_q); CHECK_CONTIGUOUS(seqused_q);
    // CHECK_SHAPE(seqused_q, batch_size);
  }
  if (/*seqused_k_.has_value()*/false) {
    // auto seqused_k = seqused_k_.value();
    // TORCH_CHECK(seqused_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
    // CHECK_DEVICE(seqused_k); CHECK_CONTIGUOUS(seqused_k);
    // CHECK_SHAPE(seqused_k, batch_size);
  }

  if (/*leftpad_k_.has_value()*/false) {
    // auto leftpad_k = leftpad_k_.value();
    // TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
    // CHECK_DEVICE(leftpad_k); CHECK_CONTIGUOUS(leftpad_k);
    // CHECK_SHAPE(leftpad_k, batch_size);
  }

  // This is what we will template on
  bool const is_varlen = is_varlen_q || is_varlen_k ;
  #ifdef FLASHATTENTION_DISABLE_VARLEN
      TORCH_CHECK(!is_varlen, "This flash attention build does not support varlen.");
  #endif

  int const alignment = q_type == torch::kFloat8_e4m3fn ? 16 : 8;
  TORCH_CHECK(head_size % alignment == 0, "head_size should be a multiple of " + std::to_string(alignment));
  TORCH_CHECK(head_size_v % alignment == 0, "head_size_v should be a multiple of " + std::to_string(alignment));

  auto opts = q.options();
  auto out_type = q_type == at::ScalarType::Float8_e4m3fn ? at::ScalarType::BFloat16 : q_type;
  at::Tensor out;
  if (out_.has_value()) {
      out = out_.value();
      TORCH_CHECK(out.scalar_type() == out_type, "For FP16/BF16 input, output must have the same dtype as inputs. For FP8 input, output must have dtype BF16");
      CHECK_DEVICE(out);
      TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
      if (!is_varlen_q) {
          CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_v);
      } else {
          CHECK_SHAPE(out, total_q, num_heads, head_size_v);
      }
  } else {
      out = !is_varlen_q
          ? torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts.dtype(out_type))
          : torch::empty({total_q, num_heads, head_size_v}, opts.dtype(out_type));
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  int const head_size_rounded = round_up_headdim(head_size);
  int const head_size_v_rounded = head_size_v == head_size ? head_size_rounded : round_up_headdimv(head_size_v);
  
  // TORCH_CHECK(head_size_rounded == head_size, "head_size_rounded should be equal to head_size");
  // TORCH_CHECK(head_size_v_rounded == head_size_v, "head_size_v_rounded should be equal to head_size_v");
  // TORCH_CHECK(head_size_rounded == HeadDim, "head_size_rounded should be equal to HeadDim");
  // TORCH_CHECK(head_size_v_rounded == HeadDimV, "head_size_v_rounded should be equal to HeadDimV");


  int const seqlen_q_rounded = round_multiple(seqlen_q, 128);
  int const seqlen_k_rounded = round_multiple(seqlen_k, 128);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  at::Tensor softmax_lse;
  if (!is_varlen_q) {
      softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
  } else {
      softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
  }

  Flash_fwd_params params;
  set_params_fprop(params,
                    batch_size,
                    seqlen_q, seqlen_k,
                    seqlen_q_rounded, seqlen_k_rounded,
                    num_heads, num_heads_k,
                    head_size, head_size_rounded,
                    q, k, v, out,
                    !is_varlen_q ? nullptr : cu_seqlens_q.data_ptr(),
                    !is_varlen_k ? nullptr : cu_seqlens_k.data_ptr(),
                    nullptr,
                    nullptr,
                    softmax_lse.data_ptr(),
                    /*p_dropout=*/0.f,
                    softmax_scale,
                    window_size_left,
                    window_size_right,
                    softcap,
                    sm_margin);
  params.total_q = total_q;
  params.total_k = total_k;
  params.b_k = batch_size_k;
  params.dv = head_size_v;
  params.dv_rounded = head_size_v_rounded;
  if (/*leftpad_k_.has_value()*/false) {  // This needs to be set before get_pagedkv_tma
    // params.leftpad_k = static_cast<int *>(leftpad_k_.value().data_ptr());
  }
  if (paged_KV) {
      params.page_table = page_table.data_ptr<int>();
      params.page_table_batch_stride = page_table.stride(0);
  }
  params.page_size = page_size;
  params.num_pages = num_pages;

  if (/*k_new_.has_value()*/ false) {  // This needs to be set before get_pagedkv_tma
    // at::Tensor k_new, v_new;
    // TORCH_CHECK(v_new_.has_value(), "If k_new is supplied, v_new must also be passed in");
    // TORCH_CHECK(seqused_k_.has_value(), "If k_new is supplied, seqlens_k must also be passed in");
    // TORCH_CHECK(seqlen_q <= seqlen_k, "If k_new is supplied, it must have seqlen <= the seqlen of the KV cache");
    // at::Tensor cu_seqlens_k_new;
    // bool const is_varlen_k_new = cu_seqlens_k_new_.has_value();
    // if (is_varlen_k_new) {
    //     cu_seqlens_k_new = cu_seqlens_k_new_.value();
    //     CHECK_DEVICE(cu_seqlens_k_new); CHECK_CONTIGUOUS(cu_seqlens_k_new);
    //     TORCH_CHECK(cu_seqlens_k_new.dtype() == torch::kInt32, "cu_seqlens_k_new must have dtype torch.int32");
    // }
    // k_new = k_new_.value();
    // v_new = v_new_.value();
    // TORCH_CHECK(k_new.dtype() == q_type, "k_new must have the same dtype as query");
    // TORCH_CHECK(v_new.dtype() == q_type, "v_new must have the same dtype as query");
    // CHECK_DEVICE(k_new); CHECK_DEVICE(v_new);
    // TORCH_CHECK(k_new.stride(-1) == 1, "k_new tensor must have contiguous last dimension");
    // TORCH_CHECK(v_new.stride(-1) == 1, "v_new tensor must have contiguous last dimension");
    // // We don't need max_seqlen_k_new, so seqlen_k_new can be whatever when is_varlen_k_new
    // int seqlen_k_new = !is_varlen_k_new ? k_new.size(1) : 0;
    // int total_k_new = !is_varlen_k_new ? batch_size * k_new.size(1): k_new.size(0);
    // if (!is_varlen_k_new) {
    //     CHECK_SHAPE(k_new, batch_size, seqlen_k_new, num_heads_k, head_size);
    //     CHECK_SHAPE(v_new, batch_size, seqlen_k_new, num_heads_k, head_size_v);
    // } else {
    //     CHECK_SHAPE(k_new, total_k_new, num_heads_k, head_size);
    //     CHECK_SHAPE(v_new, total_k_new, num_heads_k, head_size_v);
    //     CHECK_SHAPE(cu_seqlens_k_new, batch_size + 1);
    // }
    // params.seqlen_knew = seqlen_k_new;
    // params.total_knew = total_k_new;
    // params.knew_ptr = k_new.data_ptr();
    // params.vnew_ptr = v_new.data_ptr();
    // // All stride are in elements, not bytes.
    // params.knew_row_stride = k_new.stride(-3);
    // params.vnew_row_stride = v_new.stride(-3);
    // params.knew_head_stride = k_new.stride(-2);
    // params.vnew_head_stride = v_new.stride(-2);
    // if (!is_varlen_k_new) {
    //     params.knew_batch_stride = k_new.stride(0);
    //     params.vnew_batch_stride = v_new.stride(0);
    // }
    // if (is_varlen_k_new) {
    //     params.cu_seqlens_knew = static_cast<int*>(cu_seqlens_k_new.data_ptr());
    // }
  }


  // 992 = 32 * 31 is the max supported batch in prepare_varlen_num_blocks kernel
  bool const use_dynamic_split = is_varlen && params.b <= 992;
  // Temporarily set num_splits_dynamic_ptr to 1 since get_num_splits checks it
  params.num_splits_dynamic_ptr = !use_dynamic_split ? nullptr : reinterpret_cast<int*>(1);

  params.pagedkv_tma = get_pagedkv_tma(params);
  params.num_splits = num_splits <= 0 ? get_num_splits(params) : num_splits;
  // Always enable PackGQA for Split, and get_pack_gqa requires params.num_splits to decide
  params.pack_gqa = pack_gqa_.has_value() ? pack_gqa_.value() : get_pack_gqa(params);

  // This needs to be set after get_num_splits
  at::Tensor tile_count_semaphore;  // Contains the semaphore and optionally num_splits_dynamic
  // We don't use the persistent scheduler if Split and not Varlen
  bool const scheduler_needs_semaphore = params.arch >= 90
      ? (((params.is_causal || params.is_local) && (params.num_splits == 1)) || is_varlen)
      : ((params.is_causal && !is_varlen) || (is_varlen && params.num_splits > 1));
  if (scheduler_needs_semaphore || use_dynamic_split) {
      int metadata_size = int(scheduler_needs_semaphore) + int(use_dynamic_split) * params.b;
      params.skip_scheduler_metadata_computation = false;
      if ( /*scheduler_metadata_.has_value()*/ false) {
          // at::Tensor scheduler_metadata = scheduler_metadata_.value();
          // CHECK_DEVICE(scheduler_metadata);
          // CHECK_SHAPE(scheduler_metadata, metadata_size);
          // CHECK_CONTIGUOUS(scheduler_metadata);
          // TORCH_CHECK(scheduler_metadata.dtype() == torch::kInt32, "scheduler_metadata must have dtype int32");
          // tile_count_semaphore = scheduler_metadata;
      } else {
          tile_count_semaphore = torch::empty({metadata_size}, opts.dtype(torch::kInt32));
      }
      if (scheduler_needs_semaphore && !use_dynamic_split) {
          tile_count_semaphore.zero_();  // If varlen we'll manually do the zero-ing
      }
      params.tile_count_semaphore = scheduler_needs_semaphore ? tile_count_semaphore.data_ptr<int>() : nullptr;
      params.num_splits_dynamic_ptr = use_dynamic_split ? tile_count_semaphore.data_ptr<int>() + 1 : nullptr;
  }

  if (/*q_v_.has_value()*/false) {
    // TORCH_CHECK(head_size <= 64, "q_v is only supported for head_size <= 64");
    // TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
    //             "q_v is only supported for fp16 and bf16 data type");
    // TORCH_CHECK(params.arch == 90, "q_v is only supported for Hopper GPUs");
    // at::Tensor q_v = q_v_.value();
    // TORCH_CHECK(q_v.dtype() == q_type, "q_v must have the same dtype as query");
    // CHECK_DEVICE(q_v);
    // TORCH_CHECK(q_v.stride(-1) == 1, "q_v tensor must have contiguous last dimension");
    // if (!is_varlen_q) {
    //     CHECK_SHAPE(q_v, batch_size, seqlen_q, num_heads, head_size_v);
    // } else {
    //     CHECK_SHAPE(q_v, total_q, num_heads, head_size_v);
    // }
    // params.qv_ptr = q_v.data_ptr();
    // // All stride are in elements, not bytes.
    // params.qv_row_stride = q_v.stride(-3);
    // params.qv_head_stride = q_v.stride(-2);
    // if (!is_varlen_q) {
    //     params.qv_batch_stride = q_v.stride(0);
    // }
  }

  if (/*rotary_cos_.has_value()*/ false) {
      // TORCH_CHECK(k_new_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
      // auto rotary_cos = rotary_cos_.value();
      // CHECK_DEVICE(rotary_cos); CHECK_CONTIGUOUS(rotary_cos);
      // params.rotary_dim = rotary_cos.size(1) * 2;
      // TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
      // TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
      // const int seqlen_ro = rotary_cos.size(0);
      // if (paged_KV) {
      //     TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
      // }
      // CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
      // TORCH_CHECK(rotary_cos.scalar_type() == q_type, "rotary_cos must have the same dtype as query");

      // TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
      // auto rotary_sin = rotary_sin_.value();
      // CHECK_DEVICE(rotary_sin); CHECK_CONTIGUOUS(rotary_sin);
      // CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
      // TORCH_CHECK(rotary_sin.scalar_type() == q_type, "rotary_cos must have the same dtype as query");
      // params.rotary_cos_ptr = rotary_cos.data_ptr();
      // params.rotary_sin_ptr = rotary_sin.data_ptr();
      // params.is_rotary_interleaved = is_rotary_interleaved;
      // if (seqlens_rotary_.has_value()) {
      //     at::Tensor seqlens_rotary = seqlens_rotary_.value();
      //     CHECK_DEVICE(seqlens_rotary); CHECK_CONTIGUOUS(seqlens_rotary);
      //     TORCH_CHECK(seqlens_rotary.dtype() == torch::kInt32, "seqlens_rotary must have dtype torch.int32");
      //     CHECK_SHAPE(seqlens_rotary, batch_size);
      //     params.seqlens_rotary = seqlens_rotary.data_ptr<int>();
      // }
  } else {
      params.rotary_dim = 0;
  }

  if (/*kv_batch_idx_.has_value()*/false) {
    // auto kv_batch_idx = kv_batch_idx_.value();
    // CHECK_DEVICE(kv_batch_idx); CHECK_CONTIGUOUS(kv_batch_idx);
    // TORCH_CHECK(kv_batch_idx.scalar_type() == torch::kInt32, "kv_batch_idx must have dtype int32");
    // params.kv_batch_idx = reinterpret_cast<int *>(kv_batch_idx.data_ptr());
  }

  at::Tensor out_accum, softmax_lse_accum;
  auto outaccum_type = at::ScalarType::Float;
  if (params.num_splits > 1) {
    TORCH_CHECK(params.num_splits <= 256, "num_splits > 256 not supported");
    if (!is_varlen_q) {
      out_accum = torch::empty(
          {params.num_splits, batch_size, num_heads, seqlen_q, head_size_v},
          opts.dtype(outaccum_type));
      softmax_lse_accum =
          torch::empty({params.num_splits, batch_size, num_heads, seqlen_q},
                       opts.dtype(at::kFloat));
      params.oaccum_batch_stride = out_accum.stride(1);
      params.lseaccum_batch_stride = softmax_lse_accum.stride(1);
    } else {
      out_accum =
          torch::empty({params.num_splits, num_heads, total_q, head_size_v},
                       opts.dtype(outaccum_type));
      softmax_lse_accum = torch::empty({params.num_splits, num_heads, total_q},
                                       opts.dtype(at::kFloat));
    }
    params.is_fp32 = false;
    params.oaccum_ptr = out_accum.data_ptr();
    params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
    params.oaccum_split_stride = out_accum.stride(0);
    params.oaccum_row_stride = out_accum.stride(-2);
    params.oaccum_head_stride = out_accum.stride(-3);
    params.lseaccum_split_stride = softmax_lse_accum.stride(0);
    params.lseaccum_head_stride = softmax_lse_accum.stride(-2);
  }


  /*
    CUSTOM 注：
    理论上来说需要在fp8情况下给这些descale设置上合适的数值
    但是Kernel mainloop里用的是：

    float const q_descale = params.ptr_q_descale == nullptr ? 1.0f : params.ptr_q_descale[bidb * get<0>(params.stride_q_descale) + bidh_kv * get<1>(params.stride_q_descale)];
    float const k_descale = params.ptr_k_descale == nullptr ? 1.0f : params.ptr_k_descale[bidb * get<0>(params.stride_k_descale) + bidh_kv * get<1>(params.stride_k_descale)];

    所以无所谓了，我们比较时只要保持baseline和我们的实现都不传descale就行了

  */
  if (q_type == at::ScalarType::Float8_e4m3fn) {
    if (/*q_descale_.has_value()*/false) {
      // auto q_descale = q_descale_.value();
      // CHECK_DEVICE(q_descale);
      // CHECK_SHAPE(q_descale, batch_size, num_heads_k);
      // params.q_descale_ptr = q_descale.data_ptr<float>();
      // params.q_descale_batch_stride = q_descale.stride(0);
      // params.q_descale_head_stride = q_descale.stride(1);
    } else {
      params.q_descale_ptr = nullptr;
    }
    if (/*k_descale_.has_value()*/false) {
      // auto k_descale = k_descale_.value();
      // CHECK_DEVICE(k_descale);
      // CHECK_SHAPE(k_descale, batch_size, num_heads_k);
      // params.k_descale_ptr = k_descale.data_ptr<float>();
      // params.k_descale_batch_stride = k_descale.stride(0);
      // params.k_descale_head_stride = k_descale.stride(1);
    } else {
      params.k_descale_ptr = nullptr;
    }
    if (/*v_descale_.has_value()*/false) {
      // auto v_descale = v_descale_.value();
      // CHECK_DEVICE(v_descale);
      // CHECK_SHAPE(v_descale, batch_size, num_heads_k);
      // params.v_descale_ptr = v_descale.data_ptr<float>();
      // params.v_descale_batch_stride = v_descale.stride(0);
      // params.v_descale_head_stride = v_descale.stride(1);
    } else {
      params.v_descale_ptr = nullptr;
    }
  }

#if !USE_MIX_WGMMA
  #ifdef FLASHATTENTION_DISABLE_LOCAL
  TORCH_CHECK(!params.is_local, "This flash attention build does not support local attention.");
  #endif
  #ifdef FLASHATTENTION_DISABLE_SOFTCAP
  TORCH_CHECK(params.softcap == 0.0, "This flash attention build does not support tanh softcapping.");
  #endif
  #ifdef FLASHATTENTION_DISABLE_SPLIT
  TORCH_CHECK(params.num_splits == 1, "This flash attention build does not support splits.");
  #endif
  #ifdef FLASHATTENTION_DISABLE_PACKGQA
  TORCH_CHECK(!params.pack_gqa || params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1, "This flash attention build does not support pack_gqa.");
  #endif
  #ifdef FLASHATTENTION_DISABLE_PAGEDKV
  TORCH_CHECK(!(params.page_table && !params.pagedkv_tma), "This flash attention build does not support paged KV.");
  #endif
  #ifdef FLASHATTENTION_DISABLE_APPENDKV
  // TORCH_CHECK(!k_new_.has_value(), "This flash attention build does not support appending KV.");
  #endif
#else
  TORCH_CHECK(params.num_splits == 1, "USE_MIX_WGMMA does not support split attention");
  TORCH_CHECK(!params.page_table, "USE_MIX_WGMMA does not support paged KV");
  TORCH_CHECK(params.softcap == 0.0, "USE_MIX_WGMMA does not support softcap");
  TORCH_CHECK(!params.pack_gqa, "USE_MIX_WGMMA does not support PackGQA");
  TORCH_CHECK(!params.is_local, "USE_MIX_WGMMA does not support local attention");
  TORCH_CHECK(
    !params.cu_seqlens_q && !params.cu_seqlens_k &&
    !params.seqused_q && !params.seqused_k && !params.leftpad_k,
    "USE_MIX_WGMMA only supports fixed-length attention"
  );
  TORCH_CHECK(!params.knew_ptr, "USE_MIX_WGMMA does not support appended KV");
  TORCH_CHECK(!params.qv_ptr, "USE_MIX_WGMMA does not support QV");
#endif

  if (total_q > 0 && (total_k + params.total_knew) > 0 && num_heads_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
#if USE_MIX_WGMMA
    static_assert(HeadDim == HeadDimV,
                  "USE_MIX_WGMMA custom API requires HeadDim == HeadDimV");
#endif
    // NOTE: fa3 api would round BOTH HeadDim and HeadDimV to 64
    if constexpr (HeadDimV <= 64 || HeadDim <= 64) {
#if USE_MIX_WGMMA
      run_mha_fwd_custom_fixed_<90, Config, Dtype, /*HeadDim=*/64, /*kHeadDimV=*/64, IsCausal>(
        params, stream
      );
#else
      run_mha_fwd_custom_<90, Dtype, /*HeadDim=*/64, /*kHeadDimV=*/64, false, false, false, false>(
        params, stream
      );
#endif
    } else {
#if USE_MIX_WGMMA
      run_mha_fwd_custom_fixed_<90, Config, Dtype, HeadDim, HeadDimV, IsCausal>(
        params, stream
      );
#else
      run_mha_fwd_custom_<90, Dtype, HeadDim, HeadDimV, false, false, false, false>(
        params, stream
      );
#endif
    }
    
    if (scheduler_needs_semaphore && params.skip_scheduler_metadata_computation) {
      // need to zero out the semaphore in this case
      tile_count_semaphore.index({torch::indexing::Slice(0, 1)}).zero_();
    }
  } else if (total_q > 0 && num_heads_k > 0) {
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
  }
  // return {out, softmax_lse};
  return {out, softmax_lse, out_accum, softmax_lse_accum};
}

#if USE_MIX_WGMMA
template <auto Config, class Dtype, int HeadDim, int HeadDimV>
#else
template <class Dtype, int HeadDim, int HeadDimV>
#endif
std::vector<at::Tensor> custom_mha_fwd_noncausal(
  at::Tensor &q, // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
  const at::Tensor &k, // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
          // or (num_pages, page_size, h_k, d) if there is page_table.
  const at::Tensor & v, // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k
          // or (num_pages, page_size, h_k, dv) if there is page_table.
  std::optional<at::Tensor> &out_, // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
  float const softmax_scale,
  int num_splits = 1,
  int window_size_left = -1,
  int window_size_right = -1,
  float const softcap = 0.0,
  std::optional<bool> pack_gqa_ = std::nullopt,
  int const sm_margin = 0
){
#if USE_MIX_WGMMA
  return custom_mha_fwd_template_core<Config, Dtype, HeadDim, HeadDimV, false>(
    q,
    k,
    v,
    out_,
    softmax_scale,
    window_size_left,
    window_size_right,
    softcap,
    num_splits,
    pack_gqa_,
    sm_margin);
#else
  return custom_mha_fwd_template_core<Dtype, HeadDim, HeadDimV, false>(
    q,
    k,
    v,
    out_,
    softmax_scale,
    window_size_left,
    window_size_right,
    softcap,
    num_splits,
    pack_gqa_,
    sm_margin);
#endif

}
#if USE_MIX_WGMMA
template <auto Config, class Dtype, int HeadDim, int HeadDimV>
#else
template <class Dtype, int HeadDim, int HeadDimV>
#endif
std::vector<at::Tensor> custom_mha_fwd_causal(
  at::Tensor &q, // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
  const at::Tensor &k, // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
          // or (num_pages, page_size, h_k, d) if there is page_table.
  const at::Tensor & v, // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k
          // or (num_pages, page_size, h_k, dv) if there is page_table.
  std::optional<at::Tensor> &out_, // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
  float const softmax_scale,
  int num_splits = 1,
  int window_size_left = -1,
  int window_size_right = -1,
  float const softcap = 0.0,
  std::optional<bool> pack_gqa_ = std::nullopt,
  int const sm_margin = 0
){
#if USE_MIX_WGMMA
  return custom_mha_fwd_template_core<Config, Dtype, HeadDim, HeadDimV, true>(
    q,
    k,
    v,
    out_,
    softmax_scale,
    window_size_left,
    window_size_right,
    softcap,
    num_splits,
    pack_gqa_,
    sm_margin);
#else
  return custom_mha_fwd_template_core<Dtype, HeadDim, HeadDimV, true>(
    q,
    k,
    v,
    out_,
    softmax_scale,
    window_size_left,
    window_size_right,
    softcap,
    num_splits,
    pack_gqa_,
    sm_margin);
#endif
}

// helper function for converting torch tensor types

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> convert_tensor_dtype(
  at::Tensor &q,
  at::Tensor &k,
  at::Tensor &v,
  at::Tensor &out,
  at::ScalarType fa_precision
){
  bool is_allowed_precision = fa_precision == at::kFloat8_e4m3fn || 
                              fa_precision == at::kBFloat16 || 
                              fa_precision == at::kHalf;
  TORCH_CHECK(is_allowed_precision, "fa_precision must be one of float8_e4m3fn, bfloat16, or float16");
  

  if (fa_precision == at::kFloat8_e4m3fn) {
    auto q_tensor_typed = q.to(at::kFloat8_e4m3fn);
    auto k_tensor_typed = k.to(at::kFloat8_e4m3fn);
    auto v_tensor_typed = v.to(at::kFloat8_e4m3fn);
    auto out_tensor_typed = out.to(at::kBFloat16);
    return {q_tensor_typed, k_tensor_typed, v_tensor_typed, out_tensor_typed};
  } else if (fa_precision == at::kBFloat16) {
    auto q_tensor_typed = q.to(at::kBFloat16);
    auto k_tensor_typed = k.to(at::kBFloat16);
    auto v_tensor_typed = v.to(at::kBFloat16);
    auto out_tensor_typed = out.to(at::kBFloat16);
    return {q_tensor_typed, k_tensor_typed, v_tensor_typed, out_tensor_typed};
  } else if (fa_precision == at::kHalf) {
    auto q_tensor_typed = q.to(at::kHalf);
    auto k_tensor_typed = k.to(at::kHalf);
    auto v_tensor_typed = v.to(at::kHalf);
    auto out_tensor_typed = out.to(at::kHalf);
    return {q_tensor_typed, k_tensor_typed, v_tensor_typed, out_tensor_typed};
  } else {
    TORCH_CHECK(false, "Unsupported fa_precision");
  }
  __builtin_unreachable();
}
