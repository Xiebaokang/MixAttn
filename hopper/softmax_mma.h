#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "utils.h"
#include "softmax_max.cuh"
#include "softmax_add.cuh"
#include "custom_meta.cuh"
#include "custom_numerical_limits.h"

namespace flash {

using namespace cute;

//////////////// MMA CUSTOM
////////////////////////////////////////////////////////////////////////////////////////////////////

#define USE_STATIC_FOR_EACH 1

#if USE_STATIC_FOR_EACH
#define FOR_START(i, r) for_each(make_int_sequence<r>{}, [&](auto i){
#define FOR_END() });
#else
#define FOR_START(i, r) \
    _Pragma("unroll") \
    for (int i = 0; i < r; ++i) {
#define FOR_END() }
#endif

#if USE_STATIC_FOR_EACH
#define IF_CONSTEXPR(cond) if constexpr(cond)
#define ELSE_IF_CONSTEXPR(cond) else if constexpr(cond)
#define ELSE else
#else
#define IF_CONSTEXPR(cond) if (cond)
#define ELSE_IF_CONSTEXPR(cond) else if (cond)
#define ELSE else
#endif



template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ni++) {
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); mi++) {
            summary(mi) = zero_init && ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, bool warp_reduce=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
    if constexpr (warp_reduce) { quad_allreduce_(sum, sum, sum_op); }
}

CUTE_DEVICE float warp_max_reduce_twopass_offset(
    float const& val,
    float const& offset
){
    constexpr uint32_t FULLMASK = 0xFFFFFFFF;
    // max(a+m, b+m) === max(a, b) + m
    // so we first add the offset to the val
    // this can prevent the compiler from demoting the retval_float from uniform reg to simt reg
    float val_offset = val + offset;
    int val_int = __float_as_int(val_offset);
    int max_val_int = __reduce_max_sync(FULLMASK, val_int);
    int min_val_int = __reduce_min_sync(FULLMASK, val_int);
    int retval = (max_val_int < 0 && min_val_int < 0) ? min_val_int : max_val_int;
    float retval_float = __int_as_float(retval);
    return retval_float;
}

template <class FrgBMaskTensor>
CUTE_DEVICE void init_HMMA_scale_frag_B_mask(
  FrgBMaskTensor & frag_B_mask_tensor, // (2,) tensor
  int const& lane_id
){
  static_assert(decltype(cute::size(frag_B_mask_tensor))::value == 2, "frag_B_mask_tensor should have size of 2");
  
  int rem = lane_id % 9;
  int rem_augmented = (lane_id - 4) % 9;

  if (rem == 0) {
      frag_B_mask_tensor(0) = 1.0f;
  }
  if (rem_augmented == 0) {
      frag_B_mask_tensor(1) = 1.0f;
  }
}

// Apply the exp to all the elements.
template <bool Scale_max=true, bool Check_inf=true, int Max_offset=0,
        typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void custom_scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    // For FP8, we can subtract max by 8.0 so that the value after exp2 is in the range of [0, 256].
    // This lets us use more of the FP8 range (instead of just [0, 1]) to reduce underflow.
    static constexpr float max_offset = float(Max_offset);  // We can only template on int, not float
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        const float max_scaled = Check_inf
            ? (max(mi) == MASK_VALUE ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset)
            : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)). This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <class AccSTraits, class AccOTraits, class AccSRowIdxGetter, int kNRows, int Max_offset=0>
struct MmaSoftmax {

    using THIS_CLASS = MmaSoftmax<AccSTraits, AccOTraits, AccSRowIdxGetter, kNRows, Max_offset>;
    
    static constexpr int MMA_2M = AccSTraits::Acc_S_MMA_2M;
    static constexpr int MMA_M = AccSTraits::Acc_S_MMA_M;
    static constexpr int MMA_2N = AccSTraits::Acc_S_MMA_2N;
    static constexpr int MMA_N = AccSTraits::Acc_S_MMA_N;
    static constexpr int warp_size = 32;

    static_assert(kNRows == 2 * MMA_M, "kNRows must be equal to 2 * MMA_M");

    using RowTensorIdxLayout = decltype(make_layout(
        make_shape(Int<MMA_2M>{}, Int<MMA_M>{}),
        GenColMajor{}
      ));

    using RowTensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    RowTensorT row_max, row_sum;

    using MmaMaxTensorT = decltype(make_tensor<float>(Shape<Int<MMA_M>>{}));
    MmaMaxTensorT mma_max_uniform;

    using HMMA1688FragBMaskT = decltype(make_tensor<float>(Shape<Int<2>>{}));
    HMMA1688FragBMaskT fragB_mask;

    AccSRowIdxGetter accS_row_idx_getter;

    float const softmax_scale_log2;
    float const warpmax_offset;
    int const lane_id;
    int const mma_warp_idx;
    int const actual_seqlen_q;

    CUTLASS_DEVICE MmaSoftmax(
        // row idx getter params begin
        int const thread_idx_with_offset,
        int const m_block,
        cutlass::FastDivmod const &qhead_per_khead_divmod,
        // row idx getter params end
        
        float const softmax_scale_log2_,
        int const actual_seqlen_q,
        float const warpmax_offset = 0.f
    )
    : accS_row_idx_getter(thread_idx_with_offset, m_block, qhead_per_khead_divmod)
    , softmax_scale_log2(softmax_scale_log2_)
    , warpmax_offset(warpmax_offset)
    , lane_id(cutlass::canonical_lane_idx())
    , mma_warp_idx(thread_idx_with_offset / warp_size)
    , actual_seqlen_q(actual_seqlen_q) {
        cute::fill(fragB_mask, 0.f);
        init_HMMA_scale_frag_B_mask(
            fragB_mask,
            lane_id
        );
    };

    template <
        bool zero_init=true,
        typename EngineScores, typename LayoutScores,
        typename EngineRowMax, typename LayoutRowMax,
        typename EngineMmaMax, typename LayoutMmaMax
    >
    CUTE_DEVICE void reduce_mma_max_offset_pred(
        Tensor<EngineScores, LayoutScores> const &scores_tensor,
        Tensor<EngineRowMax, LayoutRowMax> &row_max_tensor,
        Tensor<EngineMmaMax, LayoutMmaMax> &mma_max_tensor,
        float const& offset
    ) {
        #if USE_DEFAULT_MAX && !USE_BINARY_TREE_MAX
        MaxOp<float> max_op;
        thread_reduce_<zero_init>(scores_tensor, row_max_tensor, max_op);
        #elif !USE_DEFAULT_MAX && USE_BINARY_TREE_MAX 
        constexpr int reduce_max_fmaxf_ratio = 8;
        reduce_max_binary_max<zero_init, reduce_max_fmaxf_ratio>(scores_tensor, row_max_tensor);
        #else
        #error "Please set exactly one of USE_DEFAULT_MAX or USE_BINARY_TREE_MAX to true"
        #endif

        auto max_tensor_idx_layout = RowTensorIdxLayout{};

        FOR_START(mma_m_idx, MMA_M)
            int row_idx_0 = accS_row_idx_getter.get_row_idx(max_tensor_idx_layout(Int<0>{}, mma_m_idx));
            bool row_idx_0_requires_masking = row_idx_0 >= actual_seqlen_q;
            float row_idx_0_val = row_idx_0_requires_masking ? MASK_VALUE : row_max_tensor(max_tensor_idx_layout(Int<0>{}, mma_m_idx));
            int row_idx_1 = accS_row_idx_getter.get_row_idx(max_tensor_idx_layout(Int<1>{}, mma_m_idx));
            bool row_idx_1_requires_masking = row_idx_1 >= actual_seqlen_q;
            float row_idx_1_val = row_idx_1_requires_masking ? MASK_VALUE : row_max_tensor(max_tensor_idx_layout(Int<1>{}, mma_m_idx));
            float local_max = fmaxf(row_idx_0_val, row_idx_1_val);
            float warp_max = warp_max_reduce_twopass_offset(
                local_max,
                offset
            );
            mma_max_tensor(mma_m_idx) = warp_max;
            row_max_tensor(max_tensor_idx_layout(Int<0>{}, mma_m_idx)) = warp_max;
            row_max_tensor(max_tensor_idx_layout(Int<1>{}, mma_m_idx)) = warp_max;
        FOR_END()
    }
    
    template<class SharedHMMAScoreScaleLayoutMeta, bool Is_first, bool Check_inf=false, typename TensorAccS, typename TensorSharedHMMAScoreScale>
    CUTE_DEVICE RowTensorT max_get_scale_smem_saved(TensorAccS &acc_s, TensorSharedHMMAScoreScale &smem_hmma_score_scale_fragB)
    {   
        // smem_hmma_score_scale has layout (2, NumMmaWarps, Stages) : ColMajor
        static_assert(MMA_M == 1, "MMA_M must be 1");
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        RowTensorT scores_scale;
        if constexpr (Is_first) {
            // flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            reduce_mma_max_offset_pred</*zero_init=*/true>(
                scores, row_max, mma_max_uniform, warpmax_offset
            );
            constexpr float score_scale = 1.f;
            cute::fill(scores_scale, score_scale);
            float hmma_scale_frag0 = fragB_mask(0) * score_scale;
            float hmma_scale_frag1 = fragB_mask(1) * score_scale;
            SharedHMMAScoreScaleLayoutMeta::store(smem_hmma_score_scale_fragB, hmma_scale_frag0, hmma_scale_frag1, mma_warp_idx, lane_id);
        } else {
            // Tensor scores_max_prev = make_fragment_like(row_max);
            // cute::copy(row_max, scores_max_prev);
            Tensor mma_max_uniform_prev = make_fragment_like(mma_max_uniform);
            cute::copy(mma_max_uniform, mma_max_uniform_prev);
            // flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            reduce_mma_max_offset_pred</*zero_init=*/false>(
                scores, row_max, mma_max_uniform, warpmax_offset
            );
            // #pragma unroll
            // for (int mi = 0; mi < size(row_max); ++mi) {
            //     float scores_max_cur = !Check_inf
            //         ? row_max(mi)
            //         : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
            //     scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            //     row_sum(mi) *= scores_scale(mi);
            // }
            auto max_tensor_idx_layout = RowTensorIdxLayout{};
            auto mma_m_idx = Int<0>{};
            float scores_max_cur = !Check_inf
            ? mma_max_uniform(mma_m_idx)
            : (mma_max_uniform(mma_m_idx) == MASK_VALUE ? 0.0f : mma_max_uniform(mma_m_idx));
            float scores_scale_val = exp2f((mma_max_uniform_prev(mma_m_idx) - scores_max_cur) * softmax_scale_log2);
            scores_scale(max_tensor_idx_layout(Int<0>{}, mma_m_idx)) = scores_scale_val;
            scores_scale(max_tensor_idx_layout(Int<1>{}, mma_m_idx)) = scores_scale_val;
            row_sum(max_tensor_idx_layout(Int<0>{}, mma_m_idx)) *= scores_scale_val;
            row_sum(max_tensor_idx_layout(Int<1>{}, mma_m_idx)) *= scores_scale_val;
            float hmma_scale_frag0 = fragB_mask(0) * scores_scale_val;
            float hmma_scale_frag1 = fragB_mask(1) * scores_scale_val;
            SharedHMMAScoreScaleLayoutMeta::store(smem_hmma_score_scale_fragB, hmma_scale_frag0, hmma_scale_frag1, mma_warp_idx, lane_id);
        }
        return scores_scale;
    }

    template<bool Is_first, bool Check_inf=false, typename TensorAccS>
    __forceinline__ __device__ RowTensorT max_get_scale(TensorAccS &acc_s) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        RowTensorT scores_scale;
        if constexpr (Is_first) {
            // flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            reduce_mma_max_offset_pred</*zero_init=*/true>(
                scores, row_max, mma_max_uniform, warpmax_offset
            );
            cute::fill(scores_scale, 1.f);
        } else {
            // Tensor scores_max_prev = make_fragment_like(row_max);
            // cute::copy(row_max, scores_max_prev);
            Tensor mma_max_uniform_prev = make_fragment_like(mma_max_uniform);
            cute::copy(mma_max_uniform, mma_max_uniform_prev);
            // flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            reduce_mma_max_offset_pred</*zero_init=*/false>(
                scores, row_max, mma_max_uniform, warpmax_offset
            );
            // #pragma unroll
            // for (int mi = 0; mi < size(row_max); ++mi) {
            //     float scores_max_cur = !Check_inf
            //         ? row_max(mi)
            //         : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
            //     scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            //     row_sum(mi) *= scores_scale(mi);
            // }
            auto max_tensor_idx_layout = RowTensorIdxLayout{};
            FOR_START(mma_m_idx, MMA_M)
                float scores_max_cur = !Check_inf
                ? mma_max_uniform(mma_m_idx)
                : (mma_max_uniform(mma_m_idx) == MASK_VALUE ? 0.0f : mma_max_uniform(mma_m_idx));
                float scores_scale_val = exp2f((mma_max_uniform_prev(mma_m_idx) - scores_max_cur) * softmax_scale_log2);
                scores_scale(max_tensor_idx_layout(Int<0>{}, mma_m_idx)) = scores_scale_val;
                scores_scale(max_tensor_idx_layout(Int<1>{}, mma_m_idx)) = scores_scale_val;
                row_sum(max_tensor_idx_layout(Int<0>{}, mma_m_idx)) *= scores_scale_val;
                row_sum(max_tensor_idx_layout(Int<1>{}, mma_m_idx)) *= scores_scale_val;
            FOR_END()
        }
        return scores_scale;
    };

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        flash::template custom_scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
        // We don't do the reduce across threads here since we don't need to use the row_sum.
        // We do that reduce at the end when we need to normalize the softmax.
        #if USE_DEFAULT_SUM && !USE_BINARY_TREE_SUM
        flash::reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, row_sum);
        #elif USE_BINARY_TREE_SUM && !USE_DEFAULT_SUM
        flash::custom_reduce_sum_default_simt</*zero_init=*/Is_first>(scores, row_sum);
        #else
        #error "Please set exactly one of USE_DEFAULT_SUM or USE_BINARY_TREE_SUM to 1"
        #endif
    };

    __forceinline__ __device__ RowTensorT finalize(float const final_scale=1.f) {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        RowTensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum * final_scale;
            // For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount.
            if constexpr (Max_offset != 0) {
                static constexpr float sum_scale = 1.f / float(1 << Max_offset);
                sum *= sum_scale;
            }
            row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
        }
        return scores_scale;
    };

    template<typename TensorAccO>
    __forceinline__ __device__ void rescale_o(TensorAccO &acc_o, RowTensorT const &scores_scale) {
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale(mi); }
        }
    };

    template<class SharedHMMAScoreScaleLayoutMeta, typename TensorAccO, typename TensorSharedHMMAScoreScale>
    CUTE_DEVICE void rescale_o_smem_loaded(TensorAccO &acc_o, TensorSharedHMMAScoreScale const& smem_hmma_score_scale) {
        // acc_o has layout ((MMA_2N, MMA_2M, MMA_V), MMA_M, Acc_O_MMA_N)
        [[maybe_unused]] constexpr int Acc_O_MMA_V = AccOTraits::Acc_O_MMA_V;
        [[maybe_unused]] constexpr int Acc_O_MMA_N = AccOTraits::Acc_O_MMA_N;
        static_assert(MMA_M == 1, "MMA_M must be 1");
        // hmma_score_scale_fragB has layout (2,) : ColMajor
        auto hmma_score_scale_fragB = SharedHMMAScoreScaleLayoutMeta::load(smem_hmma_score_scale, mma_warp_idx, lane_id);
        auto mma_m_idx = Int<0>{};
        FOR_START(mma_n_idx, Acc_O_MMA_N)
            FOR_START(mma_v_idx, Acc_O_MMA_V)
                acc_o(make_coord(Int<0>{}, Int<0>{}, mma_v_idx), mma_m_idx, mma_n_idx) *= hmma_score_scale_fragB(Int<0>{});
                acc_o(make_coord(Int<1>{}, Int<0>{}, mma_v_idx), mma_m_idx, mma_n_idx) *= hmma_score_scale_fragB(Int<1>{});
                acc_o(make_coord(Int<0>{}, Int<1>{}, mma_v_idx), mma_m_idx, mma_n_idx) *= hmma_score_scale_fragB(Int<0>{});
                acc_o(make_coord(Int<1>{}, Int<1>{}, mma_v_idx), mma_m_idx, mma_n_idx) *= hmma_score_scale_fragB(Int<1>{});
            FOR_END()
        FOR_END()
    };

};
////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// ORIGINAL SOFTMAX BELOW ///////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void orig_thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ni++) {
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); mi++) {
            summary(mi) = zero_init && ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void orig_quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void orig_reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    orig_thread_reduce_<zero_init>(tensor, summary, op);
    orig_quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void orig_reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    orig_reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, bool warp_reduce=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void orig_reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    orig_thread_reduce_<zero_init>(tensor, sum, sum_op);
    if constexpr (warp_reduce) { orig_quad_allreduce_(sum, sum, sum_op); }
}

// Apply the exp to all the elements.
template <bool Scale_max=true, bool Check_inf=true, int Max_offset=0,
        typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    // For FP8, we can subtract max by 8.0 so that the value after exp2 is in the range of [0, 256].
    // This lets us use more of the FP8 range (instead of just [0, 1]) to reduce underflow.
    static constexpr float max_offset = float(Max_offset);  // We can only template on int, not float
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        const float max_scaled = Check_inf
            ? (max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset)
            : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)). This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows, int Max_offset=0>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;
    float const softmax_scale_log2;

    CUTLASS_DEVICE Softmax(float const softmax_scale_log2_) : softmax_scale_log2(softmax_scale_log2_) {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        TensorT scores_scale;
        if constexpr (Is_first) {
            flash::template orig_reduce_max</*zero_init=*/true>(scores, row_max);
            cute::fill(scores_scale, 1.f);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            flash::template orig_reduce_max</*zero_init=*/false>(scores, row_max);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale(mi);
            }
        }
        return scores_scale;
    };

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
        // We don't do the reduce across threads here since we don't need to use the row_sum.
        // We do that reduce at the end when we need to normalize the softmax.
        flash::orig_reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, row_sum);
    };

    __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
        SumOp<float> sum_op;
        orig_quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum * final_scale;
            // For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount.
            if constexpr (Max_offset != 0) {
                static constexpr float sum_scale = 1.f / float(1 << Max_offset);
                sum *= sum_scale;
            }
            row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
        }
        return scores_scale;
    };

    template<typename Tensor1>
    __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale(mi); }
        }
    };

};

}  // namespace flash
