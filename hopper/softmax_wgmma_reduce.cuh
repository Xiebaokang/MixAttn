#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "utils.h"
#include "softmax.h"
#include "softmax_max.cuh"
#include "softmax_add.cuh"
#include "custom_meta.cuh"
#include "custom_numerical_limits.h"

namespace flash {

using namespace cute;



////////////////////////////////////////////////////////////////////////////////////////////////////

template <class AccSTraits, int kNRows, int Max_offset=0>
struct WGMMAReduceSoftmax {

    static constexpr int MMA_M = AccSTraits::Acc_S_MMA_M;
    static constexpr int MMA_N = AccSTraits::Acc_S_MMA_N;

    static_assert(kNRows == 2 * MMA_M, "kNRows must be equal to 2 * MMA_M");
    static_assert(MMA_M == 1, "MMA_M must be equal to 1 for WGMMAReduceSoftmax");

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max;
    float const softmax_scale_log2;

    using MmaRowSumTensorT = decltype(make_tensor<float>(Shape<Int<2>, Int<2>>{}));
    MmaRowSumTensorT mma_row_sum;

    TensorT simt_row_sum;
    // in our wgmma reduce softmax this row max is only
    // valid AFTER finalize being called

    CUTLASS_DEVICE WGMMAReduceSoftmax(float const softmax_scale_log2_) : softmax_scale_log2(softmax_scale_log2_) {
        clear(mma_row_sum);
        clear(simt_row_sum);
    };

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    __forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        TensorT scores_scale;
        if constexpr (Is_first) {
            reduce_max_binary_max<true, -1>(scores, row_max);
            cute::fill(scores_scale, 1.f);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            reduce_max_binary_max<false, -1>(scores, row_max);

            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            }
            simt_row_sum(0) *= scores_scale(0);
            simt_row_sum(1) *= scores_scale(1);
            mma_row_sum(0) *= scores_scale(0);
            mma_row_sum(2) *= scores_scale(0);
            mma_row_sum(1) *= scores_scale(1);
            mma_row_sum(3) *= scores_scale(1);
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
        flash::reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, simt_row_sum);
    };

    template<bool Check_inf=false, typename Tensor0>
    CUTE_DEVICE void online_softmax_rescale(Tensor0 &acc_s) {
        // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
    }

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    CUTE_DEVICE void online_softmax_reduce_simt(Tensor0 &acc_s) {
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        flash::reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, simt_row_sum);
    };

    template<typename Tensor0, typename SmemFrgB>
    CUTE_DEVICE void online_softmax_reduce_wgmma(Tensor0 &acc_s, SmemFrgB const& smem_frag_b) {

        constexpr int MMA_V = AccSTraits::Acc_S_MMA_V;
        using MmaTraits = FlashFwdWGMMAReduceMeta::MmaTraits;
        constexpr auto mma_m_idx = Int<0>{};
        warpgroup_fence_operand(acc_s);
        warpgroup_arrive();
        for (int mma_n_v_idx = 0; mma_n_v_idx < MMA_V * MMA_N; ++mma_n_v_idx) {
            int mma_v_idx = mma_n_v_idx % MMA_V;
            int mma_n_idx = mma_n_v_idx / MMA_V;
            mma_unpack(
                MmaTraits{},
                mma_row_sum,
                acc_s(make_coord(_,_,mma_v_idx), mma_m_idx, mma_n_idx),
                smem_frag_b,
                mma_row_sum
            );
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    };

    CUTE_DEVICE auto get_row_sum() {
        TensorT row_sum;
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, simt_row_sum, sum_op);
        row_sum(0) += (mma_row_sum(0) + mma_row_sum(2));
        row_sum(1) += (mma_row_sum(1) + mma_row_sum(3));
        return row_sum;
    }

    __forceinline__ __device__ TensorT finalize(float const final_scale=1.f) {
        auto row_sum = get_row_sum();
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum * final_scale;
        }
        return scores_scale;
    };

    CUTE_DEVICE TensorT get_final_row_sum(){
        auto row_sum = get_row_sum();
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float sum = row_sum(mi);
            if constexpr (Max_offset != 0) {
                // For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount.
                constexpr float sum_scale = 1.f / float(1 << Max_offset);
                row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum * sum_scale);
            } else {
                row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
            }
        }
        return row_sum;
    }

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

} // namespace flash
