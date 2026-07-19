#pragma once

#include <cmath>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>



namespace flash {

using namespace cute;

template <class EngineTensor, class LayoutTensor>
CUTE_DEVICE constexpr auto _tensor_add_binary_reduce(
  Tensor<EngineTensor, LayoutTensor> const &tensor
){
  constexpr int tensor_size = cute::size(LayoutTensor{});
  if constexpr (tensor_size == 1) {
    return tensor;
  } else if constexpr (tensor_size % 2 == 0) {
    auto reduced_tensor = cute::make_tensor<float>(
      Shape<Int<tensor_size/2>>{}
    );
    for_each(make_int_sequence<tensor_size/2>{}, [&](auto i){
      reduced_tensor(i) = tensor(i*2) + tensor(i*2+1);
    });
    return _tensor_add_binary_reduce(reduced_tensor);
  } else {
    auto reduced_tensor = cute::make_tensor<float>(
      Shape<Int<(tensor_size-1)/2 + 1>>{}
    );
    for_each(make_int_sequence<(tensor_size-1)/2>{}, [&](auto i){
      reduced_tensor(i) = tensor(i*2) + tensor(i*2+1);
    });
    // simply carry the last element
    reduced_tensor((tensor_size-1)/2) = tensor(tensor_size-1);
    return _tensor_add_binary_reduce(reduced_tensor);
  }
}

template <class EngineTensor, class LayoutTensor>
CUTE_DEVICE constexpr auto _tensor_add_default_reduce(
  Tensor<EngineTensor, LayoutTensor> const &tensor
){
  constexpr int tensor_size = cute::size(LayoutTensor{});
  auto sum_tensor = cute::make_tensor<float>(
    Shape<Int<1>>{}
  );
  CUTE_UNROLL
  for (int i = 0; i < tensor_size; ++i) {
    sum_tensor(0) += tensor(i);
  }
  return sum_tensor;
}

template <bool zero_init, class EngineScores, class LayoutScores,
          class EngineRowSum, class LayoutRowSum>
CUTE_DEVICE void custom_reduce_sum_default_simt(
  Tensor<EngineScores, LayoutScores> const &scores, // ((MMA_2M, MMA_M), (MMA_2N, MMA_N))
  [[maybe_unused]] Tensor<EngineRowSum, LayoutRowSum> &row_sum
){
  constexpr int TensorM = cute::size<0>(LayoutScores{});
  constexpr int TensorN = cute::size<1>(LayoutScores{});
  constexpr int row_sum_M = cute::size<0>(LayoutRowSum{});
  static_assert(row_sum_M == TensorM, "Row sum tensor size should be equal to the input tensor size");

  CUTE_UNROLL
  for (int mi = 0; mi < TensorM; ++mi) {
    auto sum_val = _tensor_add_default_reduce(scores(mi, _))(Int<0>{});
    if constexpr (zero_init) {
      row_sum(mi) = sum_val;
    } else {
      row_sum(mi) += sum_val;
    }
  }
}

template <bool zero_init, class EngineScores, class LayoutScores,
          class EngineRowSum, class LayoutRowSum>
CUTE_DEVICE void custom_reduce_sum_binary_simt(
  Tensor<EngineScores, LayoutScores> const &scores, // ((MMA_2M, MMA_M), (MMA_2N, MMA_N))
  [[maybe_unused]] Tensor<EngineRowSum, LayoutRowSum> &row_sum
){
  constexpr int TensorM = cute::size<0>(LayoutScores{});
  constexpr int TensorN = cute::size<1>(LayoutScores{});
  constexpr int row_sum_M = cute::size<0>(LayoutRowSum{});
  static_assert(row_sum_M == TensorM, "Row sum tensor size should be equal to the input tensor size");

  CUTE_UNROLL
  for (int mi = 0; mi < TensorM; ++mi) {
    auto sum_val = _tensor_add_binary_reduce(scores(mi, _))(Int<0>{});
    if constexpr (zero_init) {
      row_sum(mi) = sum_val;
    } else {
      row_sum(mi) += sum_val;
    }
  }
}


/*
USE_ACCS_DEFAULT_FADD : [scores((MMA_2M, MMA_M), (MMA_2N, MMA_N))] -> row_sum (MMA_2M * MMA_M) -> ordinary normalize lse + ordinary rescal_acc_o logic
USE_ACCS_BINARY_FADD : [scores((MMA_2M, MMA_M), (MMA_2N, MMA_N))] -> row_sum (MMA_2M * MMA_M) -> ordinary normalize lse + ordinary rescal_acc_o logic
USE_ACCS_DEFAULT_MMA_FADD : [acc_s(4, MMA_M, MMA_N)] -> partial_row_sum (MMA_n = 2, MMA_m = 2, MMA_M) -> specialize normalize lse + specialize rescal_acc_o logic
USE_ACCS_BINARY_MMA_FADD : [acc_s(4, MMA_M, MMA_N)] -> partial_row_sum (MMA_n = 2, MMA_m = 2, MMA_M) -> specialize normalize lse + specialize rescal_acc_o logic

1. so we design the fadd reduce family function to use signature (const& scores, const& acc_s, & row_sum, & partial_row_sum, const& fragB)
and mark [[maybe_unused]] for all the unused parameters
2. all the scale_o functions also have to be adapted
*/

enum class AccSReduceMode {
  DEFAULT_SIMT_ONLY,
  BINARY_SIMT_ONLY,
  DEFAULT_MMA_ONLY,
  BINARY_MMA_ONLY
};

template <AccSReduceMode mode>
struct AccSReduce;

template <>
struct AccSReduce<AccSReduceMode::DEFAULT_SIMT_ONLY> {
  template <bool zero_init, int tensor_ratio,
            class EngineScores, class LayoutScores,
            class EngineAccS, class LayoutAccS,
            class EngineRowSum, class LayoutRowSum,
            class EnginePartialRowSum, class LayoutPartialRowSum,
            class EngineFragB, class LayoutFragB>
  static CUTE_DEVICE void reduce_add(
    Tensor<EngineScores, LayoutScores> const &scores, // ((MMA_2M, MMA_M), (MMA_2N, MMA_N))
    [[maybe_unused]] Tensor<EngineAccS, LayoutAccS> const &acc_s, // (4, MMA_M, MMA_N)
    Tensor<EngineRowSum, LayoutRowSum> &row_sum,
    [[maybe_unused]] Tensor<EnginePartialRowSum, LayoutPartialRowSum> &partial_row_sum,
    [[maybe_unused]] Tensor<EngineFragB, LayoutFragB> const& fragB
  ){
    constexpr int TensorM = cute::size<0>(LayoutScores{});
    constexpr int TensorN = cute::size<1>(LayoutScores{});
    constexpr int row_sum_M = cute::size<0>(LayoutRowSum{});
    static_assert(row_sum_M == TensorM, "Row sum tensor size should be equal to the input tensor size");

    CUTE_UNROLL
    for (int mi = 0; mi < TensorM; ++mi) {
      auto sum_val = _tensor_add_default_reduce(scores(mi, _))(Int<0>{});
      if constexpr (zero_init) {
        row_sum(mi) = sum_val;
      } else {
        row_sum(mi) += sum_val;
      }
    }
  }
};

template <>
struct AccSReduce<AccSReduceMode::BINARY_SIMT_ONLY> {
  template <bool zero_init, int tensor_ratio,
            class EngineScores, class LayoutScores,
            class EngineAccS, class LayoutAccS,
            class EngineRowSum, class LayoutRowSum,
            class EnginePartialRowSum, class LayoutPartialRowSum,
            class EngineFragB, class LayoutFragB>
  static CUTE_DEVICE void reduce_add(
    Tensor<EngineScores, LayoutScores> const &scores, // ((MMA_2M, MMA_M), (MMA_2N, MMA_N))
    [[maybe_unused]] Tensor<EngineAccS, LayoutAccS> const &acc_s, // (4, MMA_M, MMA_N)
    Tensor<EngineRowSum, LayoutRowSum> &row_sum,
    [[maybe_unused]] Tensor<EnginePartialRowSum, LayoutPartialRowSum> &partial_row_sum,
    [[maybe_unused]] Tensor<EngineFragB, LayoutFragB> const& fragB
  ){
    constexpr int TensorM = cute::size<0>(LayoutScores{});
    constexpr int TensorN = cute::size<1>(LayoutScores{});
    constexpr int row_sum_M = cute::size<0>(LayoutRowSum{});
    static_assert(row_sum_M == TensorM, "Row sum tensor size should be equal to the input tensor size");

    CUTE_UNROLL
    for (int mi = 0; mi < TensorM; ++mi) {
      auto sum_val = _tensor_add_binary_reduce(scores(mi, _))(Int<0>{});
      if constexpr (zero_init) {
        row_sum(mi) = sum_val;
      } else {
        row_sum(mi) += sum_val;
      }
    }
  }
};

template <>
struct AccSReduce<AccSReduceMode::DEFAULT_MMA_ONLY> {
  template <bool zero_init, int tensor_ratio,
            class EngineScores, class LayoutScores,
            class EngineAccS, class LayoutAccS,
            class EngineRowSum, class LayoutRowSum,
            class EnginePartialRowSum, class LayoutPartialRowSum,
            class EngineFragB, class LayoutFragB>
  static CUTE_DEVICE void reduce_add(
    [[maybe_unused]] Tensor<EngineScores, LayoutScores> const &scores, // ((MMA_2M, MMA_M), (MMA_2N, MMA_N))
    Tensor<EngineAccS, LayoutAccS> const &acc_s, // (4, MMA_M, MMA_N)
    [[maybe_unused]] Tensor<EngineRowSum, LayoutRowSum> &row_sum,
    Tensor<EnginePartialRowSum, LayoutPartialRowSum> &partial_row_sum, // (MMA_n = 2, MMA_m = 2, MMA_M)
    Tensor<EngineFragB, LayoutFragB> const& fragB
  ){
    constexpr int MMA_4 = cute::size<0>(LayoutAccS{});
    constexpr int MMA_M = cute::size<1>(LayoutAccS{});
    constexpr int MMA_N = cute::size<2>(LayoutAccS{});
    static_assert(MMA_4 == 4, "MMA_4 should be 4");
    static_assert(MMA_M == cute::size<2>(LayoutPartialRowSum{}), "MMA_M should be equal to the size of the last dimension of partial_row_sum");
    
    using MmaArchAtom = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using MmaTraits = cute::MMA_Traits<MmaArchAtom>;
    if constexpr (zero_init) {
      cute::fill(partial_row_sum, 0.0f);
    }
    for_each(make_int_sequence<MMA_N>{}, [&](auto mma_n_idx){
      for_each(make_int_sequence<MMA_M>{}, [&](auto mma_m_idx){
        mma_unpack(
          MmaTraits{},
          partial_row_sum(_,_,mma_m_idx),
          acc_s(_,mma_m_idx,mma_n_idx),
          fragB,
          partial_row_sum(_,_,mma_m_idx)
        );
      });
    });
  }
};

template <>
struct AccSReduce<AccSReduceMode::BINARY_MMA_ONLY> {

  template <class EngineAccSSlice, class LayoutAccSSlice,
            class EngineFragB, class LayoutFragB>
  static CUTE_DEVICE constexpr auto _binary_tensor_reduce_add(
    Tensor<EngineAccSSlice, LayoutAccSSlice> const &acc_s_slice, // (4, MMA_N)
    Tensor<EngineFragB, LayoutFragB> const& fragB
  ){
    constexpr int MMA_N = cute::size<1>(LayoutAccSSlice{});
    using MmaArchAtom = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using MmaTraits = cute::MMA_Traits<MmaArchAtom>;

    if constexpr (MMA_N == 1) {
      return acc_s_slice;
    } else if constexpr (MMA_N % 2 == 0) {
      // partial row sum: (MMA_n = 2, MMA_m = 2, MMA_M)
      auto reduced_tensor = cute::make_tensor<float>(
        cute::make_layout(
          cute::make_shape(cute::make_shape(Int<2>{}, Int<2>{}), Int<MMA_N/2>{}),
          GenColMajor{}
        )
      );
      cute::fill(reduced_tensor, 0.0f);
      for_each(make_int_sequence<MMA_N/2>{}, [&](auto i){
        mma_unpack(
          MmaTraits{},
          reduced_tensor(_,i),
          acc_s_slice(_, 2*i),
          fragB,
          reduced_tensor(_,i)
        );
        mma_unpack(
          MmaTraits{},
          reduced_tensor(_,i),
          acc_s_slice(_, 2*i+1),
          fragB,
          reduced_tensor(_,i)
        );
      });
      return _binary_tensor_reduce_add(reduced_tensor, fragB);
    } else {
      static_assert(cute::dependent_false<EngineAccSSlice>, "MMA_N should be even");
    }
  }

  template <bool zero_init, int tensor_ratio,
            class EngineScores, class LayoutScores,
            class EngineAccS, class LayoutAccS,
            class EngineRowSum, class LayoutRowSum,
            class EnginePartialRowSum, class LayoutPartialRowSum,
            class EngineFragB, class LayoutFragB>
  static CUTE_DEVICE void reduce_add(
    [[maybe_unused]] Tensor<EngineScores, LayoutScores> const &scores, // ((MMA_2M, MMA_M), (MMA_2N, MMA_N))
    Tensor<EngineAccS, LayoutAccS> const &acc_s, // (4, MMA_M, MMA_N)
    [[maybe_unused]] Tensor<EngineRowSum, LayoutRowSum> &row_sum,
    Tensor<EnginePartialRowSum, LayoutPartialRowSum> &partial_row_sum, // (MMA_n = 2, MMA_m = 2, MMA_M)
    Tensor<EngineFragB, LayoutFragB> const& fragB
  ){
    constexpr int MMA_4 = cute::size<0>(LayoutAccS{});
    constexpr int MMA_M = cute::size<1>(LayoutAccS{});
    constexpr int MMA_N = cute::size<2>(LayoutAccS{});
    static_assert(MMA_4 == 4, "MMA_4 should be 4");
    static_assert(MMA_M == cute::size<2>(LayoutPartialRowSum{}), "MMA_M should be equal to the size of the last dimension of partial_row_sum");
    
    using MmaArchAtom = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using MmaTraits = cute::MMA_Traits<MmaArchAtom>;

    
    static_assert(MMA_N % 2 == 0, "MMA_N must be even");

    constexpr int offset = MMA_N / 2;

    if constexpr (zero_init) {
      cute::fill(partial_row_sum, 0.0f);
    }

    for_each(make_int_sequence<MMA_M>{}, [&](auto mma_m_idx){
      auto partial_row_sum_temp = cute::make_tensor<float>(
        cute::make_layout(
          cute::make_shape(Int<2>{}, Int<2>{}),
          GenColMajor{}
        )
      );
      cute::fill(partial_row_sum_temp, 0.0f);
      for_each(make_int_sequence<MMA_N/2>{}, [&](auto mma_n_idx){
        mma_unpack(
          MmaTraits{},
          partial_row_sum(_,_,mma_m_idx),
          acc_s(_,mma_m_idx,mma_n_idx + Int<0>{}),
          fragB,
          partial_row_sum(_,_,mma_m_idx)
        );
        mma_unpack(
          MmaTraits{},
          partial_row_sum_temp(_,_),
          acc_s(_,mma_m_idx,mma_n_idx + Int<offset>{}),
          fragB,
          partial_row_sum_temp(_,_)
        );
      });

      partial_row_sum(Int<0>{},Int<0>{},mma_m_idx) += partial_row_sum_temp(Int<0>{}, Int<0>{});
      partial_row_sum(Int<0>{},Int<1>{},mma_m_idx) += partial_row_sum_temp(Int<0>{}, Int<1>{});
      partial_row_sum(Int<1>{},Int<0>{},mma_m_idx) += partial_row_sum_temp(Int<1>{}, Int<0>{});
      partial_row_sum(Int<1>{},Int<1>{},mma_m_idx) += partial_row_sum_temp(Int<1>{}, Int<1>{});

      
    });
  }
};

} // end namespace FLASH_NAMESPACE