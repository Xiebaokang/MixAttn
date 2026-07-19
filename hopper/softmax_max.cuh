#pragma once

#include <cmath>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>



namespace flash {

using namespace cute;

template <class EngineTensor, class LayoutTensor>
CUTE_DEVICE constexpr auto _tensor_max_binary_reduce(
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
      reduced_tensor(i) = fmaxf(tensor(i*2), tensor(i*2+1));
    });
    return _tensor_max_binary_reduce(reduced_tensor);
  } else {
    auto reduced_tensor = cute::make_tensor<float>(
      Shape<Int<(tensor_size-1)/2 + 1>>{}
    );
    for_each(make_int_sequence<(tensor_size-1)/2>{}, [&](auto i){
      reduced_tensor(i) = fmaxf(tensor(i*2), tensor(i*2+1));
    });
    // simply carry the last element
    reduced_tensor((tensor_size-1)/2) = tensor(tensor_size-1);
    return _tensor_max_binary_reduce(reduced_tensor);
  }
}

template <class EngineTensor, class LayoutTensor>
CUTE_DEVICE constexpr auto _tensor_airth_max_binary_reduce(
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
      auto val0 = tensor(i*2);
      auto val1 = tensor(i*2+1);
      reduced_tensor(i) = (val0 + val1 + fabs(val0 - val1)) / 2;
    });
    return _tensor_airth_max_binary_reduce(reduced_tensor);
  } else {
    auto reduced_tensor = cute::make_tensor<float>(
      Shape<Int<(tensor_size-1)/2 + 1>>{}
    );
    for_each(make_int_sequence<(tensor_size-1)/2>{}, [&](auto i){
      auto val0 = tensor(i*2);
      auto val1 = tensor(i*2+1);
      reduced_tensor(i) = (val0 + val1 + fabs(val0 - val1)) / 2;
    });
    // simply carry the last element
    reduced_tensor((tensor_size-1)/2) = tensor(tensor_size-1);
    return _tensor_airth_max_binary_reduce(reduced_tensor);
  }
}

template <bool zero_init = true, int fmax_ratio, class EngineTensor, class LayoutTensor, class EngineMax, class LayoutMax>
CUTE_DEVICE void reduce_max_binary_max(
  Tensor<EngineTensor, LayoutTensor> const &tensor,
  Tensor<EngineMax, LayoutMax> &max_tensor
){
  constexpr int TensorM = cute::size<0>(LayoutTensor{});
  constexpr int TensorN = cute::size<1>(LayoutTensor{});
  constexpr int MaxM = cute::size<0>(LayoutMax{});
  static_assert(MaxM == TensorM, "Max tensor size should be equal to the input tensor size");
  
  CUTE_UNROLL
  for (int mi = 0; mi < TensorM; ++mi){
    auto max_val = _tensor_max_binary_reduce(tensor(mi, _))(Int<0>{});
    if constexpr (zero_init) {
      max_tensor(mi) = max_val;
    } else {
      max_tensor(mi) = fmaxf(max_tensor(mi), max_val);
    }
  }
}

template<bool zero_init=true, int fmax_ratio, typename EngineTensor, typename LayoutTensor, typename EngineMax, typename LayoutMax>
CUTE_DEVICE void reduce_max_binary_arith(
  Tensor<EngineTensor, LayoutTensor> const& tensor,
  Tensor<EngineMax, LayoutMax> &max_tensor
){
  constexpr int TensorM = cute::size<0>(LayoutTensor{});
  constexpr int TensorN = cute::size<1>(LayoutTensor{});
  constexpr int MaxM = cute::size<0>(LayoutMax{});
  static_assert(MaxM == TensorM, "Max tensor size should be equal to the input tensor size");
  
  CUTE_UNROLL
  for (int mi = 0; mi < TensorM; ++mi){
    auto max_val = _tensor_airth_max_binary_reduce(tensor(mi, _))(Int<0>{});
    if constexpr (zero_init) {
      max_tensor(mi) = max_val;
    } else {
      max_tensor(mi) = fmaxf(max_tensor(mi), max_val);
    }
  }
}

template<bool zero_init=true, int fmax_ratio, typename EngineTensor, typename LayoutTensor, typename EngineMax, typename LayoutMax>
CUTE_DEVICE void reduce_max_binary_max_arith_blend(
  Tensor<EngineTensor, LayoutTensor> const& tensor,
  Tensor<EngineMax, LayoutMax> &max_tensor
){
  [[maybe_unused]] constexpr int TensorM = cute::size<0>(LayoutTensor{});
  [[maybe_unused]] constexpr int MMA_M = TensorM / 2;
  [[maybe_unused]] constexpr int TensorN = cute::size<1>(LayoutTensor{});
  [[maybe_unused]] constexpr int MMA_N = TensorN / 2;
  static_assert(fmax_ratio <= MMA_N, "fmax_ratio should be <= MMA_N");
  [[maybe_unused]] constexpr int MaxM = cute::size<0>(LayoutMax{});
  static_assert(MaxM == TensorM, "Max tensor size should be equal to the input tensor size");
  
  if constexpr (fmax_ratio == 0) {
    // all arithmetic
    return reduce_max_binary_arith<zero_init, fmax_ratio>(tensor, max_tensor);
  } else if constexpr (fmax_ratio == MMA_N) {
    // all fmax
    return reduce_max_binary_max<zero_init, fmax_ratio>(tensor, max_tensor);
  } else {
    /*
      max_tensor: ptr[32b](0x762ccffffa00) o (_4):(_1)
      tensor: ptr[32b](0x762ccffffa80) o ((_2,_2),(_2,_8)):((_2,_4),(_1,_8))
      ((2, MMA_M),(2,MMA_N))
    */

    CUTE_UNROLL
    for (int mi = 0; mi < TensorM; ++mi){

      auto tensor_row_slice = tensor(mi, _);
      auto tensor_row_fmax_part = cute::make_tensor<float>(Shape<Int<2*fmax_ratio>>{});
      auto tensor_row_arith_part = cute::make_tensor<float>(Shape<Int<2*(MMA_N-fmax_ratio)>>{});

      // copy the fmax part
      CUTE_UNROLL
      for (int i = 0; i < fmax_ratio; ++i){
        tensor_row_fmax_part(i * 2) = tensor_row_slice(i * 2);
        tensor_row_fmax_part(i * 2 + 1) = tensor_row_slice(i * 2 + 1);
      }
      // copy the arith part
      CUTE_UNROLL
      for (int i = 0; i < MMA_N-fmax_ratio; ++i){
        tensor_row_arith_part(i * 2) = tensor_row_slice((i + fmax_ratio) * 2);
        tensor_row_arith_part(i * 2 + 1) = tensor_row_slice((i + fmax_ratio) * 2 + 1);
      }

      auto fmax_part_max = _tensor_max_binary_reduce(tensor_row_fmax_part)(Int<0>{});
      auto arith_part_max = _tensor_airth_max_binary_reduce(tensor_row_arith_part)(Int<0>{});
      auto slice_max = fmaxf(fmax_part_max, arith_part_max);
      if constexpr (zero_init) {
        max_tensor(mi) = slice_max;
      } else {
        max_tensor(mi) = fmaxf(max_tensor(mi), slice_max);
      }
    }
  }
}

} // namespace FLASH_NAMESPACE