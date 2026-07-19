#pragma once

#include <cute/tensor.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"  // For cutlass::FastDivmod
#include "utils.h"

/*
  AccO (Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_MNK_PV{}));) and 
  AccS (Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));)
  traits extraction meta templates
*/

namespace flash {

using namespace cute;

template <class AccOTensor>
struct FlashFwdAccOTensorTraits;

template <class AccOEngine, class AccOLayout>
struct FlashFwdAccOTensorTraits<Tensor<AccOEngine, AccOLayout>> {

  // AccO example: tOrO: ptr[32b](0x7f2695fffb50) o ((_2,_2,_16),_1,_1):((_1,_2,_4),_0,_0)

  static constexpr int Acc_O_MMA_2M = 2;
  static_assert(cute::size<0,1>(AccOLayout{}) == Acc_O_MMA_2M, "Acc_O_MMA_2M must be 2");
  static constexpr int Acc_O_MMA_2N = 2;
  static_assert(cute::size<0,0>(AccOLayout{}) == Acc_O_MMA_2N, "Acc_O_MMA_2N must be 2");

  static constexpr int Acc_O_MMA_V = cute::size<0,2>(AccOLayout{});

  static constexpr int Acc_O_MMA_M = cute::size<1>(AccOLayout{});
  static constexpr int Acc_O_MMA_N = cute::size<2>(AccOLayout{});

  

  CUTE_DEVICE static void printout_layout() {
    if (cute::thread(128, 0)){
      printf("AccOLayout: ");
      print(AccOLayout{});
      printf("\n");
    }
  }

};

template <class AccSTensor>
struct FlashFwdAccSTensorTraits;

template <class AccSEngine, class AccSLayout>
struct FlashFwdAccSTensorTraits <Tensor<AccSEngine, AccSLayout>> {

  // AccS example: tSrS: ptr[32b](0x7f2695fffb50) o ((_2,_2,_4),_1,_7):((_1,_2,_4),_0,_16)

  static constexpr int Acc_S_MMA_2M = 2;
  static_assert(cute::size<0,1>(AccSLayout{}) == Acc_S_MMA_2M, "Acc_S_MMA_2M must be 2");
  static constexpr int Acc_S_MMA_2N = 2;
  static_assert(cute::size<0,0>(AccSLayout{}) == Acc_S_MMA_2N, "Acc_S_MMA_2N must be 2");
  
  static constexpr int Acc_S_MMA_V = cute::size<0,2>(AccSLayout{});

  static constexpr int Acc_S_MMA_M = cute::size<1>(AccSLayout{});
  static constexpr int Acc_S_MMA_N = cute::size<2>(AccSLayout{});

  CUTE_DEVICE static void printout_layout() {
    if (cute::thread(128, 0)){
      printf("AccSLayout: ");
      print(AccSLayout{});
      printf("\n");
    }
  }

};

template <int kBlockM, int kBlockN, bool PackGQA, class TiledMma, bool SwapAB>
struct FlashFwdAccSRowIdxGetter {
  
  int const thread_idx_with_offset;
  int const m_block;
  cutlass::FastDivmod const qhead_per_khead_divmod;

  static_assert(!SwapAB, "FlashFwdAccSRowIdxGetter only supports SwapAB = false");
  
  CUTE_DEVICE FlashFwdAccSRowIdxGetter(int const thread_idx_with_offset, int const m_block, cutlass::FastDivmod const &qhead_per_khead_divmod)
    : thread_idx_with_offset(thread_idx_with_offset)
    , m_block(m_block)
    , qhead_per_khead_divmod(qhead_per_khead_divmod)
  {}

  /*
    m_idx within range ((MMA_2M, MMA_M))  
  */
  CUTE_DEVICE int get_row_idx(int m_idx) {
    auto thread_mma = TiledMma{}.get_thread_slice(thread_idx_with_offset);
    [[maybe_unused]] static constexpr int Row = !SwapAB ? 0 : 1;
    [[maybe_unused]] static constexpr int Col = !SwapAB ? 1 : 0;

    Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
    Tensor tScS = thread_mma.partition_C(cS);
    Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
    
    // If PackGQA, we split the work of compute divmod among threads in the same row
    static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});

    if constexpr (PackGQA) {
      int mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx_with_offset % kMmaThreadsPerRow, _0{})));
      int const row_idx = __shfl_sync(0xffffffff, mma_m_idx, m_idx % kMmaThreadsPerRow, kMmaThreadsPerRow);
      return row_idx;
    } else {
      int const row_idx = get<Row>(tScS_rowcol(m_idx, _0{})) + m_block * kBlockM;
      return row_idx;
    }
  }
  
};


/*
  NOTE: for acc_s hmma rescale fragC
*/
template <int NumMmaWarps, bool Expanded = false>
struct FlashFwdSharedMmaMaxScaledLayout {
  using Layout_NonExpand = decltype(cute::make_layout(
    cute::make_shape(Int<4>{}, Int<NumMmaWarps>{}),
    GenColMajor{}
  ));
  static constexpr int _rank_non_expand = Layout_NonExpand::rank;
  // the "8" mode is the "warp expand complement" mode
  using Layout_Expand = decltype(cute::make_layout(
    cute::make_shape(Int<4>{}, Int<8>{}, Int<NumMmaWarps>{}),
    GenColMajor{}
  ));
  static constexpr int _rank_expand = Layout_Expand::rank;

  using Layout = typename std::conditional_t<Expanded, Layout_Expand, Layout_NonExpand>;

  template <class EngineTensor, class LayoutTensor>
  CUTE_DEVICE static void store(
    cute::Tensor<EngineTensor, LayoutTensor> &smem_tensor,
    float const& mma_max_scaled,
    int const& mma_warp_idx,
    int const& lane_id
  ){
    constexpr int layout_rank = LayoutTensor::rank;
    if constexpr (layout_rank == _rank_non_expand) {
      // non-expanded layout case, lane 0 write
      auto temp_tensor = cute::make_tensor<float>(
        cute::make_layout(
          cute::make_shape(Int<4>{}),
          GenColMajor{}
        )
      );
      temp_tensor(Int<0>{}) = mma_max_scaled;
      temp_tensor(Int<1>{}) = mma_max_scaled;
      temp_tensor(Int<2>{}) = mma_max_scaled;
      temp_tensor(Int<3>{}) = mma_max_scaled;
      if (lane_id == 0) {
        using CopyAtom = Copy_Atom<UniversalCopy<uint128_t>, float>;
        cute::copy(CopyAtom{}, temp_tensor, smem_tensor(_, mma_warp_idx));
      }
    } else if constexpr (layout_rank == _rank_expand) {
      // expanded layout case, all lanes write with compiler fooling swizzle
      auto temp_tensor = cute::make_tensor<float>(
        cute::make_layout(
          cute::make_shape(Int<4>{}),
          GenColMajor{}
        )
      );
      temp_tensor(Int<0>{}) = mma_max_scaled;
      temp_tensor(Int<1>{}) = mma_max_scaled;
      temp_tensor(Int<2>{}) = mma_max_scaled;
      temp_tensor(Int<3>{}) = mma_max_scaled;
      using CopyAtom = Copy_Atom<UniversalCopy<uint128_t>, float>;
      int const target_lane_id = lane_id >> 2;
      cute::copy(CopyAtom{}, temp_tensor, smem_tensor(_, target_lane_id, mma_warp_idx));
    } else {
      static_assert(layout_rank == _rank_non_expand || layout_rank == _rank_expand, "Invalid rank");
    }
  }

  template <class EngineTensor, class LayoutTensor>
  CUTE_DEVICE static auto load(
    cute::Tensor<EngineTensor, LayoutTensor> const& smem_tensor,
    int const& mma_warp_idx,
    [[maybe_unused]] int const& lane_id
  ){
    constexpr int layout_rank = LayoutTensor::rank;
    if constexpr (layout_rank == _rank_non_expand) {
      // non-expanded layout case, lane 0 write
      auto result_frag = cute::make_tensor<float>(
        cute::make_layout(make_shape(Int<2>{}, Int<2>{}), GenColMajor{})
      );
      using CopyAtom = Copy_Atom<UniversalCopy<uint128_t>, float>;
      cute::copy(CopyAtom{}, smem_tensor(_, mma_warp_idx), result_frag);
      return result_frag;
    } else if constexpr (layout_rank == _rank_expand) {
      // expanded layout case, all lanes write with compiler fooling swizzle
      auto result_frag = cute::make_tensor<float>(
        cute::make_layout(make_shape(Int<2>{}, Int<2>{}), GenColMajor{})
      );
      using CopyAtom = Copy_Atom<UniversalCopy<uint128_t>, float>;
      int const target_lane_id = 8 - (lane_id >> 2);
      cute::copy(CopyAtom{}, smem_tensor(_, target_lane_id, mma_warp_idx), result_frag);
      return result_frag;
    } else {
      static_assert(layout_rank == _rank_non_expand || layout_rank == _rank_expand, "Invalid rank");
    }
  }
};

/*
  NOTE: for acc_o hmma rescale fragB
  NOTE: acc o scale is actually meant to be used in the same stage where it were computed
  so no stage mode/index is needed
  layout is exactly 2 B regs per mma warp
*/
template <int NumMmaWarps, bool Expanded = false>
struct FlashFwdSharedHMMAScoresScaleLayout {
  using Layout_NonExpand = decltype(cute::make_layout(
    cute::make_shape(Int<2>{}, Int<NumMmaWarps>{}),
    GenColMajor{}
  ));
  static constexpr int _rank_non_expand = Layout_NonExpand::rank;
  // the "16" mode is the "warp expand complement" mode
  using Layout_Expand = decltype(cute::make_layout(
    cute::make_shape(Int<2>{}, Int<16>{}, Int<NumMmaWarps>{}),
    GenColMajor{}
  ));
  static constexpr int _rank_expand = Layout_Expand::rank;
  using Layout = typename std::conditional_t<Expanded, Layout_Expand, Layout_NonExpand>;

  template <class EngineTensor, class LayoutTensor>
  CUTE_DEVICE static void store(
    cute::Tensor<EngineTensor, LayoutTensor> &smem_tensor,
    float const& hmma_scale_frag_0,
    float const& hmma_scale_frag_1,
    int const& mma_warp_idx,
    [[maybe_unused]] int const& lane_id
  ){
    constexpr int layout_rank = LayoutTensor::rank;
    if constexpr (layout_rank == _rank_non_expand) {
      // non-expanded layout case, lane 0 write
      auto temp_tensor = cute::make_tensor<float>(
        cute::make_layout(
          cute::make_shape(Int<2>{}), GenColMajor{}
        )
      );
      temp_tensor(Int<0>{}) = hmma_scale_frag_0;
      temp_tensor(Int<1>{}) = hmma_scale_frag_1;
      if (lane_id == 0) {
        using CopyAtom = Copy_Atom<UniversalCopy<uint64_t>, float>;
        cute::copy(CopyAtom{}, temp_tensor, smem_tensor(_, mma_warp_idx));
      }
    } else if constexpr (layout_rank == _rank_expand) {
      // expanded layout case, all lanes write with compiler fooling swizzle
      auto temp_tensor = cute::make_tensor<float>(
        cute::make_layout(
          cute::make_shape(Int<2>{}), GenColMajor{}
        )
      );
      temp_tensor(Int<0>{}) = hmma_scale_frag_0;
      temp_tensor(Int<1>{}) = hmma_scale_frag_1;
      using CopyAtom = Copy_Atom<UniversalCopy<uint64_t>, float>;
      int const target_lane_id = lane_id >> 1;
      cute::copy(CopyAtom{}, temp_tensor, smem_tensor(_, target_lane_id, mma_warp_idx));
    } else {
      static_assert(layout_rank == _rank_non_expand || layout_rank == _rank_expand, "Invalid rank");
    }
  }

  template <class EngineTensor, class LayoutTensor>
  CUTE_DEVICE static auto load(
    cute::Tensor<EngineTensor, LayoutTensor> const& smem_tensor,
    int const& mma_warp_idx,
    [[maybe_unused]] int const& lane_id
  ){
    constexpr int layout_rank = LayoutTensor::rank;
    if constexpr (layout_rank == _rank_non_expand) {
      // non-expanded layout case, lane 0 write
      auto result_frag = cute::make_tensor<float>(
        cute::make_layout(
          cute::make_shape(Int<2>{}), GenColMajor{}
        )
      );
      using CopyAtom = Copy_Atom<UniversalCopy<uint64_t>, float>;
      cute::copy(CopyAtom{}, smem_tensor(_, mma_warp_idx), result_frag);
      return result_frag;
    } else if constexpr (layout_rank == _rank_expand) {
      // expanded layout case, all lanes write with compiler fooling swizzle
      auto result_frag = cute::make_tensor<float>(
        cute::make_layout(
          cute::make_shape(Int<2>{}), GenColMajor{}
        )
      );
      using CopyAtom = Copy_Atom<UniversalCopy<uint64_t>, float>;
      int const target_lane_id = 16 - (lane_id >> 1);
      cute::copy(CopyAtom{}, smem_tensor(_, target_lane_id, mma_warp_idx), result_frag);
      return result_frag;
    } else {
      static_assert(layout_rank == _rank_non_expand || layout_rank == _rank_expand, "Invalid rank");
    }
  }

};


/*
  Peel WGMMA accumulator tensor
  from
  ((MMA_2N, MMA_2M, MMA_V), MMA_M, MMA_N)
  to
  ((MMA_2N, MMA_2M), MMA_V, MMA_M, MMA_N)
*/
template <class AccTensorEngine, class AccTensorLayout>
CUTE_DEVICE auto peel_wgmma_accumulator_tensor(
  Tensor<AccTensorEngine, AccTensorLayout> & acc_tensor
){
  using LayoutFlat = decltype(AccTensorLayout{}(make_coord(_,_,_),_,_));
  using LayoutReGrouped = decltype(
    cute::group<0, 2>(LayoutFlat{})
  );
  return cute::make_tensor(
    acc_tensor.data(),
    LayoutReGrouped{}
  );
}

/*
  NOTE: for acc_s wgmma rescale, fragB
  FFMA WGMMA B (softmax scale log2) shared memory layout meta
*/
template <bool UseSWINTER_ = false>
struct FlashFwdSharedWGMMASoftmaxScaleLog2Layout {
  
  static constexpr bool UseSWINTER = UseSWINTER_;
  
  using sB_M = Int<8>;
  using sB_N = Int<8>;

  // generally: (N8, K8)
  // cutlass 360: Sw<0,4,3> o smem_ptr[32b](unset) o (_8,(_4,_2)):(_4,(_1,_32))
  // cutlass 392: Sw<0,4,3> o smem_ptr[32b](unset) o ((_8,_1),(_4,_2)):((_4,_0),(_1,_32))
  using Layout_sB_K_INTER = decltype(cute::tile_to_shape(
    GMMA::Layout_K_INTER_Atom<cute::tfloat32_t>{},
    cute::make_shape(sB_M{},sB_N{})
  ));

  // generally: (N8, K8)
  // cutlass 360: Sw<1,4,3> o smem_ptr[32b](unset) o (_8,_8):(_8,_1)
  // cutlass 392: Sw<1,4,3> o smem_ptr[32b](unset) o ((_8,_1),(_8,_1)):((_8,_0),(_1,_0))
  using Layout_sB_K_SW32 = decltype(cute::tile_to_shape(
    GMMA::Layout_K_SW32_Atom<cute::tfloat32_t>{},
    cute::make_shape(sB_M{},sB_N{})
  ));

  using Layout = std::conditional_t<
    UseSWINTER,
    Layout_sB_K_INTER,
    Layout_sB_K_SW32
  >;

  template <class EngineTensor, class LayoutTensor>
  CUTE_DEVICE static void init_smem_softmax_scale_log2(
    Tensor<EngineTensor, LayoutTensor> &smem_softmax_scale_tensor,
    float const& softmax_scale_log2,
    int const& mma_warp_idx
  ){
    static_assert(UseSWINTER == false, "UseSWINTER is not supported yet");
    int const lane_id = cutlass::canonical_lane_idx();

    if (mma_warp_idx == 0){
      // fake logic for now

      // (8K, 4N1, 2N2) -> ((4N1, 2N2), 8K)
      using WriteLayout_SW32 = decltype(
        make_layout(
          make_shape(Int<8>{}, Int<4>{}, Int<2>{}),
          make_stride(Int<8>{}, Int<1>{}, Int<4>{})
        )
      );
      auto smem_softmax_scale_tensor_write_composed = cute::composition(
        smem_softmax_scale_tensor,
        WriteLayout_SW32{}
      );

      int lane_k_idx = lane_id % 8;
      int lane_n1_idx = lane_id / 8;
      CUTE_UNROLL
      for (int n2_idx = 0; n2_idx < 2; n2_idx++){
        smem_softmax_scale_tensor_write_composed(lane_k_idx, lane_n1_idx, n2_idx) = cute::tfloat32_t(softmax_scale_log2);
      }

    }
  }
};

template <bool UseSWINTER_ = false>
struct FlashFwdSharedWGMMAFaddReduceBLayout {
  
  static constexpr bool UseSWINTER = UseSWINTER_;
  
  using sB_M = Int<8>;
  using sB_N = Int<8>;

  // generally: (N8, K8)
  // cutlass 360: Sw<0,4,3> o smem_ptr[32b](unset) o (_8,(_4,_2)):(_4,(_1,_32))
  // cutlass 392: Sw<0,4,3> o smem_ptr[32b](unset) o ((_8,_1),(_4,_2)):((_4,_0),(_1,_32))
  using Layout_sB_K_INTER = decltype(cute::tile_to_shape(
    GMMA::Layout_K_INTER_Atom<cute::tfloat32_t>{},
    cute::make_shape(sB_M{},sB_N{})
  ));

  // generally: (N8, K8)
  // cutlass 360: Sw<1,4,3> o smem_ptr[32b](unset) o (_8,_8):(_8,_1)
  // cutlass 392: Sw<1,4,3> o smem_ptr[32b](unset) o ((_8,_1),(_8,_1)):((_8,_0),(_1,_0))
  using Layout_sB_K_SW32 = decltype(cute::tile_to_shape(
    GMMA::Layout_K_SW32_Atom<cute::tfloat32_t>{},
    cute::make_shape(sB_M{},sB_N{})
  ));

  using Layout = std::conditional_t<
    UseSWINTER,
    Layout_sB_K_INTER,
    Layout_sB_K_SW32
  >;

  enum class InitStrategy {
    MMAWarp0Only, // only mma warp 0 is responsible for initializing the smem
    MMAWarp01Only, // only mma warp 0 and 1 are responsible for initializing the smem
    ProducerWarp0Only, // only producer warp 0 is responsible for initializing the smem
    ProducerWarp01Only // only producer warp 0 and 1 are responsible for initializing the smem
  };

  template <bool isCalledInMMA, class EngineTensor, class LayoutTensor>
  CUTE_DEVICE static void init_smem_fadd_reduce_B(
    Tensor<EngineTensor, LayoutTensor> &smem_fadd_reduce_B_tensor,
    int const& mmaORproducer_warp_idx
  ){
    static_assert(UseSWINTER == false, "UseSWINTER is not supported yet");
    int const lane_id = cutlass::canonical_lane_idx();
    // (8K, 4N1, 2N2) -> ((4N1, 2N2), 8K)
    using WriteLayout_SW32 = decltype(
      make_layout(
        make_shape(Int<8>{}, Int<4>{}, Int<2>{}),
        make_stride(Int<8>{}, Int<1>{}, Int<4>{})
      )
    );
    // layout (8K, 4N1, 2N2) -> (N8, K8)
    auto smem_tensor_composed = cute::composition(
      smem_fadd_reduce_B_tensor,
      WriteLayout_SW32{}
    );
    
    constexpr InitStrategy strategy = InitStrategy::MMAWarp01Only; // Change this to the desired strategy

    // return guard
    if constexpr (isCalledInMMA && (strategy == InitStrategy::ProducerWarp01Only || strategy == InitStrategy::ProducerWarp0Only)) {
      return;
    } else if constexpr (!isCalledInMMA && (strategy == InitStrategy::MMAWarp01Only || strategy == InitStrategy::MMAWarp0Only)) {
      return;
    }

    auto func_valuegen_core = [](int lane_k_idx, int lane_n1_idx, int n2_idx) -> float {
      float fill_val;
      int lane_n_idx = lane_n1_idx + n2_idx * 4;
      if (lane_n_idx % 2 == 0) {
        if (lane_k_idx < 4) {
          fill_val = 1.0f;
        } else {
          fill_val = 0.0f;
        }
      } else {
        if (lane_k_idx < 4) {
          fill_val = 0.0f;
        } else {
          fill_val = 1.0f;
        }
      }
      return fill_val;
    };

    // TODO: revise these fill logic to be tiled_copy based
    // just like FA2
    
    if constexpr (strategy == InitStrategy::MMAWarp0Only) {
      // strategy 1: only mma warp 0 is responsible for initializing the smem
      if (mmaORproducer_warp_idx != 0) return;
      int lane_k_idx = lane_id % 8;
      int lane_n1_idx = lane_id / 8;
      CUTE_UNROLL
      for (int n2_idx = 0; n2_idx < 2; n2_idx++){
        float fill_val = func_valuegen_core(lane_k_idx, lane_n1_idx, n2_idx);
        smem_tensor_composed(lane_k_idx, lane_n1_idx, n2_idx) = cute::tfloat32_t(fill_val);
      }
    } else if constexpr (strategy == InitStrategy::MMAWarp01Only) {
      // strategy 2: only mma warp 0 and 1 are responsible for initializing the smem
      if (mmaORproducer_warp_idx > 1) return;
      int lane_k_idx = lane_id % 8;
      int lane_n1_idx = lane_id / 8;
      int n2_idx = mmaORproducer_warp_idx;
      float fill_val = func_valuegen_core(lane_k_idx, lane_n1_idx, n2_idx);
      smem_tensor_composed(lane_k_idx, lane_n1_idx, n2_idx) = cute::tfloat32_t(fill_val);
    } else if constexpr (strategy == InitStrategy::ProducerWarp0Only) {
      // strategy 3: only producer warp 0 is responsible for initializing the smem
      if (mmaORproducer_warp_idx != 0) return;
      int lane_k_idx = lane_id % 8;
      int lane_n1_idx = lane_id / 8;
      CUTE_UNROLL
      for (int n2_idx = 0; n2_idx < 2; n2_idx++){
        float fill_val = func_valuegen_core(lane_k_idx, lane_n1_idx, n2_idx);
        smem_tensor_composed(lane_k_idx, lane_n1_idx, n2_idx) = cute::tfloat32_t(fill_val);
      }
    } else if constexpr (strategy == InitStrategy::ProducerWarp01Only) {
      // strategy 4: only producer warp 0 and 1 are responsible for initializing the smem
      if (mmaORproducer_warp_idx > 1) return;
      int lane_k_idx = lane_id % 8;
      int lane_n1_idx = lane_id / 8;
      int n2_idx = mmaORproducer_warp_idx;
      float fill_val = func_valuegen_core(lane_k_idx, lane_n1_idx, n2_idx);
      smem_tensor_composed(lane_k_idx, lane_n1_idx, n2_idx) = cute::tfloat32_t(fill_val);
    }
  }
};

struct FlashFwdWGMMAReduceMeta {
  using ArchAtom = GMMA::MMA_64x8x8_F32TF32TF32_RS_TN<>;
  using MmaTraits = MMA_Traits<ArchAtom>;
  using MmaAtom = MMA_Atom<ArchAtom>;
  using TiledMma = decltype(
    make_tiled_mma(
      MmaAtom{},
      Layout<Shape<_1,_1,_1>, Stride<_0,_0,_0>>{}
    )
  );
  
  template <class EngineTensor, class LayoutTensor>
  CUTE_DEVICE static constexpr auto get_thr_fragB(
    Tensor<EngineTensor, LayoutTensor> const& smem_B_tensor,
    int const& thread_idx_with_offset
  ){
    constexpr int warpgroup_size = 128;
    int warpgroup_lane_idx = thread_idx_with_offset % warpgroup_size;

    auto tiled_mma = TiledMma{};
    auto thr_mma = tiled_mma.get_slice(warpgroup_lane_idx);
    auto smem_B_partition = thr_mma.partition_B(smem_B_tensor);
    auto smem_B_frag = thr_mma.make_fragment_B(smem_B_partition);
    return smem_B_frag;
  }
};

} // end namespace flash
