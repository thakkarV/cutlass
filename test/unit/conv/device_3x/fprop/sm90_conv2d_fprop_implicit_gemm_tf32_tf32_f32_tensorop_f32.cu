/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include "cutlass_unit_test.h"

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"

#include "cutlass/conv/device/conv_universal_adapter.hpp"
#include "cutlass/conv/kernel/conv_universal.hpp"
#include "cutlass/conv/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "../testbed_conv.hpp"
using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

//////////////////////////////////////////////////////////////////////////////////////////////////
// Tile shape 64x64x32
//////////////////////////////////////////////////////////////////////////////////////////////////

//
// Cluster 1x1x1
//

TEST(SM90_device_conv2d_fprop_implicitgemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32, 64x64x32_1x1x1) {
  using ElementAct     = float;
  using ElementFlt     = float;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_64, _64, Shape<_32>>;
  using ClusterShapeMNK = Shape<_1,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      float, cutlass::layout::TensorNHWC, 4,
      float, cutlass::layout::TensorNHWC, 4,
      cutlass::epilogue::TmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kFprop,
      ElementAct, cutlass::layout::TensorNHWC, 4,
      ElementFlt, cutlass::layout::TensorNHWC, 4,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>(/*alpha=*/1.0, /*beta=*/1.0));
}

//
// Cluster 2x1x1
//

TEST(SM90_device_conv2d_fprop_implicitgemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32, 64x64x32_2x1x1) {
  using ElementAct     = float;
  using ElementFlt     = float;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_64, _64, Shape<_32>>;
  using ClusterShapeMNK = Shape<_2,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      float, cutlass::layout::TensorNHWC, 4,
      float, cutlass::layout::TensorNHWC, 4,
      cutlass::epilogue::TmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kFprop,
      ElementAct, cutlass::layout::TensorNHWC, 4,
      ElementFlt, cutlass::layout::TensorNHWC, 4,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>(/*alpha=*/1.0, /*beta=*/1.0));
}

//
// Cluster 1x2x1
//

TEST(SM90_device_conv2d_fprop_implicitgemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32, 64x64x32_1x2x1) {
  using ElementAct     = float;
  using ElementFlt     = float;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_64, _64, Shape<_32>>;
  using ClusterShapeMNK = Shape<_1,_2,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      float, cutlass::layout::TensorNHWC, 4,
      float, cutlass::layout::TensorNHWC, 4,
      cutlass::epilogue::TmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kFprop,
      ElementAct, cutlass::layout::TensorNHWC, 4,
      ElementFlt, cutlass::layout::TensorNHWC, 4,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>(/*alpha=*/1.0, /*beta=*/1.0));
}

//
// Cluster 2x2x1
//

TEST(SM90_device_conv2d_fprop_implicitgemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32, 64x64x32_2x2x1) {
  using ElementAct     = float;
  using ElementFlt     = float;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_64, _64, Shape<_32>>;
  using ClusterShapeMNK = Shape<_2,_2,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      float, cutlass::layout::TensorNHWC, 4,
      float, cutlass::layout::TensorNHWC, 4,
      cutlass::epilogue::TmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kFprop,
      ElementAct, cutlass::layout::TensorNHWC, 4,
      ElementFlt, cutlass::layout::TensorNHWC, 4,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>(/*alpha=*/1.0, /*beta=*/1.0));
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Tile shape 128x64x32
//////////////////////////////////////////////////////////////////////////////////////////////////

//
// Cluster 1x1x1
//

TEST(SM90_device_conv2d_fprop_implicitgemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32, 128x64x32_1x1x1) {
  using ElementAct     = float;
  using ElementFlt     = float;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_128, _64, Shape<_32>>;
  using ClusterShapeMNK = Shape<_1,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      float, cutlass::layout::TensorNHWC, 4,
      float, cutlass::layout::TensorNHWC, 4,
      cutlass::epilogue::TmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kFprop,
      ElementAct, cutlass::layout::TensorNHWC, 4,
      ElementFlt, cutlass::layout::TensorNHWC, 4,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>(/*alpha=*/1.0, /*beta=*/1.0));
}

//
// Cluster 2x1x1
//

TEST(SM90_device_conv2d_fprop_implicitgemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32, 128x64x32_2x1x1) {
  using ElementAct     = float;
  using ElementFlt     = float;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_128, _64, Shape<_32>>;
  using ClusterShapeMNK = Shape<_2,_1,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      float, cutlass::layout::TensorNHWC, 4,
      float, cutlass::layout::TensorNHWC, 4,
      cutlass::epilogue::TmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kFprop,
      ElementAct, cutlass::layout::TensorNHWC, 4,
      ElementFlt, cutlass::layout::TensorNHWC, 4,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>(/*alpha=*/1.0, /*beta=*/1.0));
}

//
// Cluster 1x2x1
//

TEST(SM90_device_conv2d_fprop_implicitgemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32, 128x64x32_1x2x1) {
  using ElementAct     = float;
  using ElementFlt     = float;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_128, _64, Shape<_32>>;
  using ClusterShapeMNK = Shape<_1,_2,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      float, cutlass::layout::TensorNHWC, 4,
      float, cutlass::layout::TensorNHWC, 4,
      cutlass::epilogue::TmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kFprop,
      ElementAct, cutlass::layout::TensorNHWC, 4,
      ElementFlt, cutlass::layout::TensorNHWC, 4,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>(/*alpha=*/1.0, /*beta=*/1.0));
}

//
// Cluster 2x2x1
//

TEST(SM90_device_conv2d_fprop_implicitgemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32, 128x64x32_2x2x1) {
  using ElementAct     = float;
  using ElementFlt     = float;
  using ElementOut     = float;
  using ElementAcc     = float;
  using ElementCompute = float;
  using TileShapeMNK = Shape<_128, _64, Shape<_32>>;
  using ClusterShapeMNK = Shape<_2,_2,_1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      float, cutlass::layout::TensorNHWC, 4,
      float, cutlass::layout::TensorNHWC, 4,
      cutlass::epilogue::TmaWarpSpecialized
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::conv::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      cutlass::conv::Operator::kFprop,
      ElementAct, cutlass::layout::TensorNHWC, 4,
      ElementFlt, cutlass::layout::TensorNHWC, 4,
      ElementAcc,
      TileShapeMNK, ClusterShapeMNK,
      cutlass::conv::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::conv::collective::KernelScheduleAuto
    >::CollectiveOp;

  using ConvKernel = cutlass::conv::kernel::ConvUniversal<
      CollectiveMainloop,
      CollectiveEpilogue
    >;

  using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;

  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
  EXPECT_TRUE(test::conv::device::TestAllConv<Conv>(/*alpha=*/1.0, /*beta=*/1.0));
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
