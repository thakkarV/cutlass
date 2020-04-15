/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Definitions for SRGEMM configuration structures.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/min_op.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

template <
  typename OperatorClass,
  typename ArchTag,
  typename ElementA, 
  typename ElementB, 
  typename ElementC,
  typename ElementAccumulator
>
struct DefaultSrgemmConfiguration;

////////////////////////////////////////////////////////////////////////////////

template <
  typename ArchTag,
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename ElementAccumulator>
struct DefaultSrgemmConfiguration<
  arch::OpClassSimt, 
  ArchTag,
  ElementA,
  ElementB,
  ElementC,
  ElementAccumulator> {
  
  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  using ThreadblockShape = GemmShape<64, 128, 8>;
  using WarpShape = GemmShape<16, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::MinOp<ElementC, 1>;
  using Operator = arch::OpSumMin;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
