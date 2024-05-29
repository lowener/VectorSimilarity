#pragma once

#include <cstdlib> // size_t
#include <memory>  // std::shared_ptr

#include "VecSim/vec_sim.h"              //typedef VecSimIndex
#include "VecSim/vec_sim_common.h"       // CuvsIvfParams
#include "VecSim/memory/vecsim_malloc.h" // VecSimAllocator
#include "VecSim/vec_sim_index.h"

namespace CuvsIvfFactory {

VecSimIndex *NewIndex(const VecSimParams *params);
VecSimIndex *NewIndex(const CuvsIvfParams *params);
size_t EstimateInitialSize(const CuvsIvfParams *params);
size_t EstimateElementSize(const CuvsIvfParams *params);

}; // namespace CuvsIvfFactory
