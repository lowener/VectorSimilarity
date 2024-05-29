#include "VecSim/index_factories/brute_force_factory.h"
#include "VecSim/algorithms/cuvs_ivf/ivf_tiered.h"
#include "VecSim/algorithms/cuvs_ivf/ivf_interface.h"
#include "VecSim/index_factories/tiered_factory.h"
#include "VecSim/index_factories/cuvs_ivf_factory.h"

namespace TieredCuvsIvfFactory {

VecSimIndex *NewIndex(const TieredIndexParams *params) {
    assert(params->primaryIndexParams->algoParams.cuvsIvfParams.type == VecSimType_FLOAT32 &&
           "Invalid IVF data type algorithm");

    using DataType = float;
    using DistType = float;
    // initialize cuvs index
    auto *cuvs_index = reinterpret_cast<CuvsIvfInterface<DataType, DistType> *>(
        CuvsIvfFactory::NewIndex(params->primaryIndexParams));
    // initialize brute force index
    BFParams bf_params = {
        .type = params->primaryIndexParams->algoParams.cuvsIvfParams.type,
        .dim = params->primaryIndexParams->algoParams.cuvsIvfParams.dim,
        .metric = params->primaryIndexParams->algoParams.cuvsIvfParams.metric,
        .multi = params->primaryIndexParams->algoParams.cuvsIvfParams.multi,
        //.blockSize = params->primaryIndexParams->algoParams.cuvsIvfParams.blockSize
    };

    std::shared_ptr<VecSimAllocator> flat_allocator = VecSimAllocator::newVecsimAllocator();
    AbstractIndexInitParams abstractInitParams = {.allocator = flat_allocator,
                                                  .dim = bf_params.dim,
                                                  .vecType = bf_params.type,
                                                  .metric = bf_params.metric,
                                                  .blockSize = bf_params.blockSize,
                                                  .multi = bf_params.multi,
                                                  .logCtx = params->primaryIndexParams->logCtx};
    auto frontendIndex = static_cast<BruteForceIndex<DataType, DistType> *>(
        BruteForceFactory::NewIndex(&bf_params, abstractInitParams));

    // Create new tiered CuvsIVF index
    std::shared_ptr<VecSimAllocator> management_layer_allocator =
        VecSimAllocator::newVecsimAllocator();

    return new (management_layer_allocator) TieredCuvsIvfIndex<DataType, DistType>(
        cuvs_index, frontendIndex, *params, management_layer_allocator);
}

// The size estimation is the sum of the buffer (brute force) and main index initial sizes
// estimations, plus the tiered index class size. Note it does not include the size of internal
// containers such as the job queue, as those depend on the user implementation.
size_t EstimateInitialSize(const TieredIndexParams *params) {
    auto cuvs_ivf_params = params->primaryIndexParams->algoParams.cuvsIvfParams;

    // Add size estimation of VecSimTieredIndex sub indexes.
    size_t est = CuvsIvfFactory::EstimateInitialSize(&cuvs_ivf_params);

    // Management layer allocator overhead.
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();
    est += sizeof(VecSimAllocator) + allocations_overhead;

    // Size of the TieredCuvsIvfIndex struct.
    if (cuvs_ivf_params.type == VecSimType_FLOAT32) {
        est += sizeof(TieredCuvsIvfIndex<float, float>);
    } else if (cuvs_ivf_params.type == VecSimType_FLOAT64) {
        est += sizeof(TieredCuvsIvfIndex<double, double>);
    }

    return est;
}

}; // namespace TieredCuvsIvfFactory
