#include "VecSim/index_factories/brute_force_factory.h"
#include "VecSim/algorithms/cuvs_ivf/ivf.cuh"

namespace CuvsIvfFactory {

static AbstractIndexInitParams NewAbstractInitParams(const VecSimParams *params) {

    const CuvsIvfParams *cuvsIvfParams = &params->algoParams.cuvsIvfParams;
    AbstractIndexInitParams abstractInitParams = {
        .allocator = VecSimAllocator::newVecsimAllocator(),
        .dim = cuvsIvfParams->dim,
        .vecType = cuvsIvfParams->type,
        .metric = cuvsIvfParams->metric,
        //.multi = cuvsIvfParams->multi,
        //.logCtx = params->logCtx
    };
    return abstractInitParams;
}

VecSimIndex *NewIndex(const CuvsIvfParams *cuvsIvfParams,
                      const AbstractIndexInitParams &abstractInitParams) {
    assert(cuvsIvfParams->type == VecSimType_FLOAT32 && "Invalid IVF data type algorithm");
    if (cuvsIvfParams->type == VecSimType_FLOAT32) {
        return new (abstractInitParams.allocator)
            CuvsIvfIndex<float, float>(cuvsIvfParams, abstractInitParams);
    }

    // If we got here something is wrong.
    return NULL;
}

VecSimIndex *NewIndex(const VecSimParams *params) {
    const CuvsIvfParams *cuvsIvfParams = &params->algoParams.cuvsIvfParams;
    AbstractIndexInitParams abstractInitParams = NewAbstractInitParams(params);
    return NewIndex(cuvsIvfParams, NewAbstractInitParams(params));
}

VecSimIndex *NewIndex(const CuvsIvfParams *cuvsIvfParams) {
    VecSimParams params = {.algoParams{.cuvsIvfParams = CuvsIvfParams{*cuvsIvfParams}}};
    return NewIndex(&params);
}

size_t EstimateInitialSize(const CuvsIvfParams *cuvsIvfParams) {
    size_t allocations_overhead = VecSimAllocator::getAllocationOverheadSize();

    // Constant part (not effected by parameters).
    size_t est = sizeof(VecSimAllocator) + allocations_overhead;
    est += sizeof(CuvsIvfIndex<float, float>); // Object size
    if (!cuvsIvfParams->usePQ) {
        // Size of each cluster data
        est += cuvsIvfParams->nLists *
               sizeof(cuvs::neighbors::ivf_flat::list_data<float, std::int64_t>);
        // Vector of shared ptr to cluster
        est += cuvsIvfParams->nLists *
               sizeof(std::shared_ptr<cuvs::neighbors::ivf_flat::list_data<float, std::int64_t>>);
    } else {
        // Size of each cluster data
        est += cuvsIvfParams->nLists * sizeof(cuvs::neighbors::ivf_pq::list_data<std::int64_t>);
        // accum_sorted_sizes_ Array
        est += cuvsIvfParams->nLists * sizeof(std::int64_t);
        // vector of shared ptr to cluster
        est += cuvsIvfParams->nLists *
               sizeof(std::shared_ptr<cuvs::neighbors::ivf_pq::list_data<std::int64_t>>);
    }
    return est;
}

size_t EstimateElementSize(const CuvsIvfParams *cuvsIvfParams) {
    // Those elements are stored only on GPU.
    size_t est = 0;
    if (!cuvsIvfParams->usePQ) {
        // Size of vec + size of label.
        est += cuvsIvfParams->dim * VecSimType_sizeof(cuvsIvfParams->type) + sizeof(labelType);
    } else {
        size_t pq_dim = cuvsIvfParams->pqDim;
        if (pq_dim == 0) // Estimation.
            pq_dim = cuvsIvfParams->dim >= 128 ? cuvsIvfParams->dim / 2 : cuvsIvfParams->dim;
        // Size of vec after compression + size of label
        est += cuvsIvfParams->pqBits * pq_dim + sizeof(labelType);
    }
    return est;
}
}; // namespace CuvsIvfFactory
