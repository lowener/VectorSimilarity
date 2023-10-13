#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <cuda_runtime.h>
#include <library_types.h>
#include "VecSim/vec_sim.h"
// For VecSimMetric, RaftIvfParams, labelType
#include "VecSim/vec_sim_common.h"
// For VecSimIndexAbstract
#include "VecSim/vec_sim_index.h"
#include "VecSim/query_result_definitions.h" // VecSimQueryResult VecSimQueryReply
#include "VecSim/memory/vecsim_malloc.h"

#include <raft/core/device_resources_manager.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/ivf_flat.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>

inline auto constexpr GetRaftDistanceType(VecSimMetric vsm) {
    auto result = raft::distance::DistanceType{};
    switch (vsm) {
    case VecSimMetric_L2:
        result = raft::distance::DistanceType::L2Expanded;
        break;
    case VecSimMetric_IP:
        result = raft::distance::DistanceType::InnerProduct;
        break;
    default:
        throw raft::exception("Metric not supported");
    }
    return result;
}

inline auto constexpr GetRaftCodebookKind(IVFPQCodebookKind vss_codebook) {
    auto result = raft::neighbors::ivf_pq::codebook_gen{};
    switch (vss_codebook) {
    case IVFPQCodebookKind_PerCluster:
        result = raft::neighbors::ivf_pq::codebook_gen::PER_CLUSTER;
        break;
    case IVFPQCodebookKind_PerSubspace:
        result = raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
        break;
    default:
        throw raft::exception("Unexpected IVFPQ codebook kind");
    }
    return result;
}

inline auto constexpr GetCudaType(CudaType vss_type) {
    auto result = cudaDataType_t{};
    switch (vss_type) {
    case CUDAType_R_32F:
        result = CUDA_R_32F;
        break;
    case CUDAType_R_16F:
        result = CUDA_R_16F;
        break;
    case CUDAType_R_8U:
        result = CUDA_R_8U;
        break;
    default:
        throw raft::exception("Unexpected CUDA type");
    }
    return result;
}

template <typename DataType, typename DistType = DataType>
struct RaftIVFIndex : public VecSimIndexAbstract<DistType> {
    using data_type = DataType;
    using dist_type = DistType;

private:
    // Allow either IVF-flat or IVFPQ parameters
    using build_params_t = std::variant<raft::neighbors::ivf_flat::index_params,
                                        raft::neighbors::ivf_pq::index_params>;
    using search_params_t = std::variant<raft::neighbors::ivf_flat::search_params,
                                         raft::neighbors::ivf_pq::search_params>;
    using internal_idx_t = std::uint32_t;
    using index_flat_t = raft::neighbors::ivf_flat::index<data_type, internal_idx_t>;
    using index_pq_t = raft::neighbors::ivf_pq::index<internal_idx_t>;
    using ann_index_t = std::variant<index_flat_t, index_pq_t>;

public:
    RaftIVFIndex(const RaftIvfParams *raftIvfParams, const AbstractIndexInitParams &commonParams)
        : VecSimIndexAbstract<dist_type>{commonParams},
          res_{raft::device_resources_manager::get_device_resources()},
          build_params_{raftIvfParams->usePQ ? build_params_t{std::in_place_index<1>}
                                             : build_params_t{std::in_place_index<0>}},
          search_params_{raftIvfParams->usePQ ? search_params_t{std::in_place_index<1>}
                                              : search_params_t{std::in_place_index<0>}},
          index_{std::nullopt} {
        std::visit(
            [raftIvfParams](auto &&inner) {
                inner.metric = GetRaftDistanceType(raftIvfParams->metric);
                inner.n_lists = raftIvfParams->nLists;
                inner.kmeans_n_iters = raftIvfParams->kmeans_nIters;
                inner.kmeans_trainset_fraction = raftIvfParams->kmeans_trainsetFraction;
                inner.conservative_memory_allocation = raftIvfParams->conservativeMemoryAllocation;
                if constexpr (std::is_same_v<decltype(inner),
                                             raft::neighbors::ivf_flat::index_params>) {
                    inner.adaptive_centers = raftIvfParams->adaptiveCenters;
                } else if constexpr (std::is_same_v<decltype(inner),
                                                    raft::neighbors::ivf_pq::index_params>) {
                    inner.pq_bits = raftIvfParams->pqBits;
                    inner.pq_dim = raftIvfParams->pqDim;
                    inner.codebook_kind = GetRaftCodebookKind(raftIvfParams->codebookKind);
                }
            },
            build_params_);
        std::visit(
            [raftIvfParams](auto &&inner) {
                inner.n_probes = raftIvfParams->nProbes;
                if constexpr (std::is_same_v<decltype(inner),
                                             raft::neighbors::ivf_pq::search_params>) {
                    inner.lut_dtype = GetCudaType(raftIvfParams->lutType);
                    inner.internal_distance_dtype =
                        GetCudaType(raftIvfParams->internalDistanceType);
                    inner.preferred_shmem_carvout = raftIvfParams->preferredShmemCarveout;
                }
            },
            search_params_);
    }
    int addVector(const void *vector_data, labelType label, void *auxiliaryCtx = nullptr) override {
        return addVectorBatch(vector_data, &label, 1, auxiliaryCtx);
    }
    int addVectorBatchAsync(const void *vector_data, labelType *label, size_t batch_size,
                            void *auxiliaryCtx = nullptr) {
        // Convert labels to internal data type
        auto label_original = std::vector<labelType>(label, label + batch_size);
        auto label_converted =
            std::vector<internal_idx_t>(label_original.begin(), label_original.end());
        // Allocate memory on device to hold vectors to be added
        auto vector_data_gpu =
            raft::make_device_matrix<data_type, internal_idx_t>(res_, batch_size, this->dim);
        // Allocate memory on device to hold vector labels
        auto label_gpu = raft::make_device_vector<internal_idx_t, internal_idx_t>(res_, batch_size);

        // Copy vector data to previously allocated device buffer
        raft::copy(vector_data_gpu.data_handle(), static_cast<DataType const *>(vector_data),
                   this->dim * batch_size, res_.get_stream());
        // Copy label data to previously allocated device buffer
        raft::copy(label_gpu.data_handle(), label_converted.data(), batch_size, res_.get_stream());

        if (std::holds_alternative<raft::neighbors::ivf_flat::index_params>(build_params_)) {
            if (!index_) {
                index_ = raft::neighbors::ivf_flat::build(
                    res_, std::get<raft::neighbors::ivf_flat::index_params>(build_params_),
                    raft::make_const_mdspan(vector_data_gpu.view()));
            }
            raft::neighbors::ivf_flat::extend(
                res_, raft::make_const_mdspan(vector_data_gpu.view()),
                std::make_optional(raft::make_const_mdspan(label_gpu.view())),
                std::get<index_flat_t>(*index_));
        } else {
            if (!index_) {
                index_ = raft::neighbors::ivf_pq::build(
                    res_, std::get<raft::neighbors::ivf_pq::index_params>(build_params_),
                    raft::make_const_mdspan(vector_data_gpu.view()));
            }
            raft::neighbors::ivf_pq::extend(
                res_, raft::make_const_mdspan(vector_data_gpu.view()),
                std::make_optional(raft::make_const_mdspan(label_gpu.view())),
                std::get<index_pq_t>(*index_));
        }

        return batch_size;
    }
    int addVectorBatch(const void *vector_data, labelType *label, size_t batch_size,
                       void *auxiliaryCtx = nullptr) {
        auto result = addVectorBatchAsync(vector_data, label, batch_size, auxiliaryCtx);
        // Ensure that above operation has executed on device before
        // returning from this function on host
        res_.sync_stream();
        return result;
    }
    int deleteVector(labelType label) override {
        assert(!"deleteVector not implemented");
        return 0;
    }
    double getDistanceFrom_Unsafe(labelType label, const void *vector_data) const override {
        assert(!"getDistanceFrom not implemented");
        return INVALID_SCORE;
    }
    size_t indexCapacity() const override {
        assert(!"indexCapacity not implemented");
        return 0;
    }
    // void increaseCapacity() override { assert(!"increaseCapacity not implemented"); }
    inline size_t indexLabelCount() const override {
        return this->indexSize(); // TODO: Return unique counts
    }
    VecSimQueryReply *topKQuery(const void *queryBlob, size_t k,
                                VecSimQueryParams *queryParams) const override {
        auto result_list = new VecSimQueryReply(this->allocator);
        auto nVectors = this->indexSize();
        if (nVectors == 0 || k == 0 || !index_.has_value()) {
            return result_list;
        }
        // Ensure we are not trying to retrieve more vectors than exist in the
        // index
        k = std::min(k, nVectors);
        // Allocate memory on device for search vector
        auto vector_data_gpu =
            raft::make_device_matrix<data_type, internal_idx_t>(res_, 1, this->dim);
        // Allocate memory on device for neighbor and distance results
        auto neighbors_gpu = raft::make_device_matrix<internal_idx_t, internal_idx_t>(res_, 1, k);
        auto distances_gpu = raft::make_device_matrix<dist_type, internal_idx_t>(res_, 1, k);
        // Copy query vector to device
        raft::copy(vector_data_gpu.data_handle(), static_cast<const data_type*>(queryBlob), this->dim,
                   res_.get_stream());

        // Perform correct search based on index type
        if (std::holds_alternative<index_flat_t>(*index_)) {
            raft::neighbors::ivf_flat::search<data_type, internal_idx_t>(
                res_, std::get<raft::neighbors::ivf_flat::search_params>(search_params_),
                std::get<index_flat_t>(*index_), raft::make_const_mdspan(vector_data_gpu.view()),
                neighbors_gpu.view(), distances_gpu.view());
            // TODO ADD STREAM MANAGER
        } else {
            raft::neighbors::ivf_pq::search<data_type, internal_idx_t>(
                res_, std::get<raft::neighbors::ivf_pq::search_params>(search_params_),
                std::get<index_pq_t>(*index_), raft::make_const_mdspan(vector_data_gpu.view()),
                neighbors_gpu.view(), distances_gpu.view());
            // TODO ADD STREAM MANAGER
        }

        // Allocate host buffers to hold returned results
        auto neighbors = vecsim_stl::vector<internal_idx_t>(k, this->allocator);
        auto distances = vecsim_stl::vector<dist_type>(k, this->allocator);
        // Copy data back from device to host
        raft::copy(neighbors.data(), neighbors_gpu.data_handle(), this->dim, res_.get_stream());
        raft::copy(distances.data(), distances_gpu.data_handle(), this->dim, res_.get_stream());

        // Ensure search is complete and data have been copied back before
        // building query result objects on host
        res_.sync_stream();

        result_list->results.resize(k);
        for (auto i = 0; i < k; ++i) {
            result_list->results[i].id = labelType{neighbors[i]};
            result_list->results[i].score = distances[i];
        }

        return result_list;
    }

    VecSimQueryReply *rangeQuery(const void *queryBlob, double radius,
                                 VecSimQueryParams *queryParams) const override {
        assert(!"RangeQuery not implemented");
    }
    VecSimInfoIterator *infoIterator() const override { assert(!"infoIterator not implemented"); }
    virtual VecSimBatchIterator *newBatchIterator(const void *queryBlob,
                                                  VecSimQueryParams *queryParams) const override {
        assert(!"newBatchIterator not implemented");
    }
    bool preferAdHocSearch(size_t subsetSize, size_t k, bool initial_check) const override {
        assert(!"preferAdHocSearch not implemented");
    }

    auto &get_resources() const { return res_; }

    auto nLists() {
        return std::visit([](auto &&params) { return params.n_list; }, build_params_);
    }

    size_t indexSize() const override {
        auto result = size_t{};
        if (index_) {
            result = std::visit([](auto &&index) { return index.size(); }, *index_);
        }
        return result;
    }
    VecSimIndexBasicInfo basicInfo() const override {
        VecSimIndexBasicInfo info = this->getBasicInfo();
        info.algo = VecSimAlgo_RaftIVF;
        info.isTiered = false;
        return info;
    }
    VecSimIndexInfo info() const override {
        VecSimIndexInfo info;
        info.commonInfo = this->getCommonInfo();
        info.raftIvfInfo.nLists = nLists();
        if (std::holds_alternative<raft::neighbors::ivf_pq::index_params>(build_params_)) {
            const auto build_params_pq =
                std::get<raft::neighbors::ivf_pq::index_params>(build_params_);
            info.raftIvfInfo.pqBits = build_params_pq.pq_bits;
            info.raftIvfInfo.pqDim = build_params_pq.pq_dim;
        }
        return info;
    }

private:
    // An object used to manage common device resources that may be
    // expensive to build but frequently accessed
    raft::device_resources res_;
    // Store build params to allow for index build on first batch
    // insertion
    build_params_t build_params_;
    // Store search params to use with each search after initializing in
    // constructor
    search_params_t search_params_;
    // Use a std::optional to allow building of the index on first batch
    // insertion
    std::optional<ann_index_t> index_;
};
