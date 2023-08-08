#pragma once

#include "VecSim/utils/vecsim_stl.h"
#include <raft/neighbors/cagra.cuh>
#include "ivf_index.cuh"

class RaftCAGRAIndex : public RaftIVFIndex {
public:
    using raftCAGRAIndex_t = raft::neighbors::experimental::cagra::index<DataType, std::int64_t>;
    RaftCAGRAIndex(const RaftCAGRAParams *params_cagra, std::shared_ptr<VecSimAllocator> allocator)
        : RaftIVFIndex(params_cagra, allocator), idToLabelLookup(allocator) {
        assert(params_cagra->intermediate_graph_degree > 0);
        assert(params_cagra->graph_degree > 0);
        raft::logger::get("raft").set_level(RAFT_LEVEL_ERROR);
        build_params_cagra_ = std::make_unique<raft::neighbors::experimental::cagra::index_params>();
        build_params_cagra_->metric = GetRaftDistanceType(params_cagra->metric);
        build_params_cagra_->intermediate_graph_degree = params_cagra->intermediate_graph_degree;
        build_params_cagra_->graph_degree = params_cagra->graph_degree;
        search_params_cagra_ = std::make_unique<raft::neighbors::experimental::cagra::search_params>();
        search_params_cagra_->itopk_size = params_cagra->itopk_size;
    }
    int addVectorBatchGpuBuffer(const void *vector_data, std::int64_t *, size_t batch_size,
                                bool overwrite_allowed) override {
        auto vector_data_gpu = raft::make_device_matrix_view<const DataType, std::int64_t>(
            (const DataType *)vector_data, batch_size, this->dim);

        if (!cagra_index_) {
            cagra_index_ = std::make_unique<raftCAGRAIndex_t>(
                raft::neighbors::experimental::cagra::build<DataType, std::int64_t>(*res_, *build_params_cagra_,
                                                                                    vector_data_gpu));
        }
        // Add labels to map
        return batch_size;
    }


    int addVectorBatchGpuBufferCpuLabel(const void *vector_data, std::int64_t *labels, size_t batch_size,
                                        bool overwrite_allowed=true) {
        std::int64_t id = idToLabelLookup.size();
        auto result = addVectorBatchGpuBuffer(vector_data, nullptr, batch_size, overwrite_allowed);

        for (size_t i = 0; i < batch_size; ++i) {
            idToLabelLookup.emplace(id, labels[i]);
            id++;
        }
        res_->sync_stream();
        return result;
    }

    /*
    int addVectorBatchCpuBuffer(const void *vector_data, std::int64_t *labels, size_t batch_size,
                                bool overwrite_allowed) {
        auto vector_data_cpu = raft::make_host_matrix_view<const DataType, std::int64_t>(
            (const DataType *)vector_data, batch_size, this->dim);

        if (!cagra_index_) {
            cagra_index_ = std::make_unique<raftCAGRAIndex_t>(
                raft::neighbors::experimental::cagra::build<DataType, std::int64_t>(*res_, *build_params_cagra_,
                                                                                    vector_data_cpu));
        }
        std::int64_t id = indexSize();
        for (size_t i = 0; i < batch_size; ++i) {
            idToLabelLookup.emplace(id, labels[i]);
            id++;
        }
        return batch_size;
    }*/

    int addVectorBatch(const void *vector_data, labelType *labels, size_t batch_size,
                        bool overwrite_allowed) override {
        auto vector_data_gpu =
            raft::make_device_matrix<DataType, std::int64_t>(*res_, batch_size, this->dim);
        auto label_original = std::vector<labelType>(labels, labels + batch_size);
        auto label_converted = std::vector<std::int64_t>(label_original.begin(), label_original.end());
        //auto label_gpu = raft::make_device_vector<std::int64_t, std::int64_t>(*res_, batch_size);

        RAFT_CUDA_TRY(cudaMemcpyAsync(vector_data_gpu.data_handle(), (DataType *)vector_data,
                this->dim * batch_size * sizeof(float), cudaMemcpyDefault,
                res_->get_stream()));
        /*RAFT_CUDA_TRY(cudaMemcpyAsync(label_gpu.data_handle(), label_converted.data(),
                batch_size * sizeof(std::int64_t), cudaMemcpyDefault,
                res_->get_stream()));*/

        this->addVectorBatchGpuBufferCpuLabel(vector_data_gpu.data_handle(), label_converted.data(),
                                              batch_size, overwrite_allowed);
        res_->sync_stream();
        return batch_size;
    }

    void search(const void *vector_data, void *neighbors, void *distances, size_t batch_size,
                size_t k) override {
        auto vector_data_gpu = raft::make_device_matrix_view<const DataType, std::int64_t>(
            (const DataType *)vector_data, batch_size, this->dim);
        auto neighbors_gpu = raft::make_device_matrix_view<std::int64_t, std::int64_t>(
            (std::int64_t *)neighbors, batch_size, k);
        auto distances_gpu =
            raft::make_device_matrix_view<float, std::int64_t>((float *)distances, batch_size, k);
        raft::neighbors::experimental::cagra::search<DataType, std::int64_t>(*res_, *search_params_cagra_, *cagra_index_,
                                                                             vector_data_gpu,
                                                                             neighbors_gpu, distances_gpu);
    }

    VecSimQueryResult_List topKQuery(const void *queryBlob, size_t k,
                                     VecSimQueryParams *queryParams) override{
        VecSimQueryResult_List result_list = {0};
        auto nVectors = this->indexSize();
        if (nVectors == 0) {
            result_list.results = array_new<VecSimQueryResult>(0);
            return result_list;
        }
        if (k > nVectors)
            k = nVectors; // Safeguard K
        auto vector_data_gpu =
            raft::make_device_matrix<DataType, std::uint32_t>(*res_, queryParams->batchSize, this->dim);
        auto neighbors_gpu =
            raft::make_device_matrix<std::int64_t, std::uint32_t>(*res_, queryParams->batchSize, k);
        auto distances_gpu =
            raft::make_device_matrix<float, std::uint32_t>(*res_, queryParams->batchSize, k);
        RAFT_CUDA_TRY(cudaMemcpyAsync(vector_data_gpu.data_handle(), (const DataType *)queryBlob,
                                    this->dim * queryParams->batchSize * sizeof(float),
                                    cudaMemcpyDefault, res_->get_stream()));

        this->search(vector_data_gpu.data_handle(), neighbors_gpu.data_handle(),
                    distances_gpu.data_handle(), 1, k);

        auto result_size = queryParams->batchSize * k;
        auto *neighbors = array_new_len<std::int64_t>(result_size, result_size);
        auto *distances = array_new_len<float>(result_size, result_size);
        RAFT_CUDA_TRY(cudaMemcpyAsync(neighbors, neighbors_gpu.data_handle(),
                                    result_size * sizeof(std::int64_t), cudaMemcpyDefault,
                                    res_->get_stream()));
        RAFT_CUDA_TRY(cudaMemcpyAsync(distances, distances_gpu.data_handle(),
                                    result_size * sizeof(float), cudaMemcpyDefault,
                                    res_->get_stream()));
        res_->sync_stream();

        result_list.results = array_new_len<VecSimQueryResult>(k, k);
        for (size_t i = 0; i < k; ++i) {
            VecSimQueryResult_SetId(result_list.results[i], idToLabelLookup.at(neighbors[i]));
            VecSimQueryResult_SetScore(result_list.results[i], distances[i]);
        }
        array_free(neighbors);
        array_free(distances);
        return result_list;
    }

    VecSimIndexInfo info() const override {
        VecSimIndexInfo info;
        //TODO Implement this
        return info;
    }
    size_t nLists() override { return size_t{1}; }
    size_t indexSize() const override { return cagra_index_.get() == nullptr ? 0 : cagra_index_->size(); }

protected:
    std::unique_ptr<raftCAGRAIndex_t> cagra_index_;
    raft::distance::DistanceType metric_;
    // Build params are kept as class member because the build step on Raft side happens on
    // the first vector insertion
    std::unique_ptr<raft::neighbors::experimental::cagra::index_params> build_params_cagra_;
    std::unique_ptr<raft::neighbors::experimental::cagra::search_params> search_params_cagra_;
private:
    vecsim_stl::unordered_map<std::int64_t, size_t> idToLabelLookup;
};
