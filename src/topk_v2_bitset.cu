
#include "topk.h"
#include "thread_pool.h"

#include <cub/cub.cuh>
#include <cuda_fp16.hpp>
#include <chrono>
#include <numeric>
#include <cuda_pipeline.h>

#include <emmintrin.h>
#include <mmintrin.h>

#include "fast_topk.cuh"

typedef uint4 group_t;

constexpr static const int TOPK = 100;
constexpr static const int N_THREADS_IN_ONE_BLOCK = 512;
constexpr static const int MAX_DOC_SIZE = 128;

constexpr static const int max_batch = 4;
// constexpr static const int max_id = 50000;
constexpr static const int query_mask_size = 1568;  // 1568 * 32 > 50000
constexpr static const int default_sort_storage = 64 * 1024 * 1024;
constexpr static const int num_threads = 8;

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
        const uint16_t *docs, 
        const int *doc_lens,
        const size_t n_docs, 
        uint32_t *query,
        const uint16_t max_query_token,
        const int query_len,
        float *scores) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadid = threadIdx.x;

    __shared__ uint32_t query_mask[query_mask_size];

    #pragma unroll
    for (int l = threadid; l < query_mask_size; l += N_THREADS_IN_ONE_BLOCK) {
        query_mask[l] = __ldg(query + l);
    }
    __syncthreads();

    if (tid >= n_docs) {
        return;
    }

    int doc_id = tid;
    int doc_len = doc_lens[doc_id];
    int loop = (doc_len + 7) / 8;

    uint16_t tmp_score = 0;

    for (int i = 0; i < loop; ++i) {
        group_t loaded = ((group_t*)docs)[i * n_docs + doc_id];
        uint16_t* token = (uint16_t*)(&loaded);

        #pragma unroll
        for (auto j = 0; j < 8; ++j) {
            uint16_t tindex = token[j] >> 5;
            uint16_t tpos = token[j] & 31;

            tmp_score += (query_mask[tindex] >> tpos) & 0x01;
            // tmp_score += __popc(query_mask[tindex] & tmask);
        }

        if (token[7] >= max_query_token) {
            break;
        }
    }
    scores[doc_id] = 1.f * tmp_score / max(query_len, doc_len);
}

#define MYTIME

#ifdef MYTIME
struct Timer {
    const char* m_name;
    std::chrono::high_resolution_clock::time_point m_start;
    std::chrono::high_resolution_clock::time_point m_stop;

    Timer(const char* name) {
        m_name = name;
        m_start = std::chrono::high_resolution_clock::now();
    }

    void stop(const char* name = nullptr) {
        // CHECK_CUDA(cudaDeviceSynchronize());
        m_stop = std::chrono::high_resolution_clock::now();
        double cur_time = std::chrono::duration<double, std::milli>(m_stop-m_start).count();
        printf("==== %s: %.3fms\n", m_name, cur_time);

        m_name = name;
        m_start = std::chrono::high_resolution_clock::now();
    }
};
#else
struct Timer {
    Timer(const char* name) {}
    void stop(const char* name = nullptr) {}
};
#endif

struct MemsetTask : public Task {
    MemsetTask(uint16_t* ptr, size_t size)
        : m_ptr(ptr), m_size(size) 
    {}

    void run() override {
        memset(m_ptr, 0, m_size);
    }

    uint16_t* m_ptr = nullptr;
    size_t m_size = 0;
};

struct HostCopyTask : public Task {
    HostCopyTask(int start, int end, std::vector<int>& h_doc_lens_vec, uint16_t* h_docs, std::vector<std::vector<uint16_t>> & docs)
        : m_start(start), m_end(end), m_h_doc_lens(h_doc_lens_vec), m_h_docs(h_docs), m_docs(docs)
    {}

    void run() override {
        auto group_sz = sizeof(group_t) / sizeof(uint16_t);
        auto n_docs = m_docs.size();
        auto layer_0_stride = n_docs * group_sz;
        auto layer_1_stride = group_sz;
        for (int i = m_start; i < m_end; i++) {
            auto layer_1_offset = i;
            auto layer_1_offset_val = layer_1_offset * layer_1_stride;
            for (int j = 0; j < m_docs[i].size(); j++) {
                auto layer_0_offset = j / group_sz;
                auto layer_2_offset = j % group_sz;
                auto final_offset = layer_0_offset * layer_0_stride + layer_1_offset_val + layer_2_offset;
                m_h_docs[final_offset] = m_docs[i][j];
            }
            m_h_doc_lens[i] = m_docs[i].size();
        }
    }

    int m_start = 0;
    int m_end = 0;
    std::vector<int>& m_h_doc_lens;
    uint16_t* m_h_docs;
    std::vector<std::vector<uint16_t>> & m_docs;
};

struct TopkTask : public Task {

    TopkTask(int start, int end, std::vector<std::vector<uint16_t>> &querys,
            uint16_t *d_docs, int *d_doc_lens, int n_docs, std::vector<std::vector<int>> &indices)
        : m_start(start), m_end(end), m_querys(querys), m_d_docs(d_docs), m_d_doc_lens(d_doc_lens),
          m_n_docs(n_docs), m_indices(indices) {}

    void run() override {
        if (m_start >= m_end) {
            return;
        }

        cudaDeviceProp device_props;
        cudaGetDeviceProperties(&device_props, 0);
        cudaSetDevice(0);

        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        uint32_t* d_query;
        cudaMallocAsync(&d_query, sizeof(uint32_t) * query_mask_size, stream);
        float* d_scores;
        cudaMallocAsync(&d_scores, sizeof(float) * m_n_docs, stream);

        std::vector<int> indices(m_n_docs);
        for (int i = 0; i < m_n_docs; ++i) {
            indices[i] = i;
        }

        // int* s_indices;
        // cudaMallocHost(&s_indices, n_docs * sizeof(int));
        // float* scores;
        // cudaMallocHost(&scores, n_docs * sizeof(float));
        std::vector<int> s_indices(m_n_docs);
        std::vector<float> scores(m_n_docs);

        for (int i = m_start; i < m_end; ++i) {
            auto& query = m_querys[i];
            //init indices
            memcpy(s_indices.data(), indices.data(), indices.size() * sizeof(int));

            const size_t query_len = query.size();
            std::vector<uint32_t> query_mask(query_mask_size, 0u);
            for (auto& q : query) {
                int index = q / 32;
                int postion = q % 32;
                query_mask[index] |= ((1u) << postion);    
            }
            cudaMemcpyAsync(d_query, query_mask.data(), sizeof(uint32_t) * query_mask_size, cudaMemcpyHostToDevice, stream);

            // launch kernel
            int block = N_THREADS_IN_ONE_BLOCK;
            int grid = (m_n_docs + block - 1) / block;
            uint16_t max_query_token = query.back();

            docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block, 0, stream>>>(m_d_docs,
                m_d_doc_lens, m_n_docs, d_query, max_query_token, query_len, d_scores);

            // cudaDeviceSynchronize();
            cudaMemcpyAsync(scores.data(), d_scores, sizeof(float) * m_n_docs, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            // sort scores
            std::partial_sort(s_indices.begin(), s_indices.begin() + TOPK, s_indices.end(),
                            [&scores](const int& a, const int& b) {
                                if (scores[a] != scores[b]) {
                                    return scores[a] > scores[b];  // 按照分数降序排序
                                }
                                return a < b;  // 如果分数相同，按索引从小到大排序
                        });
            std::vector<int> s_ans(s_indices.begin(), s_indices.begin() + TOPK);
            m_indices[i] = std::move(s_ans);
            // cudaFree(d_query);
        }

        cudaFreeAsync(d_query, stream);
        cudaFreeAsync(d_scores, stream);
        cudaStreamDestroy(stream);
    }

    int m_start = 0;
    int m_end = 0;
    std::vector<std::vector<uint16_t>> & m_querys;
    uint16_t* m_d_docs = nullptr;
    int* m_d_doc_lens = nullptr;
    int m_n_docs = 0;
    std::vector<std::vector<int>> & m_indices;
};

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs,
    std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices //shape [querys.size(), TOPK]
    ) {

    auto n_docs = docs.size();
    std::vector<float> scores(n_docs);
    std::vector<int> s_indices(n_docs);

    float *d_scores = nullptr;
    uint16_t *d_docs = nullptr;
    uint32_t *d_query = nullptr;
    int *d_doc_lens = nullptr;

    ThreadPool pool;
    int num_threads = min(8, static_cast<int>(n_docs));
    pool.set_num_threads(num_threads);

Timer t("pre_process");

    // copy to device
    cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
    cudaMalloc(&d_scores, sizeof(float) * n_docs);
    cudaMalloc(&d_doc_lens, sizeof(int) * n_docs);

    uint16_t *h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];
#if 0
    memset(h_docs, 0, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
#else
    std::vector<Task*> tasks(num_threads, nullptr);
    size_t n_docs_per_threads = (n_docs + num_threads - 1) / num_threads;
    int offset = 0;
    for (int i = 0; i < num_threads; ++i) {
        int size = min(n_docs_per_threads, n_docs - offset) * sizeof(uint16_t) * MAX_DOC_SIZE;
        tasks[i] = new MemsetTask(h_docs + MAX_DOC_SIZE * offset, size);
        offset += n_docs_per_threads;
    }
    pool.run_task(tasks);
    pool.wait();
#endif

    std::vector<int> h_doc_lens_vec(n_docs);
#if 1
    std::vector<Task*> host_copy_tasks(num_threads, nullptr);
    // size_t n_docs_per_threads = (n_docs + num_threads - 1) / num_threads;
    offset = 0;
    for (int i = 0; i < num_threads; ++i) {
        int size = min(n_docs_per_threads, n_docs - offset);
        int end = offset + size;
        host_copy_tasks[i] = new HostCopyTask(offset, end, h_doc_lens_vec, h_docs, docs);
        offset += n_docs_per_threads;
    }
    pool.run_task(host_copy_tasks);
    pool.wait();
#else
    for (int i = 0; i < docs.size(); i++) {
        for (int j = 0; j < docs[i].size(); j++) {
            auto group_sz = sizeof(group_t) / sizeof(uint16_t);
            auto layer_0_stride = n_docs * group_sz;
            auto layer_0_offset = j / group_sz;
            auto layer_1_offset = i;
            auto layer_1_stride = group_sz;
            auto layer_2_offset = j % group_sz;
            auto final_offset = layer_0_offset * layer_0_stride + layer_1_offset * layer_1_stride + layer_2_offset;
            h_docs[final_offset] = docs[i][j];
        }
        h_doc_lens_vec[i] = docs[i].size();
    }
#endif

    cudaMemcpy(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_doc_lens, h_doc_lens_vec.data(), sizeof(int) * n_docs, cudaMemcpyHostToDevice);

    cudaDeviceProp device_props;
    cudaGetDeviceProperties(&device_props, 0);

    cudaSetDevice(0);

t.stop("topk");

#if 0
    for(auto& query : querys) {
        //init indices
        for (int i = 0; i < n_docs; ++i) {
            s_indices[i] = i;
        }

        const size_t query_len = query.size();
        cudaMalloc(&d_query, sizeof(uint32_t) * query_mask_size);
        std::vector<uint32_t> query_mask(query_mask_size, 0u);
        for (auto& q : query) {
            int index = q / 32;
            int postion = q % 32;
            query_mask[index] |= ((1u) << postion);    
        }
        cudaMemcpy(d_query, query_mask.data(), sizeof(uint32_t) * query_mask_size, cudaMemcpyHostToDevice);

        // launch kernel
        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (n_docs + block - 1) / block;
        uint16_t max_query_token = query.back();
        docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block>>>(d_docs,
            d_doc_lens, n_docs, d_query, max_query_token, query_len, d_scores);

        // cudaDeviceSynchronize();
        cudaMemcpy(scores.data(), d_scores, sizeof(float) * n_docs, cudaMemcpyDeviceToHost);

        // sort scores
        std::partial_sort(s_indices.begin(), s_indices.begin() + TOPK, s_indices.end(),
                        [&scores](const int& a, const int& b) {
                            if (scores[a] != scores[b]) {
                                return scores[a] > scores[b];  // 按照分数降序排序
                            }
                            return a < b;  // 如果分数相同，按索引从小到大排序
                    });
        std::vector<int> s_ans(s_indices.begin(), s_indices.begin() + TOPK);
        indices.push_back(s_ans);

        cudaFree(d_query);
    }
#else
    indices.resize(querys.size());
    std::vector<Task*> topk_tasks(num_threads, nullptr);
    int num_querys = querys.size();
    int n_query_per_threads = (num_querys + num_threads - 1) / num_threads;
    int start = 0;
    for (int i = 0; i < num_threads; ++i) {
        int size = min(n_query_per_threads, num_querys - start);
        int end = start + size;
        topk_tasks[i] = new TopkTask(start, end, querys, d_docs, d_doc_lens, n_docs, indices);
        start = end;
    }
    pool.run_task(topk_tasks);
    pool.wait();
#endif

t.stop();

    // deallocation
    cudaFree(d_docs);
    //cudaFree(d_query);
    cudaFree(d_scores);
    cudaFree(d_doc_lens);
    free(h_docs);
}