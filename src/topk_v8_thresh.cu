
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
constexpr static const int max_id = 50000;
constexpr static const int query_mask_size = 1568;  // 1568 * 32 > 50000
constexpr static const int default_sort_storage = 64 * 1024 * 1024;
constexpr static const int num_threads = 8;

template <int N>
struct PackData {};

template <>
struct PackData<1> {
    using dtype = uint32_t;
};

template <>
struct PackData<2> {
    using dtype = uint2;
};

template <>
struct PackData<3> {
    using dtype = uint3;
};

template <>
struct PackData<4> {
    using dtype = uint4;
};

#if __CUDA_ARCH__ == 860
__launch_bounds__(N_THREADS_IN_ONE_BLOCK, 3)
#elif __CUDA_ARCH__ == 800
__launch_bounds__(N_THREADS_IN_ONE_BLOCK, 4)
#endif
void __global__ docFirstKernel(
        const uint16_t* docs,
        const uint16_t* doc_lens,
        const int doc_offset,
        const int doc_num, 
        const int n_docs,
        const uint32_t* query,
        const int query_len,
        const uint16_t max_query_token,
        int16_t *scores) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadid = threadIdx.x;

    __shared__ uint32_t query_mask[query_mask_size];

    #pragma unroll
    for (int l = threadid; l < query_mask_size; l += N_THREADS_IN_ONE_BLOCK) {
        query_mask[l] = __ldg(query + l);
    }
    __syncthreads();

    if (tid >= doc_num) {
        return;
    }

    int doc_id = doc_offset + tid;
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
            uint32_t tmask = (1u) << tpos;

            tmp_score += __popc(query_mask[tindex] & tmask);
        }

        if (token[7] >= max_query_token) {
            break;
        }
    }
    scores[tid] = static_cast<int16_t>(1.f * 128 * 128 * tmp_score / max(query_len, doc_len));
}

#if __CUDA_ARCH__ == 860
__launch_bounds__(N_THREADS_IN_ONE_BLOCK, 3)
#elif __CUDA_ARCH__ == 800
__launch_bounds__(N_THREADS_IN_ONE_BLOCK, 4)
#endif
void __global__ docIterKernel(
        const uint16_t* docs,
        const uint16_t* doc_lens,
        const int doc_offset1,
        const int doc_num1,
        const int doc_offset2,
        const int doc_num2,
        const int n_docs,
        const uint32_t* query,
        const int query_len,
        const uint16_t max_query_token,
        const float thresh,
        int16_t *scores) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadid = threadIdx.x;

    __shared__ uint32_t query_mask[query_mask_size];

    #pragma unroll
    for (int l = threadid; l < query_mask_size; l += N_THREADS_IN_ONE_BLOCK) {
        query_mask[l] = __ldg(query + l);
    }
    __syncthreads();

    if (tid >= doc_num1 + doc_num2) {
        return;
    }

    int doc_id = tid < doc_num1 ? (doc_offset1 + tid) : (doc_offset2 + tid - doc_num1);
    int doc_len = doc_lens[doc_id];
    uint32_t tmp_score_thresh = static_cast<uint32_t>(max(doc_len, query_len) * thresh);
    int loop = (doc_len + 7) / 8;

    uint32_t tmp_score = 0;

    for (int i = 0; i < loop; ++i) {
        group_t loaded = ((group_t*)docs)[i * n_docs + doc_id];
        uint16_t* token = (uint16_t*)(&loaded);

        #pragma unroll
        for (auto j = 0; j < 8; ++j) {
            uint16_t tindex = token[j] >> 5;
            uint16_t tpos = token[j] & 31;
            uint32_t tmask = (1u) << tpos;

            // tmp_score += (query_mask[tindex] >> tpos) & 0x01;
            tmp_score += __popc(query_mask[tindex] & tmask);
        }
        if (token[7] > max_query_token 
                || tmp_score + (doc_len - (i + 1) * 8) < tmp_score_thresh) {
            break;
        }
    }
    scores[tid] = static_cast<int16_t>(1.f * 128 * 128 * tmp_score / max(query_len, doc_len));
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

struct Context {
    Context() = default;

    void init(int n_docs, int num_threads) {
 Timer t("init");
        pool.set_num_threads(num_threads);
        
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        thread_contexts.resize(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            CHECK_CUDA(cudaStreamCreateWithFlags(&thread_contexts[i].stream, cudaStreamNonBlocking));
        }

t.stop("cuda_malloc_device");
        // 计算好需要分配的显存大小
        size_t bytes = 0u;
        bytes += align_bytes(sizeof(uint16_t) * 128 * n_docs);                      // d_docs
        bytes += align_bytes(sizeof(uint16_t) * n_docs);                            // d_doc_lens
        for (int i = 0; i < num_threads; ++i) {
            bytes += align_bytes(sizeof(half) * max_batch * n_docs);                // d_scores
            bytes += align_bytes(sizeof(Pair) * max_batch * TOPK);                  // d_topk
            bytes += align_bytes(sizeof(uint32_t) * max_batch * query_mask_size);   // d_query
            bytes += align_bytes(sizeof(uint32_t) * max_batch);                     // d_query_len
            bytes += align_bytes(default_sort_storage);                             // d_temp_storage
        }
        CHECK_CUDA(cudaMalloc(&d_mem, bytes));

t.stop("cuda_malloc_pinned");
        // 计算需要分配的 pinned 内存大小
        // 由于 cudaMallocHost 分配大块内存时特别耗时, 所以 h_docs 空间使用 malloc 分配
        bytes = 0u;
        // bytes += align_bytes(sizeof(uint16_t) * 128 * n_docs);                      // h_docs
        
        for (int i = 0; i < num_threads; ++i) {
            bytes += align_bytes(sizeof(Pair) * max_batch * TOPK);                  // h_topk
            bytes += align_bytes(sizeof(uint32_t) * max_batch * query_mask_size);   // h_query
            bytes += align_bytes(sizeof(uint32_t) * max_batch);                     // h_query_len
        }
        CHECK_CUDA(cudaMallocHost(&h_pinned_mem, bytes));

t.stop("cuda_malloc_host");
        auto ret = posix_memalign((void**)(&h_mem), 256, sizeof(uint16_t) * 128 * n_docs);
        (void)(ret);

        int8_t* h_mem_pool = h_mem;
        int8_t* h_pinned_mem_pool = h_pinned_mem;
        int8_t* d_mem_pool = d_mem;
        // 初始化指针
        h_docs = reinterpret_cast<uint16_t*>(h_mem_pool);
        h_mem_pool += align_bytes(sizeof(uint16_t) * 128 * n_docs);

        d_docs = reinterpret_cast<uint16_t*>(d_mem_pool);
        d_mem_pool += align_bytes(sizeof(uint16_t) * 128 * n_docs);
        d_doc_lens = reinterpret_cast<uint16_t*>(d_mem_pool);
        d_mem_pool += align_bytes(sizeof(uint16_t) * n_docs);

        for (int i = 0; i < num_threads; ++i) {
            ThreadContext& ctx = thread_contexts[i];
            ctx.d_scores = reinterpret_cast<int16_t*>(d_mem_pool);
            d_mem_pool += align_bytes(sizeof(int16_t) * max_batch * n_docs);
            ctx.d_topk = reinterpret_cast<Pair*>(d_mem_pool);
            d_mem_pool += align_bytes(sizeof(Pair) * max_batch * TOPK);
            ctx.d_query = reinterpret_cast<uint32_t*>(d_mem_pool);
            d_mem_pool += align_bytes(sizeof(uint32_t) * max_batch * query_mask_size);
            ctx.d_query_len = reinterpret_cast<uint16_t*>(d_mem_pool);
            d_mem_pool += align_bytes(sizeof(uint16_t) * max_batch);
            ctx.d_temp_storage = reinterpret_cast<void*>(d_mem_pool);
            d_mem_pool += align_bytes(default_sort_storage);

            ctx.h_topk = reinterpret_cast<Pair*>(h_pinned_mem_pool);
            h_pinned_mem_pool += align_bytes(sizeof(Pair) * max_batch * TOPK);
            ctx.h_query = reinterpret_cast<uint32_t*>(h_pinned_mem_pool);
            h_pinned_mem_pool += align_bytes(sizeof(uint32_t) * max_batch * query_mask_size);
            ctx.h_query_len = reinterpret_cast<uint16_t*>(h_pinned_mem_pool);
            h_pinned_mem_pool += align_bytes(sizeof(uint16_t) * max_batch);
        }
t.stop("thread_pool");
        pool.wait();
t.stop();       
    }

    void clear() {
        if (d_mem) {
            cudaFree(d_mem);
        }
        if (h_mem) {
            // cudaFreeHost(h_mem);
            free(h_mem);
        }

        if (h_pinned_mem) {
            cudaFreeHost(h_pinned_mem);
        }

        cudaStreamDestroy(stream);
        for (int i = 0; i < num_threads; ++i) {
            cudaStreamDestroy(thread_contexts[i].stream);
        }
    }

    // init
    int8_t* d_mem = nullptr;
    int8_t* h_mem = nullptr;
    int8_t* h_pinned_mem = nullptr;
    cudaStream_t stream;
    cudaDeviceProp prop;
    ThreadPool pool;

    struct ThreadContext {
        cudaStream_t stream;
        int16_t* d_scores = nullptr;        // [max_batch * n_docs]
        Pair* d_topk = nullptr;             // [max_batch * TOPK]
        uint32_t* d_query = nullptr;        // [max_batch * query_mask_size]
        uint16_t* d_query_len = nullptr;    // [max_batch]
        void* d_temp_storage = nullptr;     // [64 * 1024 * 1024]

        Pair* h_topk = nullptr;             // [max_batch * TOPK]
        uint32_t* h_query = nullptr;        // [max_batch * query_mask_size]
        uint16_t* h_query_len = nullptr;    // [max_batch]
    };

    uint16_t* h_docs = nullptr;             // [16, n_docs, 8]
    uint16_t* d_docs = nullptr;             // [16, n_docs, 8]
    uint16_t* d_doc_lens = nullptr;         // [n_docs]
    std::vector<ThreadContext> thread_contexts;
};

void search_topk(
            int n_docs_pad,
            uint16_t* d_docs,
            uint16_t* d_doc_lens,
            int16_t* d_scores,
            Pair* d_topk,
            uint32_t* d_query,
            uint16_t* d_query_len,
            void* d_temp_storage,
            Pair* h_topk,
            uint32_t* h_query,
            uint16_t* h_query_len,
            std::vector<int>& doc_stat_offset,
            std::vector<std::vector<uint16_t>> &querys,
            std::vector<std::vector<int>> &indices,
            int i,
            cudaStream_t stream) {
    auto& query = querys[i];

    memset(h_query, 0, sizeof(uint32_t) * query_mask_size);
    uint16_t max_query_token = query.back();
    for (auto& q : query) {
        uint16_t index = q >> 5;
        uint16_t postion = q & 31;
        h_query[index] |= ((1u) << postion);
    }

    CHECK_CUDA(cudaMemcpyAsync(d_query, h_query,
            query_mask_size * sizeof(uint32_t),
            cudaMemcpyHostToDevice, stream));
    int query_len = query.size();

    // 如果 query 长度为 0 直接返回前 TOPK 个 doc
    if (query_len == 0) {
        std::vector<int> s_ans(TOPK);
        std::iota(s_ans.begin(), s_ans.end(), 0);
        indices[i] = std::move(s_ans);
        return;
    }

    int window = 2;
    float cur_score_thresh = 0.f;
    int cur_doc_len_start = 0;
    int cur_doc_len_end = 0;
    Pair* cur_topk = h_topk;
    {
        // step1, 先查询长度在 [query_len - 2, query_len + 4] 内的所有 query
        int doc_len_end = std::min(129, query_len + 1 + 2 * window);
        int doc_len_start = std::max(0, doc_len_end - (1 + 3 * window));
        doc_len_end = std::min(129, doc_len_start + (1 + 3 * window));

        int doc_num = doc_stat_offset[doc_len_end] - doc_stat_offset[doc_len_start];
        while (doc_num < TOPK) {
            // 如果所选范围内 doc 数不足 TOPK 个则继续扩大范围
            doc_len_start = std::max(0, doc_len_start - 1);
            doc_len_end = std::min(129, doc_len_end + 1);
            doc_num = doc_stat_offset[doc_len_end] - doc_stat_offset[doc_len_start];
        }
        int doc_offset = doc_stat_offset[doc_len_start];

        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (doc_num + N_THREADS_IN_ONE_BLOCK - 1) / N_THREADS_IN_ONE_BLOCK;

        docFirstKernel<<<grid, block, 0, stream>>>(
                d_docs, d_doc_lens, doc_offset, doc_num, n_docs_pad, d_query, query_len, max_query_token, d_scores);
        CHECK_CUDA(cudaGetLastError());

        launch_gather_topk_kernel(
                d_scores, d_topk, (int8_t*)d_temp_storage, TOPK, 1, doc_num, stream);
        CHECK_CUDA(cudaMemcpyAsync(h_topk, d_topk, TOPK * sizeof(Pair), cudaMemcpyDeviceToHost, stream));

        CHECK_CUDA(cudaStreamSynchronize(stream));
        // Pair* cur_topk = h_topk;

        for (int k = 0; k < TOPK; ++k) {
            cur_topk[k].index += doc_offset;
        }
        
        std::sort(cur_topk, cur_topk + TOPK,
                [](const Pair& a, const Pair& b) {
                    if (a.score != b.score) {
                        return a.score > b.score;
                    }
                    return a.index < b.index;
                });

        float score_thresh = cur_topk[TOPK - 1].score / 128 / 128;
        // printf("[F] %d %d %f\n", doc_len_start, doc_len_end, score_thresh);
        bool finished = score_thresh > 0.f
                && static_cast<int>(query_len * score_thresh) >= doc_len_start
                && std::min(128, static_cast<int>(query_len / score_thresh)) < doc_len_end;
        finished = finished || (doc_len_start == 0 && doc_len_end == 129);
        if (finished) {
            // 已经找到正确的 TOPK, 提前返回
            std::vector<int> s_ans(TOPK);
            for (int k = 0; k < TOPK; ++k) {
                s_ans[k] = cur_topk[k].index;
            }
            indices[i] = std::move(s_ans);
            return;
        }

        cur_doc_len_end = doc_len_end;
        cur_doc_len_start = doc_len_start;
        cur_score_thresh = score_thresh;
        window *= 2;
    }

    // step2, 逐步扩大搜索范围搜索
    while (true) {
        int doc_len_start = 0;
        int doc_len_end = 0;
        if (cur_score_thresh == 0.f) {
            doc_len_start = 0;
            doc_len_end = 129;
        } else {
            int min_start = static_cast<int>(query_len * cur_score_thresh);
            int max_end = std::min(129, static_cast<int>(query_len / cur_score_thresh + 1));
            doc_len_start = std::max(min_start, cur_doc_len_start - window);
            doc_len_start = std::min(doc_len_start, cur_doc_len_start);
            doc_len_end = std::min(max_end, cur_doc_len_end + window * 2);
            doc_len_end = std::max(doc_len_end, cur_doc_len_end);

            // 对 doc_len_end == 128 的特殊处理, 防止因没有 doc 长度为 128 而陷入循环
            if (doc_len_end == 128 && (doc_stat_offset[129] - doc_stat_offset[128] == 0)) {
                doc_len_end = 129;
            }
        }

        // 搜索范围为 [doc_len_start, cur_doc_len_start) 及 [cur_doc_len_end, doc_len_end)
        int doc_num1 = doc_stat_offset[cur_doc_len_start] - doc_stat_offset[doc_len_start];
        int doc_offset1 = doc_stat_offset[doc_len_start];
        int doc_num2 = doc_stat_offset[doc_len_end] - doc_stat_offset[cur_doc_len_end];
        int doc_offset2 = doc_stat_offset[cur_doc_len_end];

        assert(doc_num1 + doc_num2 > 0);

        int doc_num = doc_num1 + doc_num2;

        int block = N_THREADS_IN_ONE_BLOCK;
        int grid = (doc_num + N_THREADS_IN_ONE_BLOCK - 1) / N_THREADS_IN_ONE_BLOCK;

        docIterKernel<<<grid, block, 0, stream>>>(
                d_docs, d_doc_lens,
                doc_offset1, doc_num1,
                doc_offset2, doc_num2,
                n_docs_pad,
                d_query, query_len, max_query_token,
                cur_score_thresh, d_scores);
        CHECK_CUDA(cudaGetLastError());

        int new_topk = std::min(doc_num, TOPK);
        launch_gather_topk_kernel(
                d_scores, d_topk, (int8_t*)d_temp_storage, new_topk, 1, doc_num, stream);
        CHECK_CUDA(cudaMemcpyAsync(h_topk + TOPK, d_topk, new_topk * sizeof(Pair), cudaMemcpyDeviceToHost, stream));

        CHECK_CUDA(cudaStreamSynchronize(stream));
        Pair* topk = h_topk + TOPK;

        for (int k = 0; k < new_topk; ++k) {
            if (topk[k].index < doc_num1) {
                topk[k].index += doc_offset1;
            } else {
                topk[k].index += (doc_offset2 - doc_num1);
            }
        }
        
        std::sort(topk, topk + new_topk,
                [](const Pair& a, const Pair& b) {
                    if (a.score != b.score) {
                        return a.score > b.score;
                    }
                    return a.index < b.index;
                });

        // 和当前的 TOPK 合并
        Pair* temp = h_topk + TOPK * 2;
        int pre_index = 0;
        int new_index = 0;
        for (int k = 0; k < TOPK; ++k) {
            if (new_index == new_topk
                || cur_topk[pre_index].score > topk[new_index].score
                || (cur_topk[pre_index].score == topk[new_index].score
                    && cur_topk[pre_index].index < topk[new_index].index)) {
                temp[k].score = cur_topk[pre_index].score;
                temp[k].index = cur_topk[pre_index].index;
                ++pre_index;
            } else {
                temp[k].score = topk[new_index].score;
                temp[k].index = topk[new_index].index;
                ++new_index;
            }
        }
        memcpy(cur_topk, temp, sizeof(Pair) * TOPK);

        float score_thresh = cur_topk[TOPK - 1].score / 128 / 128;
        bool finished = score_thresh > 0.f
                && static_cast<int>(query_len * score_thresh) >= doc_len_start
                && std::min(128, static_cast<int>(query_len / score_thresh)) < doc_len_end;
        finished = finished || (doc_len_start == 0 && doc_len_end == 129);
        if (finished) {
            // 已经找到正确的 TOPK, 提前返回
            std::vector<int> s_ans(TOPK);
            for (int k = 0; k < TOPK; ++k) {
                s_ans[k] = cur_topk[k].index;
            }
            indices[i] = std::move(s_ans);
            return;
        }

        cur_doc_len_end = doc_len_end;
        cur_doc_len_start = doc_len_start;
        cur_score_thresh = score_thresh;

        // 下一次搜索时扩大搜索窗口, 继续搜索
        window *= 2;
    }
}

struct HostCopyTask : public Task {
    HostCopyTask(
            Context& ctx_,
            int id_,
            int world_,
            int start_,
            int end_,
            int n_docs_pad_,
            uint16_t* h_docs_,
            std::vector<std::vector<uint16_t>> & docs_)
        : ctx(ctx_),
          id(id_),
          world(world_),
          start(start_),
          end(end_),
          n_docs_pad(n_docs_pad_),
          h_docs(h_docs_),
          docs(docs_) {}

    void run() override {

Timer t("host_copy");
        auto group_sz = sizeof(group_t) / sizeof(uint16_t);
        auto layer_0_stride = n_docs_pad * group_sz;
        auto layer_1_stride = group_sz;
        int max_len = 0;
        for (int i = start; i < end; i++) {
            auto layer_1_offset = i;

            int doc_len = docs[i].size();
            max_len = std::max(doc_len, max_len);

            int n = doc_len / 8;
            int leftover = doc_len % 8;
            uint16_t * ptr = docs[i].data();

            int offset = layer_1_offset * layer_1_stride;
            for (int j = 0; j < n; ++j) {
                __m128i a = _mm_loadu_si128((__m128i*)(ptr));
                _mm_store_si128((__m128i*)(h_docs + offset), a);
                ptr += 8;
                offset += layer_0_stride;
            }
            if (leftover) {
                // 此处补 0 避免了对整个 h_docs 进行置 0 操作
                alignas(16) int16_t data[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                for (int j = 0; j < leftover; ++j) {
                    data[j] = ptr[j];
                }
                __m128i a = _mm_load_si128((__m128i*)(data));
                _mm_store_si128((__m128i*)(h_docs + offset), a);
            }
        }
t.stop("device_copy");

        // [16, n_doc_pad, 8]
        uint16_t* d_docs = ctx.d_docs;
        cudaStream_t stream = ctx.thread_contexts[id].stream;
        // 并不需要拷贝所有的 16 个 group 的数据, 可以只拷贝有效的数值, 在 kernel 中也只会读取有效的部分数据
        int loop = (max_len + 7) / 8;
        for (int i = 0; i < loop; ++i) {
            CHECK_CUDA(cudaMemcpyAsync(d_docs + i * n_docs_pad * 8 + start * 8,
                    h_docs + i * n_docs_pad * 8 + start * 8,
                    (end - start) * 8 * sizeof(uint16_t),
                    cudaMemcpyHostToDevice, stream));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
t.stop();
    }

    Context& ctx;
    int id = 0;
    int world = 0;
    int start = 0;
    int end = 0;
    int n_docs_pad = 0;
    uint16_t* h_docs;
    std::vector<std::vector<uint16_t>> & docs;
};

struct TopkTask : public Task {

    TopkTask(
            Context& ctx_,
            int id_,
            int world_,
            int start_,
            int end_,
            std::vector<int>& doc_stat_offset_,
            std::vector<std::vector<uint16_t>> &querys_,
            std::vector<int> &query_idx_,
            uint16_t *d_docs_,
            uint16_t *d_doc_lens_,
            int n_docs_pad_,
            std::vector<std::vector<int>> &indices_)
        : ctx(ctx_),
          id(id_),
          world(world_),
          start(start_),
          end(end_),
          doc_stat_offset(doc_stat_offset_),
          querys(querys_),
          query_idx(query_idx_),
          d_docs(d_docs_),
          d_doc_lens(d_doc_lens_),
          n_docs_pad(n_docs_pad_),
          indices(indices_) {}

    void run() override {
        // if (start >= end) {
        //     return;
        // }

        Context::ThreadContext& tctx = ctx.thread_contexts[id];
        cudaStream_t stream = tctx.stream;

        int16_t* d_scores = tctx.d_scores;
        Pair* d_topk = tctx.d_topk;
        uint32_t* d_query = tctx.d_query;
        uint16_t* d_query_len = tctx.d_query_len;
        void* d_temp_storage = tctx.d_temp_storage;

        Pair* h_topk = tctx.h_topk;
        uint32_t* h_query = tctx.h_query;
        uint16_t* h_query_len = tctx.h_query_len;

        int total_items = querys.size();
        // for (int i = max_batch * id; i < total_items; i += max_batch * world) {
        //     int cur_batch = std::min<int>(total_items - i, max_batch);

        for (int i = id; i < total_items; i += world) {
            search_topk(
                n_docs_pad,
                d_docs,
                d_doc_lens,
                d_scores,
                d_topk,
                d_query,
                d_query_len,
                d_temp_storage,
                h_topk,
                h_query,
                h_query_len,
                doc_stat_offset,
                querys,
                indices,
                i,
                stream);
        }
    }

    Context& ctx;
    int id = 0;
    int world = 0;
    int start = 0;
    int end = 0;
    std::vector<int>& doc_stat_offset;
    std::vector<std::vector<uint16_t>> & querys;
    std::vector<int> & query_idx;
    uint16_t* d_docs = nullptr;
    uint16_t* d_doc_lens = nullptr;
    int n_docs_pad = 0;
    std::vector<std::vector<int>> & indices;
};

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs,
    std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices //shape [querys.size(), TOPK]
    ) {

Timer t("pre_malloc_host");
    size_t n_docs = docs.size();
    size_t n_docs_pad = (n_docs + 4095) / 4096 * 4096;

    // 分配资源，包括显存、内存、流、线程池等
    Context ctx;
    ctx.init(n_docs_pad, num_threads);

    uint16_t* h_docs = ctx.h_docs;

t.stop("pre_thread_pool");
    ThreadPool& pool = ctx.pool;

    // 使用多线程来将分散的 docs 拷贝到连续的 h_docs 内存中
    // 每个线程会将自己处理的内存拷贝到 d_docs 中
    // 这样可以避免主线程需要拷贝一大块空间到显存
    std::vector<Task*> tasks(num_threads, nullptr);
    size_t n_docs_per_threads = (n_docs_pad + num_threads - 1) / num_threads;
    int offset = 0;
    for (int i = 0; i < num_threads; ++i) {
        int size = min(n_docs_per_threads, n_docs_pad - offset);
        int end = offset + size;
        tasks[i] = new HostCopyTask(ctx, i, num_threads, offset, end, n_docs_pad, h_docs, docs);
        offset += n_docs_per_threads;
    }
    pool.run_task(tasks);

    // 线程池在处理 d_docs 时, 主线程处理其他耗时的操作
t.stop("pre_query_stat");
    int doc_stat[129] = {0};
    for (int i = 0; i < lens.size(); ++i) {
        doc_stat[lens[i]]++;
    }
    std::vector<int> doc_stat_offset(130, 0);
    for (int i = 0; i < 129; ++i) {
        doc_stat_offset[i + 1] = doc_stat_offset[i] + doc_stat[i];
    }

    std::vector<int> query_idx(querys.size());
    std::iota(query_idx.begin(), query_idx.end(), 0);
    std::sort(query_idx.begin(), query_idx.end(),
            [&querys](int a, int b) {
                return querys[a].back() < querys[b].back();
            });

t.stop("pre_malloc_device");
    cudaStream_t stream = ctx.stream;
    uint16_t* d_docs = ctx.d_docs;
    uint16_t* d_doc_lens = ctx.d_doc_lens;
    CHECK_CUDA(cudaMemcpyAsync(d_doc_lens, lens.data(), sizeof(uint16_t) * n_docs,
            cudaMemcpyHostToDevice, stream));
    if (n_docs != n_docs_pad) {
        CHECK_CUDA(cudaMemsetAsync(d_doc_lens + n_docs, 0,
                (n_docs_pad - n_docs) * sizeof(uint16_t), stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    indices.resize(querys.size());

t.stop("pre_memcpy_device");
    // 等待线程池完成 docs-> h_docs -> d_docs 的任务
    pool.wait();

t.stop("topk");
    if (false) {
    // if (true) {
        Context::ThreadContext& tctx = ctx.thread_contexts[0];
        int16_t* d_scores = tctx.d_scores;
        Pair* d_topk = tctx.d_topk;
        uint32_t* d_query = tctx.d_query;
        uint16_t* d_query_len = tctx.d_query_len;
        void* d_temp_storage = tctx.d_temp_storage;

        Pair* h_topk = tctx.h_topk;
        uint32_t* h_query = tctx.h_query;
        uint16_t* h_query_len = tctx.h_query_len;

        indices.resize(querys.size());

        for (int i = 0; i < querys.size(); ++i) {
// Timer tt("query");
            search_topk(
                n_docs_pad,
                d_docs,
                d_doc_lens,
                d_scores,
                d_topk,
                d_query,
                d_query_len,
                d_temp_storage,
                h_topk,
                h_query,
                h_query_len,
                doc_stat_offset,
                querys,
                indices,
                i,
                stream);
// tt.stop();
        }
    } else {
        indices.resize(querys.size());
        std::vector<Task*> topk_tasks(num_threads);
        int num_querys = querys.size();
        int n_query_per_threads = (num_querys + num_threads - 1) / num_threads;
        int start = 0;
        for (int i = 0; i < num_threads; ++i) {
            int size = min(n_query_per_threads, num_querys - start);
            int end = start + size;
            topk_tasks[i] = new TopkTask(
                    ctx, i, num_threads, start, end, doc_stat_offset,
                    querys, query_idx, d_docs, d_doc_lens, n_docs_pad, indices);
            start = end;
        }
        pool.run_task(topk_tasks);
        pool.wait();
    }
t.stop();
}