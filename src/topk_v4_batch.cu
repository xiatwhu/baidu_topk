
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

__device__ __forceinline__
uint32_t getBitfield(uint32_t val, int pos, int len) {
    uint32_t ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
    return ret;
}

__device__ __forceinline__
uint64_t getBitfield64(uint64_t val, int pos, int len) {
    uint64_t ret;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
    return ret;
}

template<int N=4>
#if __CUDA_ARCH__ == 860
__launch_bounds__(N_THREADS_IN_ONE_BLOCK, 3)
#elif __CUDA_ARCH__ == 800
__launch_bounds__(N_THREADS_IN_ONE_BLOCK, 4)
#endif
void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
        const uint16_t* docs,
        const uint16_t* doc_lens,
        const size_t n_docs, 
        const uint32_t* query,
        const uint16_t* query_len,
        const uint16_t max_query_token,
        int16_t *scores) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    __shared__ uint32_t query_mask[N * query_mask_size];
    // __shared__ group_t doc_cache[N_THREADS_IN_ONE_BLOCK];

    int threadid = threadIdx.x;

    #pragma unroll
    for (int l = threadid; l < N * query_mask_size; l += N_THREADS_IN_ONE_BLOCK) {
        query_mask[l] = __ldg(query + l);
    }
    __syncthreads();

    for (int doc_id = tid; doc_id < n_docs; doc_id += stride) {
        int doc_len = doc_lens[doc_id];
        int loop = (doc_len + 7) / 8;

        int tmp_score[N] = {0};

        for (int i = 0; i < loop; ++i) {

            group_t loaded = ((group_t*)docs)[i * n_docs + doc_id];
            uint16_t* token = (uint16_t*)(&loaded);

            #pragma unroll
            for (auto j = 0; j < 8; ++j) {
                uint16_t tindex = token[j] >> 5;
                uint16_t tpos = token[j] & 31;

                #pragma unroll
                for (auto k = 0; k < N; ++k) {
                    tmp_score[k] += (query_mask[k * query_mask_size + tindex] >> tpos) & 0x01;
                }
            }

            if (token[7] >= max_query_token) {
                break;
            }
        }

        for (auto i = 0; i < N; ++i) {
            scores[i * n_docs + doc_id] = static_cast<int16_t>(
                1.f * 128 * 128 * tmp_score[i] / max(query_len[i], doc_len));
        }
    }
}

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
        std::vector<std::vector<uint16_t>> &querys,
        std::vector<std::vector<int>> &indices,
        int start,
        int batch,
        cudaStream_t stream) {

    int cur_batch = batch;
    memset(h_query, 0, sizeof(uint32_t) * cur_batch * query_mask_size);
    uint16_t max_query_token = 0;
    for (int j = 0; j < cur_batch; ++j) {
        auto& query = querys[start + j];
        h_query_len[j] = query.size();
        for (auto& q : query) {
            uint16_t index = q >> 5;
            uint16_t postion = q & 31;
            h_query[j * query_mask_size + index] |= ((1u) << postion);
            // h_query[cur_batch * index + j] |= ((1u) << postion);
        }
        max_query_token = std::max(max_query_token, query.back());
    }

    CHECK_CUDA(cudaMemcpyAsync(d_query, h_query,
            cur_batch * query_mask_size * sizeof(uint32_t),
            cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_query_len, h_query_len,
            cur_batch * sizeof(uint16_t),
            cudaMemcpyHostToDevice, stream));

    int block = N_THREADS_IN_ONE_BLOCK;
    int grid = n_docs_pad / 4096;

    if (cur_batch == 4) {
        docQueryScoringCoalescedMemoryAccessSampleKernel<4><<<grid, block, 0, stream>>>(
                d_docs, d_doc_lens, n_docs_pad, d_query, d_query_len, max_query_token, d_scores);
        CHECK_CUDA(cudaGetLastError());
    } else if (cur_batch == 3) {
        docQueryScoringCoalescedMemoryAccessSampleKernel<3><<<grid, block, 0, stream>>>(
                d_docs, d_doc_lens, n_docs_pad, d_query, d_query_len, max_query_token, d_scores);
        CHECK_CUDA(cudaGetLastError());
    } else if (cur_batch == 2) {
        docQueryScoringCoalescedMemoryAccessSampleKernel<2><<<grid, block, 0, stream>>>(
                d_docs, d_doc_lens, n_docs_pad, d_query, d_query_len, max_query_token, d_scores);
        CHECK_CUDA(cudaGetLastError());
    } else if (cur_batch == 1) {
        docQueryScoringCoalescedMemoryAccessSampleKernel<1><<<grid, block, 0, stream>>>(
                d_docs, d_doc_lens, n_docs_pad, d_query, d_query_len, max_query_token, d_scores);
        CHECK_CUDA(cudaGetLastError());
    }

    launch_gather_topk_kernel(
            d_scores, d_topk, (int8_t*)d_temp_storage, TOPK, cur_batch, n_docs_pad, stream);
    CHECK_CUDA(cudaMemcpyAsync(h_topk, d_topk, cur_batch * TOPK * sizeof(Pair), cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int j = 0; j < cur_batch; ++j) {
        Pair* cur_topk = h_topk + j * TOPK;
        std::vector<int> s_ans(TOPK);
        std::sort(cur_topk, cur_topk + TOPK,
                [](const Pair& a, const Pair& b) {
                    if (a.score != b.score) {
                        return a.score > b.score;
                    }
                    return a.index < b.index;
                });
        for (int k = 0; k < TOPK; ++k) {
            s_ans[k] = cur_topk[k].index;
        }
        indices[start + j] = std::move(s_ans);
    }
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

t.stop("cuda_malloc_host");
        auto ret = posix_memalign((void**)(&h_mem), 256, sizeof(uint16_t) * 128 * n_docs);
        (void)(ret);

        int8_t* h_mem_pool = h_mem;
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
        }
t.stop("thread_pool");
        pool.wait();
t.stop();
    }

    void init_pinned(int n_docs, int num_threads) {
Timer t("cuda_malloc_pinned");
        // 计算需要分配的 pinned 内存大小
        // 由于 cudaMallocHost 分配大块内存时特别耗时, 所以 h_docs 空间使用 malloc 分配
        size_t bytes = 0u;
        
        for (int i = 0; i < num_threads; ++i) {
            bytes += align_bytes(sizeof(Pair) * max_batch * TOPK);                  // h_topk
            bytes += align_bytes(sizeof(uint32_t) * max_batch * query_mask_size);   // h_query
            bytes += align_bytes(sizeof(uint32_t) * max_batch);                     // h_query_len
            // bytes += align_bytes(sizeof(uint16_t) * 1 * n_docs);                    // h_scores
        }
        CHECK_CUDA(cudaMallocHost(&h_pinned_mem, bytes));

        int8_t* h_pinned_mem_pool = h_pinned_mem;

        for (int i = 0; i < num_threads; ++i) {
            ThreadContext& ctx = thread_contexts[i];

            ctx.h_topk = reinterpret_cast<Pair*>(h_pinned_mem_pool);
            h_pinned_mem_pool += align_bytes(sizeof(Pair) * max_batch * TOPK);
            ctx.h_query = reinterpret_cast<uint32_t*>(h_pinned_mem_pool);
            h_pinned_mem_pool += align_bytes(sizeof(uint32_t) * max_batch * query_mask_size);
            ctx.h_query_len = reinterpret_cast<uint16_t*>(h_pinned_mem_pool);
            h_pinned_mem_pool += align_bytes(sizeof(uint16_t) * max_batch);
            // ctx.h_scores = reinterpret_cast<int16_t*>(h_pinned_mem_pool);
            // h_pinned_mem_pool += align_bytes(sizeof(int16_t) * 1 * n_docs);
        }
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
        int16_t* h_scores = nullptr;        // [max_batch * n_docs]
    };

    // update
    uint16_t* h_docs = nullptr;             // [16, n_docs, 8]
    uint16_t* d_docs = nullptr;             // [16, n_docs, 8]
    uint16_t* d_doc_lens = nullptr;         // [n_docs]
    std::vector<ThreadContext> thread_contexts;
};

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

        // [16, n_docs_pad, 8]
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
          querys(querys_),
          query_idx(query_idx_),
          d_docs(d_docs_),
          d_doc_lens(d_doc_lens_),
          n_docs_pad(n_docs_pad_),
          indices(indices_) {}

    void run() override {
        if (start >= end) {
            return;
        }

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
        for (int i = max_batch * id; i < total_items; i += max_batch * world) {
            int cur_batch = std::min<int>(total_items - i, max_batch);

            search_topk(n_docs_pad,
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
                        querys,
                        indices,
                        i,
                        cur_batch,
                        stream);
        }
    }

    Context& ctx;
    int id = 0;
    int world = 0;
    int start = 0;
    int end = 0;
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

    ctx.init_pinned(n_docs_pad, num_threads);

    // 线程池在处理 d_docs 时, 主线程处理其他耗时的操作
t.stop("pre_init_cuda");

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
    // if (false) {
    if (true) {
        // 单线程处理 query
        Context::ThreadContext& tctx = ctx.thread_contexts[0];
        int16_t* d_scores = tctx.d_scores;
        Pair* d_topk = tctx.d_topk;
        uint32_t* d_query = tctx.d_query;
        uint16_t* d_query_len = tctx.d_query_len;
        void* d_temp_storage = tctx.d_temp_storage;

        Pair* h_topk = tctx.h_topk;
        uint32_t* h_query = tctx.h_query;
        uint16_t* h_query_len = tctx.h_query_len;

        for (int i = 0; i < querys.size(); i += max_batch) {
            int cur_batch = std::min<int>(querys.size() - i, max_batch);

// Timer tt("query");
            search_topk(n_docs_pad,
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
                        querys,
                        indices,
                        i,
                        cur_batch,
                        stream);
// tt.stop();
        }
    } else {
        // 多线程处理 query
        std::vector<Task*> topk_tasks(num_threads);
        int num_querys = querys.size();
        int n_query_per_threads = (num_querys + num_threads - 1) / num_threads;
        int start = 0;
        for (int i = 0; i < num_threads; ++i) {
            int size = min(n_query_per_threads, num_querys - start);
            int end = start + size;
            topk_tasks[i] = new TopkTask(
                    ctx, i, num_threads, start, end, querys, query_idx, d_docs, d_doc_lens, n_docs_pad, indices);
            start = end;
        }

        pool.run_task(topk_tasks);
        pool.wait();
    }
t.stop();
}