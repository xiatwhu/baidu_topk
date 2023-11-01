#include <cub/cub.cuh>
#include <cub/version.cuh>
#include <cuda_fp16.hpp>

#if CUB_VERSION < 101500
#include "cub_scan_by_key.cuh"
#endif

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at file %s line %d with error: %s (%d)\n",     \
               __FILE__, __LINE__, cudaGetErrorString(status), status);        \
        return;                                                                \
    }                                                                          \
}

struct Pair {
    float score;
    int index;
};

inline size_t align_bytes(size_t a) {
    return (a + 255) / 256 * 256;
}

namespace mbtopk {

constexpr int BLOCK_THREADS = 256;

// Over what radix we are selecting values
constexpr int RADIX_BITS = 8;
constexpr int RADIX_DIGITS = 1 << RADIX_BITS; // 2 ^ RADIX_BITS
constexpr int RADIX_MASK = (RADIX_DIGITS - 1);
static_assert(RADIX_DIGITS <= BLOCK_THREADS, "radixFindKthValues kernel requires RADIX_DIGITS <= BLOCK_THREADS");
constexpr int MIN_ITEMS_PER_THREAD = 4;
constexpr int MAX_ITEMS_PER_THREAD = 64;


template <typename T>
struct Bitfield {};

template <>
struct Bitfield<unsigned int> {
    static __device__ __host__ __forceinline__
    unsigned int getBitfield(unsigned int val, int pos, int len) {
#if !defined(__CUDA_ARCH__)
        pos &= 0xff;
        len &= 0xff;

        unsigned int m = (1u << len) - 1u;
        return (val >> pos) & m;
#else
        unsigned int ret;
        asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
        return ret;
#endif
    }

    static __device__ __host__ __forceinline__
    unsigned int setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
#if !defined(__CUDA_ARCH__)
        pos &= 0xff;
        len &= 0xff;

        unsigned int m = (1u << len) - 1u;
        toInsert &= m;
        toInsert <<= pos;
        m <<= pos;

        return (val & ~m) | toInsert;
#else
        unsigned int ret;
        asm("bfi.b32 %0, %1, %2, %3, %4;" :
            "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
        return ret;
#endif
    }
};

template <typename T>
struct TopKTypeConfig {};

template <>
struct TopKTypeConfig<int16_t> {
  typedef uint32_t RadixType;

  static inline __device__ RadixType convert(int16_t v) {
    static_assert(sizeof(short) == 2, "");
    return 32768u + v;
  }

  static inline __device__ int16_t deconvert(RadixType v) {
    return v - 32768;
  }
};

template <>
struct TopKTypeConfig<float> {
    typedef unsigned int RadixType;

    static inline __device__ RadixType convert(float v) {
        RadixType x = __float_as_int(v);
        RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

        return (x ^ mask);
    }

    static inline __device__ float deconvert(RadixType v) {
        RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

        return __int_as_float(v ^ mask);
    }
};

template <>
struct TopKTypeConfig<half> {
    typedef uint32_t RadixType;

    static inline __device__ RadixType convert(half v) {
        RadixType x = __half_as_ushort(v);
        RadixType mask = (x & 0x00008000) ? 0x0000ffff : 0x00008000;
        return x ^ mask;
    }

    static inline __device__ half deconvert(RadixType v) {
        RadixType mask = (v & 0x00008000) ? 0x00008000 : 0x0000ffff;
        return __ushort_as_half(v ^ mask);
    }
};

template <typename T>
__global__ void fill(T* x, T value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < size; i += gridDim.x * blockDim.x) {
        x[i] = value;
    }
}

template <typename T, int Dim>
__launch_bounds__(BLOCK_THREADS)
__global__ void radixFindKthValues(
        T* in,
        uint32_t slice_size,
        uint32_t* ks_to_find,   // size: batch
        uint32_t batch,
        int current_bit,
        int items_per_thread,
        uint32_t blocks_per_slice,
        uint32_t desiredMask,
        // outputs
        uint32_t* semaphores,   // size: batch
        uint32_t* desires,       // size: batch
        short* counts,          // size: batch * blocks_per_slice * radix_digits
        T* kthValues) {

    int items_per_block = items_per_thread * BLOCK_THREADS;
    int tidx = threadIdx.x;
    uint32_t block_idx = blockIdx.x;
    uint32_t slice_idx = block_idx / blocks_per_slice;
    uint32_t blk_idx_in_slice = block_idx % blocks_per_slice;

    if (slice_idx >= batch) {
        return;
    }

    uint32_t desired = desires[slice_idx];
    uint32_t k_to_find = ks_to_find[slice_idx];
    int slice_start_index = slice_idx * slice_size;

    T* data = in + slice_start_index;

    typedef cub::BlockScan<uint32_t, BLOCK_THREADS> BlockScan;
    
    union __align__(16) TempStorage {
        uint32_t digit_counters[RADIX_DIGITS];
        uint32_t digit_count_cumsum[RADIX_DIGITS];  // only used if this it the last block for this slice
        typename BlockScan::TempStorage scan_storage;
    };
    __shared__ TempStorage temp_storage;

    // fill digit_counters with zeros
    if (tidx < RADIX_DIGITS) {
        temp_storage.digit_counters[tidx] = 0;
    }
    __syncthreads();

    items_per_thread = (blk_idx_in_slice + 1 < blocks_per_slice)
        ? items_per_thread 
        : (slice_size - blk_idx_in_slice * items_per_block + BLOCK_THREADS - 1) / BLOCK_THREADS;

    // collect digit counts and store in shared memory
    for (int i = 0; i < items_per_thread; ++i) {
        int idx = blk_idx_in_slice * items_per_block + i * BLOCK_THREADS + tidx;
        if (idx < slice_size) {
            uint32_t val = TopKTypeConfig<T>::convert(__ldg(&data[idx]));
            bool has_val = ((val & desiredMask) == (desired & desiredMask));
            uint32_t digit = Bitfield<uint32_t>::getBitfield(val, current_bit, RADIX_BITS);
            if (has_val) {
                atomicAdd(&temp_storage.digit_counters[digit], 1);
            }
        }
    }

    __syncthreads();

    // load digit counter to register, one digit per thread
    static_assert(RADIX_DIGITS <= BLOCK_THREADS, "this kernel requires RADIX_DIGITS <= BLOCK_THREADS");
    uint32_t digit_count = 0;
    if (tidx < RADIX_DIGITS) {
        digit_count = temp_storage.digit_counters[tidx];
    }

    // We always write out counts regardless if blocks_per_slice == 1 because
    // it will be used to compute offsets for `gatherTopK`.
    if (tidx < RADIX_DIGITS) {
        counts[block_idx * RADIX_DIGITS + tidx] = digit_count;
    }

    // if blocks_per_slice == 1, there is no need to do cross-block reduction
    // in this case we use counts saved at registers directly
    if (blocks_per_slice > 1) {
        __threadfence();    // make sure writes are globally visible
        __syncthreads();    // make sure all writes are finished before update semaphores
    }

    // the last block of each slice accumulates counters from multiple blocks and updates desired and ks_to_find
    __shared__ bool s_is_last_block_done;

    if (tidx == 0) {
        if (blocks_per_slice == 1) {
            s_is_last_block_done = true;
        } else {
            uint32_t blocks_finished_old = atomicAdd(&semaphores[slice_idx], 1);
            s_is_last_block_done = (blocks_finished_old == blocks_per_slice - 1);
        }
    }
    __syncthreads();

    if (!s_is_last_block_done) {
        return;
    }

    if (tidx < RADIX_DIGITS && blocks_per_slice > 1) {
        digit_count = 0;
        for (int blk = 0; blk < blocks_per_slice; ++blk) {
            digit_count += counts[(slice_idx * blocks_per_slice + blk) * RADIX_DIGITS + tidx];
        }
    }

    // compute the block-wide inclusive prefix sum
    uint32_t digit_count_cumsum;
    BlockScan(temp_storage.scan_storage).InclusiveSum(digit_count, digit_count_cumsum);
    __syncthreads();

    // every thread also need the perfix_sum of it's left value for comparison, so save a copy in shared mem
    if (tidx < RADIX_DIGITS) {
        temp_storage.digit_count_cumsum[tidx] = digit_count_cumsum;
    }
    __syncthreads();

    if (tidx < RADIX_DIGITS) {
        uint32_t digit_count_cumsum_left = (tidx == 0) ? 0 : temp_storage.digit_count_cumsum[tidx - 1];

        // if not the last pass: update desired and ks_to_find
        // if last pass: write out the kth value
        if (digit_count_cumsum_left < k_to_find && k_to_find <= digit_count_cumsum) {
            desired = Bitfield<uint32_t>::setBitfield(desired, tidx, current_bit, RADIX_BITS);
            desires[slice_idx] = desired;
            if (current_bit > 0) {
                ks_to_find[slice_idx] = k_to_find - digit_count_cumsum_left;
            } else {
                kthValues[slice_idx] = TopKTypeConfig<T>::deconvert(desired);
            }
        }
    }

    // reset semaphores for the next pass
    if (tidx == 0) {
        semaphores[slice_idx] = 0;
    }
}

// Assumption: k can not be larger than UINT32_MAX
__launch_bounds__(RADIX_DIGITS)  // one thread per digit
__global__ void computeBlockwiseWithinKCounts(
        uint32_t* desires,              // size: num_slices
        short* counts,                  // size: num_slices * blocks_per_slice * radix_digits
        uint32_t blocks_per_slice,
        int current_bit,
        // outputs:
        uint32_t* withinKCounts,        // size: num_slices * blocks_per_slice == num_blocks
        uint32_t num_blocks
        ) {
    // This kernel should be launched with the same number of blocks as the `radixFindKthValues` kernel.
    int tidx = threadIdx.x;
    uint32_t block_idx = blockIdx.x;
    uint32_t slice_idx = block_idx / blocks_per_slice;

    if (block_idx >= num_blocks) {
        return;
    }

    uint32_t desired = __ldg(desires + slice_idx);
    uint32_t desired_digit = Bitfield<uint32_t>::getBitfield(desired, current_bit, RADIX_BITS);

    bool warp_is_active, thread_is_active;
    int warp = tidx / 32;
    int end_of_warp = warp * 32 + 32 - 1;
    warp_is_active = end_of_warp > desired_digit;
    thread_is_active = tidx > desired_digit;

    uint32_t count = 0;
    if (warp_is_active) {
        if (thread_is_active) {
            count = __ldg(counts + block_idx * RADIX_DIGITS + tidx);
        }
        for (int offset = 32 / 2; offset > 0; offset /= 2) {
            count += __shfl_down_sync(0xffffffff, count, offset, 32);
        }
    }

    constexpr int num_warps = RADIX_DIGITS / 32;
    __shared__ uint32_t warp_counts[num_warps];
    if (tidx % 32 == 0) {
        warp_counts[warp] = count;
    }
    __syncthreads();
    static_assert(RADIX_DIGITS < 32 * 32, "Assuming only 1 warp is needed for final reduction");
    if (warp != 0) {
        return;
    }
    count = 0;
    if (tidx < num_warps) {
        count = warp_counts[tidx];
    }
    for (int offset = num_warps / 2; offset > 0; offset /= 2) {
        count += __shfl_down_sync(0xffffffff, count, offset, 32);
    }
    if (tidx == 0) {
        withinKCounts[block_idx] += count;
    }
}

// Assumption: slice_size can not be larger than UINT32_MAX
__global__ void computeBlockwiseKthCounts(
    uint32_t* desires,              // size: num_slices
    short* counts,                  // size: num_slices * blocks_per_slice * radix_digits
    uint32_t num_blocks,            // the number of blocks used by `radixFindKthValues` kernel
    uint32_t blocks_per_slice,
    // outputs:
    uint32_t* kthCounts             // size: num_slices * blocks_per_slice == num_blocks
) {
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            idx < num_blocks;
            idx += blockDim.x * gridDim.x) {
        uint32_t slice_idx = idx / blocks_per_slice;
        uint32_t desired = __ldg(desires + slice_idx);
        uint32_t desired_digit = Bitfield<uint32_t>::getBitfield(desired, 0, RADIX_BITS);
        kthCounts[idx] = __ldg(counts + idx * RADIX_DIGITS + desired_digit);
    }
}

template <typename T, int Dim=2>
__launch_bounds__(BLOCK_THREADS)
__global__ void gatherTopK(T* input,
                           int inputSliceSize,
                           int outputSliceSize, // aka `k`
                           uint32_t numInputSlices,
                           Pair* topK,
                           uint32_t items_per_thread,
                           uint32_t blocks_per_slice,
                           T *kthValues,
                           uint32_t* withinKCounts,
                           uint32_t* kthCounts,
                           uint32_t num_blocks) {

    uint32_t items_per_block = items_per_thread * BLOCK_THREADS;
    uint32_t tidx = threadIdx.x;
    uint32_t block_idx = blockIdx.x;

    // The grid is computed from `getGridFromTiles`, when there are lots of
    // elements, we will use both blockIdx.x and blockIdx.y, and maybe blockIdx.z
    // when this is the case, the number of blocks that we are launching can be
    // more than the number of blocks we need. So we need to check the range of
    // `block_idx`.
    if (block_idx >= num_blocks) {
        return;
    }

    uint32_t slice_idx = block_idx / blocks_per_slice;
    uint32_t blk_idx_in_slice = block_idx % blocks_per_slice;

    items_per_thread = (blk_idx_in_slice + 1 < blocks_per_slice)
        ? items_per_thread
        : (inputSliceSize - blk_idx_in_slice * items_per_block + BLOCK_THREADS - 1) / BLOCK_THREADS;

    // Find the start offset for our slice
    int sliceStartIndex = slice_idx * inputSliceSize;
    int topKSliceStartIndex = slice_idx * outputSliceSize;

    T* inputSliceStart = input + sliceStartIndex;
    Pair* topKSliceStart = topK + topKSliceStartIndex;

    // Find the k-th highest element in our input
    T kthValue = kthValues[slice_idx];
    const auto kthValueConverted = TopKTypeConfig<T>::convert(kthValue);

    // Find the start index in output tensor of this block
    uint32_t startWithinK = 0;
    if (blk_idx_in_slice > 0) {
        startWithinK = withinKCounts[block_idx - 1];
    }
    uint32_t startKth = withinKCounts[slice_idx * blocks_per_slice + blocks_per_slice - 1];
    if (blk_idx_in_slice > 0) {
        startKth += kthCounts[block_idx - 1];
    }

    // Read input, select topk out and write
    typedef cub::BlockScan<uint32_t, BLOCK_THREADS> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    for (int i = 0; i < items_per_thread; ++i) {
        // Find the start offset for this slice
        int idx = blk_idx_in_slice * items_per_block + i * BLOCK_THREADS + tidx;
        T val;
        int withinK = 0;
        int kth = 0;
        if (idx < inputSliceSize) {
            val = __ldg(inputSliceStart + idx);
            const auto valConverted = TopKTypeConfig<T>::convert(val);
            withinK = valConverted > kthValueConverted;
            kth = (valConverted == kthValueConverted);
        }

        uint32_t withinKIndex;
        uint32_t numWithinK;
        BlockScan(temp_storage).ExclusiveSum(withinK, withinKIndex, numWithinK);
        __syncthreads();
        if (withinK) {
            uint32_t offset = withinKIndex + startWithinK;
            topKSliceStart[offset] = {(float)val, idx};
        }
        startWithinK += numWithinK;

        if (startKth < outputSliceSize) {
            uint32_t kthIndex;
            uint32_t numKth;
            BlockScan(temp_storage).ExclusiveSum(kth, kthIndex, numKth);
            __syncthreads();
            if (kth) {
                uint32_t offset = kthIndex + startKth;
                if (offset < outputSliceSize) {
                    topKSliceStart[offset] = {(float)val, idx};
                }
            }
            startKth += numKth;
        }
    }
}

struct CudaDevicePropSingleton {
CudaDevicePropSingleton() {
    cudaSetDevice(0);
    cudaFree(0);
    cudaGetDeviceProperties(&prop, 0);
}
cudaDeviceProp prop;
} prop_singleton;

cudaDeviceProp* getCurrentDeviceProperties() {
    return &prop_singleton.prop;
}

int get_items_per_thread(int num_slices, int slice_size) {
    // occupancy of this kernel is limited by registers per threads
    constexpr int REGS_PER_THREAD = 40; // from nsight launch statistics
    constexpr int REGS_PER_BLOCK = REGS_PER_THREAD * BLOCK_THREADS;
    cudaDeviceProp* prop = getCurrentDeviceProperties();
    int mpc = prop->multiProcessorCount;                        // 56
    int regs_per_mp = prop->regsPerMultiprocessor;              // 64KB
    int max_blocks_per_mp = prop->maxBlocksPerMultiProcessor;   // 16

    int blocks_per_mp = std::min(regs_per_mp / REGS_PER_BLOCK, max_blocks_per_mp);
    int total_size = slice_size * num_slices;
    int total_threads = mpc * blocks_per_mp * BLOCK_THREADS;
    int64_t items_per_thread = (total_size + total_threads - 1) / total_threads;
    items_per_thread = std::max(MIN_ITEMS_PER_THREAD, std::min((int)items_per_thread, MAX_ITEMS_PER_THREAD)); // clamp to (4, 64)
    return items_per_thread;
}

class BlockIdxToKey {
    uint32_t blocks_per_slice;
public:
    BlockIdxToKey(uint32_t blocks_per_slice): blocks_per_slice(blocks_per_slice) {}
    __device__ __forceinline__ uint32_t operator()(uint32_t blk) const {
        return blk / blocks_per_slice;
    }
};

template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT>
inline void inclusive_sum_by_key(
        KeysInputIteratorT keys, 
        ValuesInputIteratorT input,
        ValuesOutputIteratorT output,
        int64_t num_items,
        int8_t* workspace,
        cudaStream_t stream) {
#if CUB_VERSION < 101500
    size_t temp_storage_bytes = 0;
    CHECK_CUDA(cub::InclusiveSumByKey(nullptr, temp_storage_bytes,
            keys, input, output, num_items, cub::Equality(), stream));
    assert(temp_storage_bytes < 64 * 1024 * 1024);
    CHECK_CUDA(cub::InclusiveSumByKey(workspace, temp_storage_bytes,
            keys, input, output, num_items, cub::Equality(), stream));
#else
    size_t temp_storage_bytes = 0;
    CHECK_CUDA(cub::DeviceScan::InclusiveSumByKey(nullptr, temp_storage_bytes,
            keys, input, output, num_items, cub::Equality(), stream));
    assert(temp_storage_bytes < 64 * 1024 * 1024);
    CHECK_CUDA(cub::DeviceScan::InclusiveSumByKey(workspace, temp_storage_bytes,
            keys, input, output, num_items, cub::Equality(), stream));
#endif
}

template <typename T, int Dim>
void launch(
        T* input,
        int inputSliceSize,
        int outputSliceSize, // aka `k`
        int numInputSlices,
        Pair* topK,
        int8_t* workspace,
        cudaStream_t stream) {

    // configure items_per_thread based on device architecture and input size
    int items_per_thread = get_items_per_thread(numInputSlices, inputSliceSize);
    int items_per_block = items_per_thread * BLOCK_THREADS;

    uint32_t blocks_per_slice = (inputSliceSize + items_per_block - 1) / items_per_block;
    uint32_t num_blocks = numInputSlices * blocks_per_slice;

    // temporary storage
    T* kthValues = reinterpret_cast<T*>(workspace);   // numInputSlices * sizeof(T)
    workspace += align_bytes(numInputSlices * sizeof(T));
    uint32_t* semaphores = reinterpret_cast<uint32_t*>(workspace);
    workspace += align_bytes(numInputSlices * sizeof(uint32_t));
    CHECK_CUDA(cudaMemsetAsync(semaphores, 0, numInputSlices * sizeof(uint32_t), stream));
    uint32_t* ks_to_find = reinterpret_cast<uint32_t*>(workspace);
    workspace += align_bytes(numInputSlices * sizeof(uint32_t));

    uint32_t k_to_find = inputSliceSize - outputSliceSize + 1;
    fill<uint32_t><<<std::min(((int64_t)numInputSlices + 511) / 512, (int64_t)1073741824), 512, 0, stream>>>(
        ks_to_find, k_to_find, numInputSlices);
    CHECK_CUDA(cudaGetLastError());

    uint32_t* desired = reinterpret_cast<uint32_t*>(workspace);
    workspace += align_bytes(numInputSlices * sizeof(uint32_t));

    short* counts = reinterpret_cast<short*>(workspace);
    workspace += align_bytes(num_blocks * RADIX_DIGITS * sizeof(short));

    uint32_t* withinKCounts = reinterpret_cast<uint32_t*>(workspace);
    workspace += align_bytes(num_blocks * sizeof(uint32_t));
    CHECK_CUDA(cudaMemsetAsync(withinKCounts, 0, num_blocks * sizeof(uint32_t), stream));
    
    uint32_t* kthCounts = reinterpret_cast<uint32_t*>(workspace);
    workspace += align_bytes(num_blocks * sizeof(uint32_t));

    uint32_t desiredMask = 0;
    int grid = num_blocks;
    dim3 block(BLOCK_THREADS);

    // iterate radix bits for multiple passes
    for (int current_bit = sizeof(T) * 8 - RADIX_BITS; current_bit >= 0; current_bit -= RADIX_BITS) {
        radixFindKthValues<T, Dim><<<grid, block, 0, stream>>>(
            input,
            inputSliceSize,
            ks_to_find,
            numInputSlices,
            current_bit,
            items_per_thread,
            blocks_per_slice,
            desiredMask,
            semaphores,
            desired,
            counts,
            kthValues);
        CHECK_CUDA(cudaGetLastError());
        computeBlockwiseWithinKCounts<<<grid, RADIX_DIGITS, 0, stream>>>(
            desired, counts, blocks_per_slice, current_bit, withinKCounts, num_blocks);
        CHECK_CUDA(cudaGetLastError());
        desiredMask = Bitfield<uint32_t>::setBitfield(desiredMask, RADIX_MASK, current_bit, RADIX_BITS);
    }

    computeBlockwiseKthCounts<<<std::min(((int64_t)numInputSlices + 255) / 256, (int64_t)1073741824), 256, 0, stream>>>(
        desired, counts, num_blocks, blocks_per_slice, kthCounts);
    CHECK_CUDA(cudaGetLastError());
    // Do a prefix scan of withinKCounts and kthCounts using slice_idx as keys to get the starting index of each block
    using counting_iter_t = cub::CountingInputIterator<uint32_t, uint32_t>;
    using slice_idx_iter_t = cub::TransformInputIterator<uint32_t, BlockIdxToKey, counting_iter_t>;
    slice_idx_iter_t slice_idx_iter(counting_iter_t(0), BlockIdxToKey(blocks_per_slice));
    inclusive_sum_by_key(slice_idx_iter, withinKCounts, withinKCounts, num_blocks, workspace, stream);
    inclusive_sum_by_key(slice_idx_iter, kthCounts, kthCounts, num_blocks, workspace, stream);
    // copy topk values to output tensor
    gatherTopK<T, Dim><<<grid, block, 0, stream>>>(
        input, inputSliceSize, outputSliceSize, numInputSlices,
        topK, items_per_thread,
        blocks_per_slice, kthValues, withinKCounts, kthCounts, num_blocks);
    CHECK_CUDA(cudaGetLastError());
}

}   // namespace mbtopk


template <typename T>
void launch_gather_topk_kernel(
        T* scores,
        Pair* topk_out,
        int8_t* workspace,
        const int k,
        const int batch_size,
        const int dim,
        cudaStream_t stream) {
    mbtopk::launch<T, 2>(
        scores,
        dim,
        k,
        batch_size,
        topk_out,
        workspace,
        stream);
}