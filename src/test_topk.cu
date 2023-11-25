#include <algorithm>
#include <vector>
#include <chrono>
#include <numeric>

#include <cuda.h>
#include <cub/cub.cuh>
#include <cuda_fp16.hpp>

#include "fast_topk.cuh"

template <typename T>
void test_topk() {
    const int size = 7853052;
    const int topk = 100;

    std::vector<T> h_data(size);
    for (auto& item : h_data) {
        item = static_cast<T>(std::rand() % 100);
    }

    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);

    // std::partial_sort
    {
        T* d_data = nullptr;
        cudaMalloc(&d_data, sizeof(T) * size);
        cudaMemcpy(d_data, h_data.data(), sizeof(T) * size, cudaMemcpyHostToDevice);

        std::vector<T> scores(size);

        // warmup
        for (int i = 0; i < 3; ++i) {
            std::iota(indices.begin(), indices.end(), 0);

            cudaMemcpy(scores.data(), d_data, sizeof(T) * size, cudaMemcpyDeviceToHost);
            std::partial_sort(indices.begin(), indices.begin() + topk, indices.end(),
                [&scores](const int& a, const int& b) {
                    if (scores[a] != scores[b]) {
                        return scores[a] > scores[b];  // 按照分数降序排序
                    }
                    return a < b;  // 如果分数相同，按索引从小到大排序
                });
        }

        for (int i = 0; i < 10; ++i) {
            std::iota(indices.begin(), indices.end(), 0);
            cudaDeviceSynchronize();

            auto start = std::chrono::high_resolution_clock::now();
            cudaMemcpy(scores.data(), d_data, sizeof(T) * size, cudaMemcpyDeviceToHost);

            std::partial_sort(indices.begin(), indices.begin() + topk, indices.end(),
                [&scores](const int& a, const int& b) {
                    if (scores[a] != scores[b]) {
                        return scores[a] > scores[b];  // 按照分数降序排序
                    }
                    return a < b;  // 如果分数相同，按索引从小到大排序
                });

            auto stop = std::chrono::high_resolution_clock::now();
            double cur_time = std::chrono::duration<double, std::milli>(stop - start).count();
            printf("std::partial_sort: %.3fms\n", cur_time);
        }

        cudaFree(d_data);
    }

    // cub::DeviceRadixSort
    {
        T* d_data;
        T* d_data_sorted;
        int* d_indices;
        int* d_indices_sorted;
        void* workspace;

        cudaMalloc(&d_data, sizeof(T) * size);
        cudaMalloc(&d_data_sorted, sizeof(T) * size);
        cudaMalloc(&d_indices, sizeof(int) * size);
        cudaMalloc(&d_indices_sorted, sizeof(int) * size);
        cudaMalloc(&workspace, 64 * size_t(1024 * 1024));
        cudaMemcpy(d_data, h_data.data(), sizeof(T) * size, cudaMemcpyHostToDevice);

        std::iota(indices.begin(), indices.end(), 0);
        cudaMemcpy(d_indices, indices.data(), sizeof(int) * size, cudaMemcpyHostToDevice);

        for (int i = 0; i < 3; ++i) {
            size_t temp_storage_bytes = 0u;
            cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes,
                    d_data, d_data_sorted, d_indices, d_indices_sorted,
                    size, 0, sizeof(T) * 8);
            
            assert(temp_storage_bytes <= 64 * size_t(1024 * 1024));
            
            cub::DeviceRadixSort::SortPairsDescending(workspace, temp_storage_bytes,
                    d_data, d_data_sorted, d_indices, d_indices_sorted,
                    size, 0, sizeof(T) * 8);
            cudaDeviceSynchronize();
        }

        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            size_t temp_storage_bytes = 0u;
            cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes,
                    d_data, d_data_sorted, d_indices, d_indices_sorted,
                    size, 0, sizeof(T) * 8);
            
            assert(temp_storage_bytes <= 64 * size_t(1024 * 1024));
            
            cub::DeviceRadixSort::SortPairsDescending(workspace, temp_storage_bytes,
                    d_data, d_data_sorted, d_indices, d_indices_sorted,
                    size, 0, sizeof(T) * 8);
            std::vector<int> ans(topk);
            cudaMemcpy(ans.data(), d_indices_sorted, topk * sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            auto stop = std::chrono::high_resolution_clock::now();
            double cur_time = std::chrono::duration<double, std::milli>(stop - start).count();
            printf("cub::DeviceRadixSort: %.3fms\n", cur_time);
        }
    }

    // pytorch::topk
    {
        T* d_data;
        Pair* d_topk;
        void* workspace;

        for (int batch = 1; batch <= 4; ++batch) {
            cudaMalloc(&d_data, sizeof(T) * size * batch);
            cudaMalloc(&d_topk, sizeof(Pair) * topk * batch);
            cudaMalloc(&workspace, 64 * size_t(1024 * 1024));
            for (int i = 0; i < batch; ++i) {
                cudaMemcpy(d_data + i * size, h_data.data(), sizeof(T) * size, cudaMemcpyHostToDevice);
            }

            CHECK_CUDA(cudaDeviceSynchronize());
            
            for (int i = 0; i < 3; ++i) {
                launch_gather_topk_kernel(
                    d_data, d_topk, (int8_t*)workspace,
                    topk, batch, size, 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            for (int i = 0; i < 10; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                launch_gather_topk_kernel(
                    d_data, d_topk, (int8_t*)workspace,
                    topk, batch, size, 0);
                cudaDeviceSynchronize();
                std::vector<Pair> h_topk(batch * topk);
                cudaMemcpy(h_topk.data(), d_topk, sizeof(Pair) * batch * topk, cudaMemcpyDeviceToHost);
                
                auto stop = std::chrono::high_resolution_clock::now();
                double cur_time = std::chrono::duration<double, std::milli>(stop - start).count();
                printf("pytorch::topk[%d]: %.3fms\n", batch, cur_time);
            }

            cudaFree(d_data);
            cudaFree(d_topk);
            cudaFree(workspace);
        }
    }
}

template <>
void test_topk<half>() {
    const int size = 7853052;
    const int topk = 100;

    using T = half;

    std::vector<T> h_data(size);
    for (auto& item : h_data) {
        item = static_cast<T>(std::rand() % 100);
    }

    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);

    // cub::DeviceRadixSort
    {
        T* d_data;
        T* d_data_sorted;
        int* d_indices;
        int* d_indices_sorted;
        void* workspace;

        cudaMalloc(&d_data, sizeof(T) * size);
        cudaMalloc(&d_data_sorted, sizeof(T) * size);
        cudaMalloc(&d_indices, sizeof(int) * size);
        cudaMalloc(&d_indices_sorted, sizeof(int) * size);
        cudaMalloc(&workspace, 64 * 1024 * 1024);
        cudaMemcpy(d_data, h_data.data(), sizeof(T) * size, cudaMemcpyHostToDevice);

        std::iota(indices.begin(), indices.end(), 0);
        cudaMemcpy(d_indices, indices.data(), sizeof(int) * size, cudaMemcpyHostToDevice);

        for (int i = 0; i < 3; ++i) {
            size_t temp_storage_bytes = 0u;
            cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes,
                    d_data, d_data_sorted, d_indices, d_indices_sorted,
                    size, 0, sizeof(T) * 8);
            
            assert(temp_storage_bytes <= 64 * size_t(1024 * 1024));
            
            cub::DeviceRadixSort::SortPairsDescending(workspace, temp_storage_bytes,
                    d_data, d_data_sorted, d_indices, d_indices_sorted,
                    size, 0, sizeof(T) * 8);
            cudaDeviceSynchronize();
        }

        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            size_t temp_storage_bytes = 0u;
            cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_storage_bytes,
                    d_data, d_data_sorted, d_indices, d_indices_sorted,
                    size, 0, sizeof(T) * 8);
            
            assert(temp_storage_bytes <= 64 * size_t(1024 * 1024));
            
            cub::DeviceRadixSort::SortPairsDescending(workspace, temp_storage_bytes,
                    d_data, d_data_sorted, d_indices, d_indices_sorted,
                    size, 0, sizeof(T) * 8);
            
            std::vector<int> ans(topk);
            cudaMemcpy(ans.data(), d_indices_sorted, topk * sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            auto stop = std::chrono::high_resolution_clock::now();
            double cur_time = std::chrono::duration<double, std::milli>(stop - start).count();
            printf("cub::DeviceRadixSort: %.3fms\n", cur_time);
        }
    }

    // pytorch::topk
    {
        T* d_data;
        Pair* d_topk;
        void* workspace;

        for (int batch = 1; batch <= 4; ++batch) {
            cudaMalloc(&d_data, sizeof(T) * size * batch);
            cudaMalloc(&d_topk, sizeof(Pair) * topk * batch);
            cudaMalloc(&workspace, 64 * size_t(1024 * 1024));
            for (int i = 0; i < batch; ++i) {
                cudaMemcpy(d_data + i * size, h_data.data(), sizeof(T) * size, cudaMemcpyHostToDevice);
            }

            CHECK_CUDA(cudaDeviceSynchronize());
            
            for (int i = 0; i < 3; ++i) {
                launch_gather_topk_kernel(
                    d_data, d_topk, (int8_t*)workspace,
                    topk, batch, size, 0);
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            for (int i = 0; i < 10; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                launch_gather_topk_kernel(
                    d_data, d_topk, (int8_t*)workspace,
                    topk, batch, size, 0);
                cudaDeviceSynchronize();
                std::vector<Pair> h_topk(batch * topk);
                cudaMemcpy(h_topk.data(), d_topk, sizeof(Pair) * batch * topk, cudaMemcpyDeviceToHost);
                
                auto stop = std::chrono::high_resolution_clock::now();
                double cur_time = std::chrono::duration<double, std::milli>(stop - start).count();
                printf("pytorch::topk[%d]: %.3fms\n", batch, cur_time);
            }

            cudaFree(d_data);
            cudaFree(d_topk);
            cudaFree(workspace);
        }
    }
}

int main() {
    printf("==== test_float ====\n");
    test_topk<float>();
    printf("==== test_int16 ====\n");
    test_topk<int16_t>();
    printf("==== test_half ====\n");
    test_topk<half>();
}