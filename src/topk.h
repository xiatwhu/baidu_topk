#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <cuda.h>

#define MAX_DOC_SIZE 128
#define MAX_QUERY_SIZE 4096
#define N_THREADS_IN_ONE_BLOCK 512
#define TOPK 100

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>> &query,
    std::vector<std::vector<uint16_t>> &docs,
    std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices);
