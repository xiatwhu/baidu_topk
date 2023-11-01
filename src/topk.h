#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <cuda.h>

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>> &query,
    std::vector<std::vector<uint16_t>> &docs,
    std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices);
