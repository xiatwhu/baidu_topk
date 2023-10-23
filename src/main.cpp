/*
 * SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cuda.h>
#include <chrono>
#include <vector>
#include <random>
#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <sys/time.h>
#include <stdio.h>
#include <dirent.h>
#include "topk.h"
#include <sys/stat.h>

std::vector<std::string> getFilesInDirectory(const std::string& directory)
{
    std::vector<std::string> files;
    DIR* dirp = opendir(directory.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        struct stat path_stat;
        stat((directory + "/" + dp->d_name).c_str(), &path_stat);
        if (S_ISREG(path_stat.st_mode)) // Check if it's a regular file - not a directory
            files.push_back(dp->d_name);
    }
    closedir(dirp);
    std::sort(files.begin(), files.end()); // sort the files
    return files;
}

struct UserSpecifiedInput
{
    int n_docs;
    std::vector<std::vector<uint16_t>> querys;
    std::vector<std::vector<uint16_t>> docs;
    std::vector<uint16_t> doc_lens;

    UserSpecifiedInput(std::string qf, std::string df) {
        load(qf, df);
    }

    void load(std::string query_file_dir, std::string docs_file_name) {
        std::stringstream ss;
        std::string tmp_str;
        std::string tmp_index_str;

        std::vector<std::string> files = getFilesInDirectory(query_file_dir);
        for(const auto& query_file_name: files)
        {
            std::vector<uint16_t> single_query;

            std::ifstream query_file(query_file_dir + "/" + query_file_name);
            while (std::getline(query_file, tmp_str)) {
                ss.clear();
                ss << tmp_str;
                std::cout << query_file_name << ":" << tmp_str << std::endl;
                while (std::getline(ss, tmp_index_str, ',')) {
                    single_query.emplace_back(std::stoi(tmp_index_str));
                }
            }
            query_file.close();
            ss.clear();
            std::sort(single_query.begin(), single_query.end()); // pre-sort the query
            querys.emplace_back(single_query);
        }
        std::cout << "query_size: " << querys.size() << std::endl;

        std::ifstream docs_file(docs_file_name);
        while (std::getline(docs_file, tmp_str)) {
            std::vector<uint16_t> next_doc;
            ss.clear();
            ss << tmp_str;
            while (std::getline(ss, tmp_index_str, ',')) {
                next_doc.emplace_back(std::stoi(tmp_index_str));
            }
            docs.emplace_back(next_doc);
            doc_lens.emplace_back(next_doc.size());
        }
        docs_file.close();
        ss.clear();
        n_docs = docs.size();
        std::cout << "doc_size: " << docs.size() << std::endl;
    }
};

int main(int argc, char *argv[])
{    
    if (argc != 4) {
        std::cout << "Usage: query_doc_scoring.bin <doc_file_name> <query_file_name> <output_filename>" << std::endl;
        return -1;
    }
    std::string doc_file_name = argv[1];;
    std::string query_file_dir = argv[2];;
    std::string output_file = argv[3];

    std::cout << "start get topk" << std::endl;

    // �文件
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    UserSpecifiedInput inputs(query_file_dir, doc_file_name);
    std::vector<std::vector<int>> indices;
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << "read file cost " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms " << std::endl;

    // 计算得分
    doc_query_scoring_gpu_function(inputs.querys, inputs.docs, inputs.doc_lens, indices);
    
    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    std::cout << "topk cost " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms " << std::endl;

    // get total time
    std::chrono::milliseconds total_time = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t1);
    // write result data
    std::ofstream ofs;
    ofs.open(output_file, std::ios::out);
    // first line topk cost time in ms
    ofs << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()  << std::endl;
    // topk index
    for (auto& s_indices : indices) { //makesure indices.size() == querys.size()
        for(size_t i = 0; i < s_indices.size(); ++i)
        {
            ofs << s_indices[i];
            if(i != s_indices.size() - 1) // if not the last element
                ofs << "\t";
        }
        ofs << "\n";
    }
    ofs.close();

    std::cout << "all cost " << total_time.count() << " ms " << std::endl;
    std::cout << "end get topk" << std::endl;
    return 0;
}
