// 倒排索引搜索实现
// Created on 25-04-21

#pragma once
#include "inverted_index.cuh"


// 用于计算查询向量与聚类中心的距离
__global__ void computeQueryCenterDist(float* data_d, float* c_vec, int* qid_list, int qnum, 
                                            float* q_center_dist, int dim, int* data_info) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程处理一个查询向量与一个聚类中心的距离计算
    if (tid < qnum * K_NUM) {
        int qid = tid / K_NUM; // 查询ID
        int cid = tid % K_NUM; // 聚类ID
        
        float dist = 0.0f;
        
        
        if (data_info[2] == 5) { // Cosine similarity -> angle distance
            float sa1 = 0.0f, sa2 = 0.0f, sa3 = 0.0f;
            for (int d = 0; d < dim; d++) {
                sa1 += data_d[qid_list[qid] * dim + d] * data_d[qid_list[qid] * dim + d];
                sa2 += c_vec[cid * dim + d] * c_vec[cid * dim + d];
                sa3 += data_d[qid_list[qid] * dim + d] * c_vec[cid * dim + d];
            }
            sa1 = pow(sa1, 0.5);
            sa2 = pow(sa2, 0.5);
            if (sa1 * sa2 == 0) {
                printf("Error!!!\n");
            }
            dist = sa3 / (sa1 * sa2);
            if (dist > 1) {
                dist = 0.99999999999999999;
            }
            dist = abs(acos(dist) * 180 / 3.1415926);
        }
        q_center_dist[qid * K_NUM + cid] = dist;
    }
}

// 检查聚类是否满足筛选条件
__global__ void filterClusters(float* q_center_dist, float r, float* min_dis,
                             bool* cluster_filter, int qnum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < qnum * K_NUM) {
        int qid = tid / K_NUM;  // 查询ID
        int cid = tid % K_NUM;  // 聚类ID
        
        float dist = q_center_dist[qid * K_NUM + cid];
        

        if (dist + r < min_dis[cid]) {
            cluster_filter[qid * K_NUM + cid] = false;  
        } else {
            cluster_filter[qid * K_NUM + cid] = true;   
        }
    }
}


__global__ void processClusterVectors(float* data_d, int* qid_list, int qnum, bool* cluster_filter,
                                    int* inverted_list, int* list_length, int* list_offset,
                                    int* result_counts, float r, int dim, int* data_info) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // 每个线程处理一组查询和聚类的组合
    for (int idx = tid; idx < qnum * K_NUM; idx += total_threads) {
        int qid = idx / K_NUM;  // 查询ID
        int cid = idx % K_NUM;  // 聚类ID
        
        // 如果该聚类被过滤掉，跳过处理
        if (!cluster_filter[qid * K_NUM + cid]) continue;
        
        // 获取聚类中的向量列表信息
        int list_start = list_offset[cid];
        int list_end = list_start + list_length[cid];
        
        // 处理聚类中的所有向量
        for (int i = list_start; i < list_end; i++) {
            int vector_id = inverted_list[i];
            
            // 计算向量与查询向量的距离
            float dist = 0.0f;
            
            
            if (data_info[2] == 5) { // Cosine similarity -> angle distance
                float sa1 = 0.0f, sa2 = 0.0f, sa3 = 0.0f;
                for (int d = 0; d < dim; d++) {
                    sa1 += data_d[vector_id * dim + d] * data_d[vector_id * dim + d];
                    sa2 += data_d[qid_list[qid] * dim + d] * data_d[qid_list[qid] * dim + d];
                    sa3 += data_d[vector_id * dim + d] * data_d[qid_list[qid] * dim + d];
                }
                sa1 = pow(sa1, 0.5);
                sa2 = pow(sa2, 0.5);
                if (sa1 * sa2 == 0) {
                    printf("Error!!!\n");
                }
                dist = sa3 / (sa1 * sa2);
                if (dist > 1) {
                    dist = 0.99999999999999999;
                }
                dist = abs(acos(dist) * 180 / 3.1415926);
            }
            if (dist <= r) {
                atomicAdd(&result_counts[qid], 1);
            }
        }
    }
}


// 范围查询的主函数 
void searchInvertedIndexRnn(float* data_d, InvertedIndex* inverted_index, int* qid_list, int qnum, float r, int* data_info,int* &res) {
    auto st = std::chrono::high_resolution_clock::now();
    printf("Starting range query using inverted index...\n");
    cudaMallocManaged(&res, qnum * sizeof(int));
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return ;
    }
    cudaDeviceSynchronize();
    cudaMemset(res, 0, qnum * sizeof(int));
    int v_dim = data_info[0];
    
    float* q_center_dist;  //大小为 qnum * K_NUM
    /*
    查询向量0到聚类中心1的距离 = q_center_dist[0 * K_NUM + 1];
    查询向量1到聚类中心2的距离 = q_center_dist[1 * K_NUM + 2]; 
    */
    cudaMallocManaged(&q_center_dist, qnum * K_NUM * sizeof(float));
    bool* cluster_filter;  //标记每个聚类是否需要被该查询q处理,大小同样为 qnum * K_NUM
    cudaMallocManaged(&cluster_filter, qnum * K_NUM * sizeof(bool));
    
    cudaDeviceSynchronize();
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "initializeResults error: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }
    
    // 计算每个查询向量与每个聚类中心的距离
    computeQueryCenterDist<<<(qnum * K_NUM + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(
        data_d, inverted_index->c_vec, qid_list, qnum, 
        q_center_dist, v_dim, data_info
    );
    cudaDeviceSynchronize();
    
    // 检查错误
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "computeQueryCentroidDistances error: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }
    
    // 筛选聚类
    filterClusters<<<(qnum * K_NUM + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(
        q_center_dist, r, inverted_index->min_dis, 
        cluster_filter, qnum
    );
    cudaDeviceSynchronize();
    
    // 检查错误
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "filterClusters error: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }
    
    processClusterVectors<<<(qnum * K_NUM + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(    //这里的gridSize可能有错，ai用了min(128,xxx)
        data_d, qid_list, qnum, cluster_filter,
        inverted_index->inverted_list, inverted_index->list_length, inverted_index->list_offset,
        res, r, v_dim, data_info
    );
    cudaDeviceSynchronize();
    
    // 检查错误
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "processClusterVectors error: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }
    cudaDeviceSynchronize();
    // 释放临时内存
    cudaFree(q_center_dist);
    cudaFree(cluster_filter);
    
    printf("Inverted index range query completed!\n");
    auto ed = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration_seconds = ed - st;
    printf("Execution time: %.6f seconds\n", duration_seconds.count());
}