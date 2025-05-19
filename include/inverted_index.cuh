// GTS Inverted Index Construction
// Created on 25-04-21

#pragma once
#include <cfloat> // For FLT_MAX
#include <cstdio> // For printf



#define K_NUM 10                     // K-means聚类中心数量
#define MAX_ITER 20                  // K-means最大迭代次数
#define CONVERGE_THRESHOLD 0.001f    // K-means收敛阈值
#define THREAD_NUM 512



typedef struct {
    //聚类编号为索引
    float* c_vec;                    // 聚类中心向量
    float* min_dis;                  // 聚类中向量到中心的最小距离
    
    
    //向量编号为索引
    int* c_id;                       //c_id[i]表示编号为i的向量所属的聚类 

    //顺序存储的倒排索引表
    int* inverted_list;             // 长度是v_num，按照聚类0，聚类1...顺序进行分割
    int* list_offset;               // 聚类在inverted_list表中的起始偏移量
    int* list_length;               // 聚类中向量的数量
    /*
        对于i号聚类
        其包含的向量数量是list_length[i]
        聚类i包含的具体向量编号有：inverted_list[ list_offset[i] ] ~ inverted_list[ list_offset[i] + list_length[i] - 1 ]
    */
} InvertedIndex;



__managed__ int v_dim;              // 向量维度
__managed__ int v_num;           // 向量数量
__managed__ float total_dist;     // 所有样本点到其聚类中心的总距离（用于收敛判断,前后两轮该值的变化量小于阈值就停止迭代）

// 初始化随机聚类中心
__global__ void init_clusters(float* data_d, float* c_vec, int dim, int vecnum, int* cluster_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 每个线程负责一个聚类中心
    if (tid < K_NUM) {
        cluster_count[tid] = -1;  //后面使用cluster_count时是先++再读取，因此这里先初始化为-1
        for (int i = 0; i < dim; i++) {
            c_vec[tid * dim + i] = data_d[tid * dim + i];
        }
    }
    //哪怕指定选用前K_NUM个向量，也可能有重复的。初始中心不能有重复。代码没有处理这种潜在情况。
}




// 计算向量到所有聚类中心的距离，并分配到最近的中心
__global__ void assignToClusters(float* data_d, float* c_vec, int* c_id, int dim, int vecnum, int* data_info) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    //每个线程处理一个向量
    if (vid < vecnum) {
        int best_cluster = 0;
        float min_dist = FLT_MAX;
        
        // 计算向量到每个聚类中心的距离
        for (int c = 0; c < K_NUM; c++) {
            float dist = 0.0f;
            
            
            if (data_info[2] == 5) { // Cosine similarity -> angle distance
                float sa1 = 0.0f, sa2 = 0.0f, sa3 = 0.0f;
                for (int d = 0; d < dim; d++) {
                    sa1 += data_d[vid * dim + d] * data_d[vid * dim + d];
                    sa2 += c_vec[c * dim + d] * c_vec[c * dim + d];
                    sa3 += data_d[vid * dim + d] * c_vec[c * dim + d];
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
            
            // 更新最近聚类中心
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
        
        // 将向量分配给最近的聚类
        c_id[vid] = best_cluster;
        
        // 原子操作累加总距离，用于判断收敛
        atomicAdd(&total_dist, min_dist);
    }
}

// 计算每个聚类包含的向量数量
__global__ void countClusterSizes(int* c_id, int* lengths, int vecnum) {
    int vid = threadIdx.x + blockIdx.x * blockDim.x;
    if (vid < vecnum) atomicAdd(&lengths[c_id[vid]], 1);
}

//计算聚类中心的偏移量
__global__ void computeOffset(int* lengths, int* offsets) {
    //这里假定了K_NUM小于blockSize的上限。否则需要改代码
    int tid = threadIdx.x;
    if (tid == 0)
     {
        offsets[0] = 0;
        return;
    }
    __syncthreads();
    if (tid < K_NUM) offsets[tid] = offsets[tid-1] + lengths[tid-1];
}

//更新聚类中心
__global__ void update_center(float* data_d, float* c_vec, int* c_id, int* lengths, int dim, int vecnum) {
    int cid = blockIdx.x; // 每个block负责一个聚类中心
    int tid = threadIdx.x; // 每个线程负责一个维度
    
    //这里假定了向量维数小于THREAD_NUM。如果不是则还要修改代码，让一个线程负责处理多个维度
    if (cid < K_NUM && tid < dim) {
        float sum = 0.0f;
        int count = lengths[cid];
        
        // 如果该聚类没有分配到向量，则保持不变
        if (count == 0) return;
        
        // 计算所有分配到此聚类的向量在该维度上的平均值
        //这里可能成为性能瓶颈，可以优化
        for (int i = 0; i < vecnum; i++)
        {
            //float delta = c_id[i] == cid ? data_d[i * dim + tid] : 0;
            if(c_id[i] != cid)continue;
            sum += data_d[i * dim + tid];
        }
        
        // 更新聚类中心
        c_vec[cid * dim + tid] = sum / count;
    }
}

//构建倒排列表
__global__ void buildInvertedLists(int* c_id, int* inverted_list, int* offsets, int* lengths, int vecnum, int* cluster_count) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vid < vecnum) {
        int cluster = c_id[vid];
        int pos = atomicAdd(&cluster_count[cluster], 1); //先作原子加，然后把和读入pos
        // 将向量ID添加到倒排列表中的正确位置
        inverted_list[offsets[cluster] + pos] = vid;
    }
}

//为计算聚类的min_dis服务
__device__ static float atomicMinFloat(float* addr, float value) {
    int* addr_as_uint = (int*)addr;
    float old = *addr;
    while (value < old) {
        int assumed = __float_as_int(old);
        int desired = __float_as_int(value);
        int current = atomicCAS(addr_as_uint, assumed, desired);
        if (current == assumed) {
            break;
        }
        old = __int_as_float(current);
    }
    return old;
}

// 计算每个聚类的min_dis
__global__ void computeClusterMinDist(float* data_d, float* c_vec, int* inverted_list, int* list_length, 
                                       int* list_offset, float* min_dis, int dim, int* data_info) {
    //求一组数的最小值，可以通过数组归约来更高效的进行。这是一个优化点

    int cid = blockIdx.x; // 每个block处理一个聚类
    int tid = blockDim.x * blockIdx.x + threadIdx.x; //0~511号线程处理该聚类中的0~511号向量，依此类推
    float cluster_min_dist = FLT_MAX;
    if (cid < K_NUM) {
        int list_len = list_length[cid];
        int offset = list_offset[cid];

        for (int i = tid; i < list_len; i+= THREAD_NUM) {
            int vec_id = inverted_list[offset + i];
            float dist = 0.0f;
            
            
            if (data_info[2] == 5) { // Cosine similarity -> angle distance
                float sa1 = 0.0f, sa2 = 0.0f, sa3 = 0.0f;
                for (int d = 0; d < dim; d++) {
                    sa1 += data_d[vec_id * dim + d] * data_d[vec_id * dim + d];
                    sa2 += c_vec[cid * dim + d] * c_vec[cid * dim + d];
                    sa3 += data_d[vec_id * dim + d] * c_vec[cid * dim + d];
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
            
            // 使用原子操作更新聚类的min_dis
            atomicMinFloat(&cluster_min_dist, dist);
        }
        
        // 保存结果
        min_dis[cid] = cluster_min_dist;
    }
}

// 倒排索引构建的主函数
void invertedIndexConstruct(float* data_d, int* data_info, InvertedIndex *inverted_index) {
    auto st = std::chrono::high_resolution_clock::now();
    printf("Starting to build inverted index...\n");
    v_dim = data_info[0];
    v_num = data_info[1];
    
    printf("Vector dimension: %d, Vector count: %d\n", v_dim, v_num);
    cudaMallocManaged(&inverted_index->c_id, v_num * sizeof(int));
    cudaMallocManaged(&inverted_index->c_vec, K_NUM * v_dim * sizeof(float));
    cudaMallocManaged(&inverted_index->list_length, K_NUM * sizeof(int));
    cudaMallocManaged(&inverted_index->list_offset, K_NUM * sizeof(int));
    cudaMallocManaged(&inverted_index->inverted_list, v_num * sizeof(int));
    cudaMallocManaged(&inverted_index->min_dis, K_NUM * sizeof(float));
    cudaMemset(inverted_index->list_length, 0, K_NUM * sizeof(int));
    cudaMemset(inverted_index->list_offset, 0, K_NUM * sizeof(int));
    
    // 初始化随机聚类中心
    int* cluster_count;  //这个数组在buildInvertedLists时用，先在这里把各元素初始化为0
    cudaMallocManaged(&cluster_count, K_NUM * sizeof(int));

    init_clusters<<<(K_NUM + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(data_d, inverted_index->c_vec, v_dim, v_num, cluster_count);
    cudaDeviceSynchronize();
    
    // K-means迭代
    float prev_total_dist = FLT_MAX;
    for (int iter = 0; iter < MAX_ITER; iter++) {
        total_dist = 0;
        
        // 分配向量到最近的聚类中心
        assignToClusters<<<(v_num + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(data_d, inverted_index->c_vec, inverted_index->c_id, v_dim, v_num, data_info);
        cudaDeviceSynchronize();
        
        // 检查收敛
        printf("Iteration %d: Total distance = %.6f\n", iter, total_dist);
        if (iter > 0 && fabs(prev_total_dist - total_dist) < CONVERGE_THRESHOLD * prev_total_dist) {
            printf("K-means converged!\n");
            break;
        }
        prev_total_dist = total_dist;
        
        // 计算每个聚类的大小
        cudaMemset(inverted_index->list_length, 0, K_NUM * sizeof(int));
        countClusterSizes<<<K_NUM, THREAD_NUM>>>(inverted_index->c_id, inverted_index->list_length, v_num);
        cudaDeviceSynchronize();


        // 更新聚类中心
        update_center<<<K_NUM, THREAD_NUM>>>(data_d, inverted_index->c_vec, inverted_index->c_id, inverted_index->list_length, v_dim, v_num);
        cudaDeviceSynchronize();
    }
    
    // 计算倒排列表的偏移量
    computeOffset<<<1, K_NUM>>>(inverted_index->list_length, inverted_index->list_offset);
    cudaDeviceSynchronize();
    
    // 构建倒排列表
    buildInvertedLists<<<(v_num + THREAD_NUM - 1) / THREAD_NUM,THREAD_NUM>>>(inverted_index->c_id, inverted_index->inverted_list, inverted_index->list_offset, inverted_index->list_length, v_num, cluster_count);
    cudaDeviceSynchronize();
    cudaFree(cluster_count);
    
    // 计算每个聚类的min_dis
    computeClusterMinDist<<<(K_NUM + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM>>>(data_d, inverted_index->c_vec, inverted_index->inverted_list, inverted_index->list_length, 
                                                     inverted_index->list_offset, inverted_index->min_dis, v_dim, data_info);
    cudaDeviceSynchronize();
    
    // 输出每个聚类的大小和距离范围
    printf("Inverted index construction completed! Cluster distribution:\n");
    //这里会轻微影响性能，内存页会从device迁移到host
    for (int i = 0; i < K_NUM; i++)
    {
        printf("Cluster %d: %d vectors, min_dis %.4f\n", 
               i, inverted_index->list_length[i], inverted_index->min_dis[i]);
    }
    //上述代码都只为输出每个聚类的大小和距离范围
    printf("Inverted index construction completed!\n");
    auto ed = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration_seconds = ed - st;
    printf("Execution time: %.6f seconds\n", duration_seconds.count());
}

// 释放倒排索引占用的内存
__host__ void freeInvertedIndex(InvertedIndex* inverted_index) {
    cudaFree(inverted_index->c_id);
    cudaFree(inverted_index->c_vec);
    cudaFree(inverted_index->list_length);
    cudaFree(inverted_index->list_offset);
    cudaFree(inverted_index->inverted_list);
    cudaFree(inverted_index->min_dis);
}