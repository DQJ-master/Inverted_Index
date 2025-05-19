// GTS index construction and similarity search with GTS
// Created on 24-01-05

#include <chrono>
#include "file.cuh"
#include "inverted_index.cuh"
#include "search_inverted_index.cuh"



#include <windows.h>
int *data_info;
float *data_d;
char *data_s;
int *size_s;
char *file;
char *file_q;
char *file_u;
float time_index = 0;
float time_search = 0;


int k;	 // k for knn
float r; // r for range query
int *qid_list;
int qnum;
int* res; //存储查询结果，每个查询查到的向量数量
int process_type;
InvertedIndex *inverted_index; // 倒排索引全局变量

int main(int argc, char **argv)
{
	SetConsoleOutputCP(CP_UTF8);
	file = argv[1];
	load(file, data_info, data_d, data_s, size_s);
	process_type = (int)atoi(argv[3]);
	if (process_type != 2)
	{
		file_q = argv[2];
		loadQuery(file_q, qid_list, qnum);
		k = (int)atoi(argv[4]);
		r = (float)stod(argv[4]);
	}

	// Index Construction
	cudaMallocManaged(&inverted_index, sizeof(InvertedIndex));
	auto s = std::chrono::high_resolution_clock::now();
	invertedIndexConstruct(data_d, data_info, inverted_index);
	cudaDeviceSynchronize();
	auto e = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> diff = e - s;
	time_index += diff.count();
	//printf("\nTime of index construction: %f\n", time_index);

	// Range query	if (process_type == 1)
	{
		// Check if file exists and remove it
		if (remove(argv[5]) == 0) {
			printf("Previous result file removed successfully.\n");
		}
		
		FILE *fcost = fopen(argv[5], "w");
		if (fcost == NULL) {
			printf("Error creating result file!\n");
			return -1;
		}
		fprintf(fcost, "Range search radius: %f\nResult num: \n", r);
		fflush(fcost);
		s = std::chrono::high_resolution_clock::now();
		searchInvertedIndexRnn(data_d, inverted_index, qid_list, qnum, r, data_info, res);
		e = std::chrono::high_resolution_clock::now();
		diff = e - s;
		time_search += diff.count();
		// Output results
		for (int i = 0; i < qnum; i++)
		{
			fprintf(fcost, "%d ", res[i]);
			fflush(fcost);
		}
		
		//printf("Search time: %f\n", time_search / qnum);
		//fprintf(fcost, "\nTime of index construction: %f\n", time_index);
		//fprintf(fcost, "Search time: %f\n", time_search / qnum);
		fflush(fcost);
		fclose(fcost);
	}

	// Release memory
	cudaFree(data_info);
	cudaFree(data_d);
	cudaFree(data_s);
	cudaFree(size_s);
	
	// 只有当process_type != 2时才释放查询相关内存
	if (process_type != 2) {
		cudaFree(qid_list);
		cudaFree(res);
	}
	
	// 释放倒排索引内存
	if(inverted_index != nullptr) {
		freeInvertedIndex(inverted_index);
		cudaFree(inverted_index);
	}
	
	return 0;
}