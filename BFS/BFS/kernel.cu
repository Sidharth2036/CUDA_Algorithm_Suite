#include <iostream>
#include <queue>
#include <vector>
#include <cuda_runtime.h>

#define N 30000 // Number of vertices in the graph

// Kernel function for BFS on GPU
__global__ void BFS_GPU(int* adj_matrix, bool* visited, int* queue, int* queue_end, int* result, int start_node) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        visited[start_node] = true;
        queue[0] = start_node;
        *queue_end = 1;
    }

    __syncthreads();

    while (*queue_end > 0) {
        int idx = atomicSub(queue_end, 1) - 1;
        int node = queue[idx];

        if (node == -1)
            break;

        for (int i = 0; i < N; ++i) {
            if (adj_matrix[node * N + i] && !visited[i]) {
                visited[i] = true;
                int new_idx = atomicAdd(queue_end, 1);
                queue[new_idx] = i;
            }
        }
    }

    if (tid == 0) {
        *result = 1;
    }
}

// CPU implementation of BFS
void BFS_CPU(int* adj_matrix, bool* visited, int start_node) {
    std::queue<int> q;
    q.push(start_node);
    visited[start_node] = true;

    while (!q.empty()) {
        int node = q.front();
        q.pop();

        for (int i = 0; i < N; ++i) {
            if (adj_matrix[node * N + i] && !visited[i]) {
                visited[i] = true;
                q.push(i);
            }
        }
    }
}

int main() {
    // Initialize adjacency matrix (assuming an undirected graph for simplicity)
    int* adj_matrix = new int[N * N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            adj_matrix[i * N + j] = rand() % 2; // Randomly assign edges
        }
    }

    // Choose a random starting node
    int start_node = rand() % N;

    // Initialize visited array
    bool* visited = new bool[N];
    for (int i = 0; i < N; ++i) {
        visited[i] = false;
    }

    // Perform BFS on CPU and measure time
    clock_t cpu_start = clock();
    BFS_CPU(adj_matrix, visited, start_node);
    clock_t cpu_end = clock();
    double cpu_time = double(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Reset visited array
    for (int i = 0; i < N; ++i) {
        visited[i] = false;
    }

    // Allocate memory on GPU
    int* d_adj_matrix, * d_queue, * d_queue_end, * d_result;
    bool* d_visited;
    cudaMalloc((void**)&d_adj_matrix, N * N * sizeof(int));
    cudaMalloc((void**)&d_visited, N * sizeof(bool));
    cudaMalloc((void**)&d_queue, N * sizeof(int));
    cudaMalloc((void**)&d_queue_end, sizeof(int));
    cudaMalloc((void**)&d_result, sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_adj_matrix, adj_matrix, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, visited, N * sizeof(bool), cudaMemcpyHostToDevice);

    // Perform BFS on GPU and measure time
    clock_t gpu_start = clock();
    BFS_GPU << <1, N >> > (d_adj_matrix, d_visited, d_queue, d_queue_end, d_result, start_node);
    cudaDeviceSynchronize();
    clock_t gpu_end = clock();
    double gpu_time = double(gpu_end - gpu_start) / CLOCKS_PER_SEC;

    // Copy result from GPU
    int gpu_result;
    cudaMemcpy(&gpu_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_adj_matrix);
    cudaFree(d_visited);
    cudaFree(d_queue);
    cudaFree(d_queue_end);
    cudaFree(d_result);

    // Output results
    std::cout << "Start Node: " << start_node << std::endl;
    std::cout << "CPU Time: " << cpu_time << " seconds" << std::endl;
    std::cout << "GPU Time: " << gpu_time << " seconds" << std::endl;

    // Cleanup
    delete[] adj_matrix;
    delete[] visited;

    return 0;
}
