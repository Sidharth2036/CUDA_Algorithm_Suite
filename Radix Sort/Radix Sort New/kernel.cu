#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <thread>
#define BLOCK_SIZE 256

// Function to find the maximum element in the array
__global__ void findMax(int* array, int* max, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_max = 0;

    while (tid < size) {
        if (array[tid] > local_max) {
            local_max = array[tid];
        }
        tid += blockDim.x * gridDim.x;
    }

    atomicMax(max, local_max);
}

// Function to perform counting sort on GPU
__global__ void countingSort(int* input, int* output, int digit, int size) {
    extern __shared__ int temp[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int* count = &temp[10 * blockDim.x];
    int* scan = &temp[11 * blockDim.x];

    // Initialize count array
    for (int i = threadIdx.x; i < 10; i += blockDim.x) {
        count[i] = 0;
    }
    __syncthreads();

    // Count occurrences of each digit
    while (tid < size) {
        int d = (input[tid] / digit) % 10;
        atomicAdd(&count[d], 1);
        tid += blockDim.x * gridDim.x;
    }
    __syncthreads();

    // Perform exclusive scan on count array
    if (threadIdx.x == 0) {
        scan[0] = 0;
        for (int i = 1; i < 10; ++i) {
            scan[i] = scan[i - 1] + count[i - 1];
        }
    }
    __syncthreads();

    // Perform radix sort
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        int d = (input[tid] / digit) % 10;
        int index = atomicAdd(&scan[d], 1);
        output[index] = input[tid];
        tid += blockDim.x * gridDim.x;
    }
}

// Radix sort on GPU
void radixSortGPU(int* array, int size) {
    int* d_input, * d_output;
    cudaMalloc((void**)&d_input, size * sizeof(int));
    cudaMalloc((void**)&d_output, size * sizeof(int));
    cudaMemcpy(d_input, array, size * sizeof(int), cudaMemcpyHostToDevice);

    int* d_max;
    cudaMalloc((void**)&d_max, sizeof(int));
    cudaMemcpy(d_max, &array[0], sizeof(int), cudaMemcpyHostToDevice);
    findMax << <(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (d_input, d_max, size);
    int max;
    cudaMemcpy(&max, d_max, sizeof(int), cudaMemcpyDeviceToHost);

    int digit = 1;
    while (max / digit > 0) {
        countingSort << <(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 22 * BLOCK_SIZE * sizeof(int) >> > (d_input, d_output, digit, size);
        cudaMemcpy(d_input, d_output, size * sizeof(int), cudaMemcpyDeviceToDevice);
        digit *= 10;
    }

    cudaMemcpy(array, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_max);
}

// Radix sort on CPU
void countingSortCPU(int* array, int size, int digit) {
    int* output = new int[size];
    int count[10] = { 0 };

    // Count occurrences of each digit
    for (int i = 0; i < size; ++i) {
        count[(array[i] / digit) % 10]++;
    }

    // Perform exclusive scan on count array
    for (int i = 1; i < 10; ++i) {
        count[i] += count[i - 1];
    }

    // Build the output array
    for (int i = size - 1; i >= 0; --i) {
        output[count[(array[i] / digit) % 10] - 1] = array[i];
        count[(array[i] / digit) % 10]--;
    }

    // Copy the output array to the original array
    for (int i = 0; i < size; ++i) {
        array[i] = output[i];
    }

    delete[] output;
}

// Function to generate random numbers
void generateRandomArray(int* array, int size) {
    srand(time(nullptr));
    for (int i = 0; i < size; ++i) {
        array[i] = rand() % 1000; // Adjust the range as needed
    }
}

// Function to print array
void printArray(int* array, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    const int SIZE = 100000000; // Adjust the size as needed

    int* array_cpu = new int[SIZE];
    generateRandomArray(array_cpu, SIZE);
    int* array_gpu = new int[SIZE];
    memcpy(array_gpu, array_cpu, SIZE * sizeof(int));

    // Perform radix sort on CPU and measure time
    clock_t cpu_start = clock();
    for (int digit = 1; digit <= 1000; digit *= 10) {
        countingSortCPU(array_cpu, SIZE, digit);
    }
    clock_t cpu_end = clock();
    double cpu_time = double(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Perform radix sort on GPU and measure time
    clock_t gpu_start = clock();
    radixSortGPU(array_gpu, SIZE);
    clock_t gpu_end = clock();
    double gpu_time = double(gpu_end - gpu_start) / CLOCKS_PER_SEC;

    // Verify if the arrays are sorted correctly
    bool success = true;
    for (int i = 0; i < SIZE; ++i) {
        if (array_cpu[i] != array_gpu[i]) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Arrays are sorted correctly!" << std::endl;
    }
    else {
        std::cout << "Arrays are not sorted correctly!" << std::endl;
    }

    std::cout << "CPU Time: " << cpu_time << " seconds" << std::endl;
    std::cout << "GPU Time: " << gpu_time << " seconds" << std::endl;

    delete[] array_cpu;
    delete[] array_gpu;

    return 0;
}
