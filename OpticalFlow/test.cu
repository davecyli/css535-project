#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "profiler.h"

#include <stdio.h>
#include <iostream>

CudaProfiler ExampleProfiler = CudaProfiler();

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {  // Prevent out-of-bounds access
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Memory for benchmarking memcpy
    const size_t dataSize = 100 * 1024 * 1024; // 100 MB
    int* h_data, * d_data;

    // Allocate host and device memory
    cudaMallocHost((void**)&h_data, dataSize);
    cudaMalloc((void**)&d_data, dataSize);

    int result = ExampleProfiler.profileMemcpyHtoD(d_data, h_data, dataSize);
    if (result == -1) {
        fprintf(stderr, "cudaMemcpy HtoD failed!\n");
    }
    else {
        std::cout << "Elapsed time Host to Device for h_data and d_data: ";
        std::cout << result << " ms" << std::endl;
        double bandwidth = (dataSize / (result / 1000.0)) / (1024 * 1024 * 1024);
        std::cout << "Memcpy Host->Device Speed: " << bandwidth << " GB/s\n";
    }

    cudaFreeHost(h_data);
    cudaFree(d_data);

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!\n");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // Ensure proper cleanup and profiling completion
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size) {
    int result = 0;
    int* dev_a = nullptr;
    int* dev_b = nullptr;
    int* dev_c = nullptr;

    // Define the grid and block configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    cudaError_t cudaStatus;

    // Select the GPU
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! CUDA-capable GPU installed?\n");
        goto Error;
    }

    // Allocate GPU buffers for three vectors
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for dev_a failed!\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for dev_b failed!\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for dev_c failed!\n");
        goto Error;
    }

    // Copy input vectors to GPU
    result = ExampleProfiler.profileMemcpyHtoD(dev_a, a, size * sizeof(int));
    if (result == -1) {
        fprintf(stderr, "cudaMemcpy HtoD for a failed!\n");
        goto Error;
    }
    else {
        std::cout << "Elapsed time Host to Device for a: ";
        std::cout << result << " ms" << std::endl;
    }

    result = ExampleProfiler.profileMemcpyHtoD(dev_b, b, size * sizeof(int));
    if (result == -1) {
        fprintf(stderr, "cudaMemcpy HtoD for b failed!\n");
        goto Error;
    }
    else {
        std::cout << "Elapsed time Host to Device for b: ";
        std::cout << result << " ms" << std::endl;
    }

    // Start kernel execution and profile it
    ExampleProfiler.startTimer();
    addKernel <<<blocksPerGrid, threadsPerBlock>>> (dev_c, dev_a, dev_b, size);
    cudaDeviceSynchronize();
    std::cout << "addKernel elapsed time: ";
    std::cout << ExampleProfiler.stopTimer() << " ms" << std::endl;

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr,
            "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Ensure kernel execution completes
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr,
            "cudaDeviceSynchronize returned error %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory
    result = ExampleProfiler.profileMemcpyDtoH(c, dev_c, size * sizeof(int));
    if (result == -1) {
        fprintf(stderr, "cudaMemcpy DtoH for c failed!\n");
        goto Error;
    }
    else {
        std::cout << "Elapsed time Device to Host for c: ";
        std::cout << result << " ms" << std::endl;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
