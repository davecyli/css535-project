/*

*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "spatialConvolution.h"

__global__ void spatialConvolveNaiveKernel(float* input, float* output, float* kernel, 
    int width, int height, int kernelSize, bool isX) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int halfKernel = kernelSize / 2;

    if (idx >= width * height) return;

    int x, y;
    if (isX) {
        x = idx % width;  // Column index
        y = idx / width;  // Row index
    }
    else {
        y = idx % height;  // Row index
        x = idx / height;  // Column index
    }

    float sum = 0.0f;
    for (int k = -halfKernel; k <= halfKernel; ++k) {
        int index = isX ? (x + k) : (y + k);
        if (index >= 0 && (isX ? index < width : index < height)) {
            sum += input[(isX ? y * width + index : index * width + x)] * kernel[k + halfKernel];
        }
    }
    output[idx] = sum;
}

__global__ void spatialConvolveSharedMemKernel(float* input, float* output, float* kernel,
    int width, int height, int kernelSize, bool isX) {

    extern __shared__ float shared_mem[];

    // Partition shared memory: first part for kernel, second part for input data
    float* s_kernel = &shared_mem[0];
    float* s_data = &shared_mem[kernelSize];

    int halfKernel = kernelSize / 2;
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bdx = blockDim.x;

    // Global thread index
    int globalIdx = bx * bdx + tx;

    // Load kernel into shared memory (cooperatively)
    if (tx < kernelSize) {
        s_kernel[tx] = kernel[tx];
    }

    __syncthreads();

    // Calculate the starting position for shared memory loading
    int startX, startY;
    if (isX) {
        // For X convolution
        startX = (bx * bdx) % width - halfKernel;
        startY = (bx * bdx) / width;
    }
    else {
        // For Y convolution
        startX = (bx * bdx) / height;
        startY = (bx * bdx) % height - halfKernel;
    }

    // Number of elements to load into shared memory
    int sharedSize = bdx + 2 * halfKernel;

    if (isX) {
        // Load horizontal strip (for X convolution)
        for (int i = tx; i < sharedSize; i += bdx) {
            int loadCol = startX + i;
            int loadRow = startY;

            // Handle boundary conditions
            if (loadRow >= 0 && loadRow < height && loadCol >= 0 && loadCol < width) {
                s_data[i] = input[loadRow * width + loadCol];
            }
            else {
                s_data[i] = 0.0f; // Zero padding
            }
        }
    }
    else {
        // Load vertical strip (for Y convolution)
        for (int i = tx; i < sharedSize; i += bdx) {
            int loadCol = startX;
            int loadRow = startY + i;

            // Handle boundary conditions
            if (loadRow >= 0 && loadRow < height && loadCol >= 0 && loadCol < width) {
                s_data[i] = input[loadRow * width + loadCol];
            }
            else {
                s_data[i] = 0.0f; // Zero padding
            }
        }
    }

    __syncthreads(); // Ensure all data is loaded before computation

    // Check if this thread is within image bounds
    if (globalIdx >= width * height || x >= width || y >= height) return;

    // Perform convolution using shared memory
    float sum = 0.0f;

    if (isX) {
        // For X direction convolution
        // Find the position of this thread's data in shared memory
        // This accounts for the halo region
        int s_pos = threadIdx.x;

        // Apply convolution
        for (int k = 0; k < kernelSize; k++) {
            sum += s_data[s_pos + k] * s_kernel[k];
        }
    }
    else {
        // For Y direction convolution
        // Handle the case where threads in a block access different rows but same column
        int s_pos = threadIdx.x;

        // Apply convolution
        for (int k = 0; k < kernelSize; k++) {
            sum += s_data[s_pos + k] * s_kernel[k];
        }
    }

    // Determine x, y coordinates based on direction of convolution
    int x, y;
    if (isX) {
        x = globalIdx % width;
        y = globalIdx / width;
    }
    else {
        y = globalIdx % height;
        x = globalIdx / height;
    }

    // Write result to global memory
    if (x < width && y < height) {
        output[y * width + x] = sum;
    }
    else {
        output[y * width + x] = 0;  // Prevent uninitialized access
    }
}

SpatialConvolveKernelPtr getSpatialConvolveNaiveKernel() {
    return spatialConvolveNaiveKernel;
}

SpatialConvolveKernelPtr getSpatialConvolveSharedMemKernel() {
    return spatialConvolveSharedMemKernel;
}

Mat SpatialConvolution::launchConvolveKernel(const Mat& frame, const Kernel& kernel, bool isX,
    int blockSize, void (*convolveKernel)(float*, float*, float*, int, int, int, bool)) {

    if (frame.empty()) return frame;

    Mat converted = convert(frame);

    // Allocate memory on GPU for the input frame
    float* d_input;
    float* d_output;
    float* d_kernel;

    int width = converted.cols;
    int height = converted.rows;
    int frameSize = width * height;
    int kernelSize = kernel.getSize();

    size_t frameBytes = frameSize * sizeof(float);
    size_t kernelBytes = kernelSize * sizeof(float);

    // Allocate memory on GPU
    cudaMalloc(&d_input, frameBytes);
    cudaMalloc(&d_output, frameBytes);
    cudaMalloc(&d_kernel, kernelBytes);

    // Copy data to GPU
    cudaMemcpy(d_input, converted.ptr<float>(), frameBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.getRawData(), kernelBytes, cudaMemcpyHostToDevice);

    // Use default block size if not specified
    if (blockSize == 0) {
        blockSize = 256;
    }

    // Calculate shared memory size:
    // kernelSize for the kernel +
    // (blockSize + kernelSize - 1) for the data with halo regions
    int sharedMemSize = (kernelSize + blockSize + kernelSize - 1) * sizeof(float);

    // Launch kernel
    convolveKernel << <(frameSize + blockSize - 1) / blockSize, blockSize, sharedMemSize >> > (
        d_input, d_output, d_kernel, width, height, kernelSize, isX);

    cudaDeviceSynchronize();

    // Copy the result back to host
    Mat result(height, width, converted.type());
    cudaMemcpy(result.ptr<float>(), d_output, frameBytes, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return result;
}
