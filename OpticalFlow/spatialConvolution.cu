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

    // Load kernel into shared memory (cooperatively)
    if (tx < kernelSize) {
        s_kernel[tx] = kernel[tx];
    }

    __syncthreads();

    // Number of elements to load into shared memory
    int sharedSize = bdx + 2 * halfKernel;

    // X-direction convolution
    if (isX) {
        // Calculate block starting x-position (leftmost needed pixel)
        // Find which block of columns we're in
        int blockStartX = (blockIdx.x * blockDim.x) - halfKernel;
        // Adjust for the specific row
        while (blockStartX < 0) blockStartX += width;
        blockStartX = blockStartX % width;

        // Each thread loads one or more elements into shared memory
        int sharedSize = blockDim.x + kernelSize - 1;
        for (int i = threadIdx.x; i < sharedSize; i += blockDim.x) {
            int loadX = blockStartX + i;
            // Handle wrap-around for horizontal convolution
            if (loadX >= width) loadX -= width;

            // Only load valid data for this row
            if (y < height) {
                s_data[i] = input[y * width + loadX];
            }
            else {
                s_data[i] = 0.0f;
            }
        }

        __syncthreads(); // Wait for all data to be loaded

        // Perform convolution using shared memory
        float sum = 0.0f;
        for (int k = 0; k < kernelSize; k++) {
            // Map from global position to position in shared memory tile
            int s_x = threadIdx.x + halfKernel;
            sum += s_data[s_x - halfKernel + k] * s_kernel[k];
        }

        output[globalIdx] = sum;
    }
    // Y-direction convolution
    else {
        // Each thread block handles a set of columns
        // Each thread processes a different row in the same column

        // Calculate block starting position (in terms of rows)
        int blockStartY = (blockIdx.x * blockDim.x) - halfKernel;
        // Adjust if negative
        while (blockStartY < 0) blockStartY += height;
        blockStartY = blockStartY % height;

        // Load data into shared memory
        int sharedSize = blockDim.x + kernelSize - 1;
        for (int i = threadIdx.x; i < sharedSize; i += blockDim.x) {
            int loadY = blockStartY + i;
            // Handle wrap-around
            if (loadY >= height) loadY -= height;

            // Only load valid data for this column
            if (x < width) {
                s_data[i] = input[loadY * width + x];
            }
            else {
                s_data[i] = 0.0f;
            }
        }

        __syncthreads(); // Wait for all data to be loaded

        // Perform convolution
        float sum = 0.0f;
        for (int k = 0; k < kernelSize; k++) {
            // Map from global position to position in shared memory tile
            int s_y = threadIdx.x + halfKernel;
            sum += s_data[s_y - halfKernel + k] * s_kernel[k];
        }

        output[globalIdx] = sum;
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
