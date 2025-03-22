/* ------------------------------------------------------------------------------------------------
Alanna Koser, David Li, Jonah Kolar
CSS 535 A
March 23, 2025
Final Project: Optical Flow

Description:
Implements 1D spatial convolution operations across video frames for optical flow processing with
support for 2 GPU implementation: naive and shared memory. This includes CUDA kernels for efficient 
parallel processing on GPU hardware.
------------------------------------------------------------------------------------------------ */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "spatialConvolution.h"

/**
 * @brief CUDA kernel for naive parallel convolution
 *
 * @param input: Input image data
 * @param output: Output image data
 * @param kernel: Convolution kernel coefficients
 * @param width: Image width
 * @param height: Image height
 * @param kernelSize: Size of the convolution kernel
 * @param isX: Direction flag (true for X-axis, false for Y-axis)
 */
__global__ void spatialConvolveNaiveKernel(float* input, float* output, float* kernel, 
    int width, int height, int kernelSize, bool isX) {

    // Calculate global thread index
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int halfKernel = kernelSize / 2;

    // Check if index is within bounds
    if (idx >= width * height) return;

    // Convert 1D index to 2D coordinates
    int x, y;
    if (isX) {
        x = idx % width;  // Column index
        y = idx / width;  // Row index
    }
    else {
        y = idx % height;  // Row index
        x = idx / height;  // Column index
    }

    // Perform convolution
    float sum = 0.0f;
    for (int k = -halfKernel; k <= halfKernel; ++k) {
        int index;
        if (isX) { // x case
            index = x + k;
            // Border handling
            if (index < 0) {
                index = -index; // Reflect across border
            }
            else if (index >= width) {
                index = 2 * width - index - 2; // Reflect across opposite border
            }
            sum += input[y * width + index] * kernel[k + halfKernel];
        }
        else { // y case
            index = y + k;
            // Border handling
            if (index < 0) {
                index = -index; // Reflect across border
            }
            else if (index >= height) {
                index = 2 * height - index - 2; // Reflect across opposite border
            }
            sum += input[index * width + x] * kernel[k + halfKernel];
        }
    }
    output[idx] = sum;
}

/**
 * @brief CUDA kernel for optimized parallel convolution using shared memory for kernel only
 *
 * @param input: Input image data
 * @param output: Output image data
 * @param kernel: Convolution kernel coefficients
 * @param width: Image width
 * @param height: Image height
 * @param kernelSize: Size of the convolution kernel
 * @param isX: Direction flag (true for X-axis, false for Y-axis)
 */
__global__ void spatialConvolveSharedMemKernel(float* input, float* output, float* kernel,
    int width, int height, int kernelSize, bool isX)
{
    // Allocate shared memory dynamically
    extern __shared__ float s_kernel[];

    int halfKernel = kernelSize / 2;
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bdx = blockDim.x;

    // Calculate global thread index
    int globalIdx = bx * bdx + tx;

    // Convert 1D index to 2D coordinates
    int x, y;
    if (isX) {
        // Process by rows
        x = globalIdx % width;
        y = globalIdx / width;
    }
    else {
        // Process by columns - keep track of original linear index
        y = globalIdx % height;
        x = globalIdx / height;
    }

    // Check boundaries
    if (x >= width || y >= height) return;

    // Load kernel into shared memory (cooperatively)
    if (tx < kernelSize) {
        s_kernel[tx] = kernel[tx];
    }

    __syncthreads(); // Ensure all threads see the kernel data

    int sharedSize = bdx + kernelSize - 1;
    float sum = 0.0f;
    // X-direction convolution
    if (isX) {
        // X-direction convolution
        for (int k = 0; k < kernelSize; k++) {
            int sampleX = x + k - halfKernel;

            // Border Handling
            if (sampleX < 0) sampleX = -sampleX;
            if (sampleX >= width) sampleX = 2 * width - sampleX - 2;

            sum += input[y * width + sampleX] * s_kernel[k];
        }
    }
    else {
        // Y-direction convolution
        for (int k = 0; k < kernelSize; k++) {
            int sampleY = y + k - halfKernel;

            // Border Handling
            if (sampleY < 0) sampleY = -sampleY;
            if (sampleY >= height) sampleY = 2 * height - sampleY - 2;

            sum += input[sampleY * width + x] * s_kernel[k];
        }
    }

    // Write output
    output[y * width + x] = sum;
}

/**
 * @brief CUDA kernel for optimized parallel convolution using shared memory for both kernel and 
 *   frame tiles
 *
 * @param input: Input image data
 * @param output: Output image data
 * @param kernel: Convolution kernel coefficients
 * @param width: Image width
 * @param height: Image height
 * @param kernelSize: Size of the convolution kernel
 * @param isX: Direction flag (true for X-axis, false for Y-axis)
 */
__global__ void spatialConvolveSharedMemTileKernel(float* input, float* output, float* kernel,
    int width, int height, int kernelSize, bool isX) {

    // Allocate shared memory dynamically
    extern __shared__ float shared_mem[];

    // Partition shared memory: first part for kernel, second part for input data
    float* s_kernel = &shared_mem[0];
    float* s_data = &shared_mem[kernelSize];

    int halfKernel = kernelSize / 2;
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bdx = blockDim.x;

    // Calculate global thread index
    int globalIdx = bx * bdx + tx;

    // Convert 1D index to 2D coordinates
    int x, y;
    if (isX) {
        // Process by rows
        x = globalIdx % width;
        y = globalIdx / width;
    }
    else {
        // Process by columns - keep track of original linear index
        y = globalIdx % height;
        x = globalIdx / height;
    }

    // Check boundaries
    if (x >= width || y >= height) return;

    // Load kernel into shared memory (cooperatively)
    if (tx < kernelSize) {
        s_kernel[tx] = kernel[tx];
    }

    __syncthreads(); // Ensure all threads see the kernel data

    int sharedSize = bdx + kernelSize - 1;
    // X-direction convolution
    if (isX) {
        // Calculate block starting x-position (leftmost needed pixel)
        // Find which block of columns we're in
        int blockStartX = (bx * bdx) - halfKernel;
        // Adjust for the specific row
        while (blockStartX < 0) blockStartX += width;
        blockStartX = blockStartX % width;

        // Each thread loads one or more elements into shared memory
        for (int i = tx; i < sharedSize; i += bdx) {
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
            int s_idx = tx + k;
            sum += s_data[s_idx] * s_kernel[k];
        }

        output[globalIdx] = sum;
    }
    // Y-direction convolution
    else {
        // Each thread block handles a set of columns
        // Each thread processes a different row in the same column

        // Calculate block starting position (in terms of rows)
        int blockStartY = (bx * bdx) - halfKernel;

        // Adjust if negative
        while (blockStartY < 0) blockStartY += height;
        blockStartY = blockStartY % height;

        // Load data into shared memory
        for (int i = tx; i < sharedSize; i += bdx) {
            int loadY = blockStartY + i;
            
            // Handle wrap-around
            if (loadY >= height) loadY -= height;

            // Only load valid data for this column
            if (x < width) {
                s_data[i] = input[loadY * width + x];
            }
            else {
                s_data[i] = 0.0f; // Zero padding
            }
        }

        __syncthreads(); // Wait for all data to be loaded

        // Perform convolution
        float sum = 0.0f;
        for (int k = 0; k < kernelSize; k++) {
            // Map from global position to position in shared memory tile
            int s_idx = tx + k;
            sum += s_data[s_idx] * s_kernel[k];
        }
        output[globalIdx] = sum;
    }
}


/**
 * @brief Wrapper function that gets a function pointer to the naive CUDA kernel implementation
 *   because we cannot use function pointer directly from *.cpp file
 *
 * @return Function pointer to temporalConvolveNaiveKernel
 */
SpatialConvolveKernelPtr getSpatialConvolveNaiveKernel() {
    return spatialConvolveNaiveKernel;
}

/**
 * @brief Wrapper function that gets a function pointer to the shared memory CUDA kernel
 *   implementation that loads both the kernel and frame tiles into shared memory, because we 
 *   cannot use function pointer directly from *.cpp file
 *
 * @return Function pointer to temporalConvolveSharedMemKernel
 */
SpatialConvolveKernelPtr getSpatialConvolveSharedMemTileKernel() {
    return spatialConvolveSharedMemTileKernel;
}

/**
 * @brief Wrapper function that gets a function pointer to the shared memory CUDA kernel
 *   implementation that only loads the kernel into shared memory, because we cannot use 
 *   function pointer directly from *.cpp file
 *
 * @return Function pointer to temporalConvolveSharedMemKernel
 */
SpatialConvolveKernelPtr getSpatialConvolveSharedMemKernel() {
    return spatialConvolveSharedMemKernel;
}

/**
 * @brief Helper method to launch the CUDA kernel with appropriate parameters
 *
 * @param frame: Input image frame
 * @param kernel: Convolution kernel to apply
 * @param isX: Direction flag (true for X-axis, false for Y-axis)
 * @param blockSize: CUDA thread block size
 * @param convolveKernel: Function pointer to the CUDA kernel implementation
 * @return Mat: Convolved image
 */
Mat SpatialConvolution::launchConvolveKernel(const Mat& frame, const Kernel& kernel, bool isX,
    int blockSize, void (*convolveKernel)(float*, float*, float*, int, int, int, bool)) {

    // Memory pointers for GPU data
    float* d_input;
    float* d_output;
    float* d_kernel;

    // Calculate sizes
    int width = frame.cols;
    int height = frame.rows;
    int frameSize = width * height;
    int kernelSize = kernel.getSize();

    size_t frameBytes = frameSize * sizeof(float);
    size_t kernelBytes = kernelSize * sizeof(float);

    // Allocate memory on GPU
    cudaMalloc(&d_input, frameBytes);
    cudaMalloc(&d_output, frameBytes);
    cudaMalloc(&d_kernel, kernelBytes);

    // Copy data to GPU
    cudaMemcpy(d_input, frame.ptr<float>(), frameBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.getRawData(), kernelBytes, cudaMemcpyHostToDevice);

    // Use default block size if not specified
    if (blockSize == 0) {
        blockSize = 256;
    }

    // Calculate shared memory size:
    // kernelSize for the kernel + (blockSize + kernelSize - 1) for the data with halo regions
    int sharedMemSize;
    // Select implementation based on the configured strategy
    switch (implementation) {
    case Implementation::GPU_NAIVE:
        sharedMemSize = 0;
        break;
    case Implementation::GPU_SHARED_MEMORY:
        sharedMemSize = (kernelSize + blockSize + kernelSize - 1) * sizeof(float);
        break;
    case Implementation::GPU_SHARED_MEMORY_TILES:
        sharedMemSize = (kernelSize + blockSize + kernelSize - 1) * sizeof(float);
        break;
    default:
        throw std::invalid_argument("Unknown implementation");
    }

    // Calculate grid dimensions and launch kernel
    convolveKernel << <(frameSize + blockSize - 1) / blockSize, blockSize, sharedMemSize >> > (
        d_input, d_output, d_kernel, width, height, kernelSize, isX);

    cudaDeviceSynchronize(); // Ensure all GPU operations complete

    // Copy the result back to host
    Mat result(height, width, frame.type());
    cudaMemcpy(result.ptr<float>(), d_output, frameBytes, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return result;
}
