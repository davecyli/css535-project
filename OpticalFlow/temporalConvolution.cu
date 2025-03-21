/* ------------------------------------------------------------------------------------------------
Alanna Koser, David Li, Jonah Kolar
CSS 535 A
March 23, 2025
Final Project: Optical Flow
Description:
CUDA kernel implementations for temporal convolution operations with 2 different versions: naive 
and shared memory.
------------------------------------------------------------------------------------------------ */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "temporalConvolution.h"

/**
 * @brief CUDA kernel for naive temporal convolution
 *   Each thread handles one pixel position across all frames in the temporal window
 *
 * @param d_frames: Device pointer to frame data
 * @param d_output: Device pointer to output data
 * @param d_kernel: Device pointer to kernel data
 * @param frameSize: Total number of pixels in a frame
 * @param bufferSize: Number of frames in the buffer
 * @param currentIndex: Current position in the circular buffer
 */
__global__ void temporalConvolveNaiveKernel(
    float* d_frames, float* d_output, float* d_kernel,
    int frameSize, int bufferSize, int currentIndex)
{
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frameSize) return;

    float sum = 0.0f;
    // Iterate through all frames in the buffer
    for (int i = 0; i < bufferSize; i++) {
        // Compute correct index in circular buffer
        int frameIdx = (currentIndex + bufferSize - i) % bufferSize;
        // Accumulate weighted value from each frame
        sum += d_frames[frameIdx * frameSize + idx] * d_kernel[i];
    }
    // Store result
    d_output[idx] = sum;
}

/**
 * @brief CUDA kernel for shared memory optimized temporal convolution
 *   Uses shared memory to cache kernel coefficients and pixel values for better memory access 
 *   patterns and reduced global memory traffic
 *
 * @param d_frames: Device pointer to frame data
 * @param d_output: Device pointer to output data
 * @param d_kernel: Device pointer to kernel data
 * @param frameSize: Total number of pixels in a frame
 * @param bufferSize: Number of frames in the buffer
 * @param currentIndex: Current position in the circular buffer
 */
__global__ void temporalConvolveSharedMemKernel(
    float* d_frames, float* d_output, float* d_kernel,
    int frameSize, int bufferSize, int currentIndex)
{
    // Allocate shared memory for both kernel and frame data samples
    extern __shared__ float sharedMem[];

    // Split shared memory: first part for kernel, second part for frame samples
    float* s_kernel = sharedMem;
    float* s_frames = &sharedMem[bufferSize];

    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frameSize) return;

    // Cooperatively load kernel into shared memory
    if (threadIdx.x < bufferSize) {
        s_kernel[threadIdx.x] = d_kernel[threadIdx.x];
    }

    // Each thread loads its own samples from the different frames
    for (int i = 0; i < bufferSize; i++) {
        int frameIdx = (currentIndex + bufferSize - i) % bufferSize;
        s_frames[threadIdx.x * bufferSize + i] = d_frames[frameIdx * frameSize + idx];
    }

    __syncthreads(); // Ensure all data is loaded before computation

    // Compute convolution using shared memory
    float sum = 0.0f;
    for (int i = 0; i < bufferSize; i++) {
        sum += s_frames[threadIdx.x * bufferSize + i] * s_kernel[i];
    }

    // Store result
    d_output[idx] = sum;
}

/**
 * @brief Wrapper function that gets a function pointer to the naive CUDA kernel implementation 
 *   because we cannot use function pointer directly from *.cpp file
 *
 * @return Function pointer to temporalConvolveNaiveKernel
 */
TemporalConvolveKernelPtr getTemporalConvolveNaiveKernel() {
    return temporalConvolveNaiveKernel;
}


/**
 * @brief Wrapper function that gets a function pointer to the shared memory CUDA kernel 
 *   implementation because we cannot use function pointer directly from *.cpp file
 *
 * @return Function pointer to temporalConvolveSharedMemKernel
 */
TemporalConvolveKernelPtr getTemporalConvolveSharedMemKernel() {
    return temporalConvolveSharedMemKernel;
}

/**
 * @brief Helper method to launch CUDA kernel for temporal convolution
 *   Handles memory management, data transfers, and kernel launch configuration
 *
 * @param frame: The current input frame
 * @param kernel: The convolution kernel to apply
 * @param blockSize: CUDA block size
 * @param useSharedMemory: Flag indicating whether to use shared memory optimizations
 * @param convolveKernel: Function pointer to the CUDA kernel implementation
 * @return Convolved result frame
 */
Mat TemporalConvolution::launchConvolveKernel(const Mat& frame, const Kernel& kernel, 
    int blockSize, const bool useSharedMemory,
    void (*convolveKernel)(float*, float*, float*, int, int, int)) {

    size_t frameSize = frame.rows * frame.cols; // Number of float elements
    size_t frameBytes = frameSize * sizeof(float);

    // Create a temporary buffer for the converted frame
    float* floatFrame = new float[frameSize];
    memcpy(floatFrame, frame.ptr<float>(), frameBytes);

    // Add to GPU buffer
    gpuBuffer.enqueue(floatFrame, frameSize);

    // Return empty result if buffer isn't full yet
    if (!gpuBuffer.isFull()) {
        delete[] floatFrame;  // Clean up temporary buffer
        return Mat();
    }

    // Initialize GPU resources on first pass
    if (isFirstPass) {
        // Allocate device memory for circular buffer of frames
        cudaMalloc(&d_frames, frameBytes * kernel.getSize());
        cudaMemset(d_frames, 0, frameBytes * kernel.getSize());

        // Allocate device memory for convolution result
        cudaMalloc(&d_convolved, frameBytes);
        cudaMemset(d_convolved, 0, frameBytes);

        // Allocate and copy kernel to device
        cudaMalloc(&d_kernel, kernel.getSize() * sizeof(float));
        cudaMemcpy(d_kernel, kernel.getRawData(), kernel.getSize() * sizeof(float), cudaMemcpyHostToDevice);

        isFirstPass = false;
    }

    // Compute the offset where the new frame should go
    int offset = frameIndex * frameSize;

    // Copy the new frame into the correct position in circular buffer
    cudaMemcpy(d_frames + offset, floatFrame, frameBytes, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    // Increment the frame index circularly
    frameIndex = (frameIndex + 1) % gpuBuffer.getSize();

    // Use default block size if not specified
    if (blockSize == 0) {
        blockSize = 256;
    } 

    // Calculate shared memory size if using shared memory
    int sharedMemSize = 0;
    if (useSharedMemory) {
        sharedMemSize = kernel.getSize() * sizeof(float) +
            blockSize * kernel.getSize() * sizeof(float);
    }

    // Launch kernel with calculated grid and block configuration
    convolveKernel << <(frameSize + blockSize - 1) / blockSize, blockSize, sharedMemSize >> > (
        d_frames, d_convolved, d_kernel, frameSize, gpuBuffer.getSize(), frameIndex);

    // Download result from device to host
    Mat result(frame.size(), CV_32F);
    cudaMemcpy(result.ptr<float>(), d_convolved, frameBytes, cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] floatFrame;

    return result;
}