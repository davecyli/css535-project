/* ------------------------------------------------------------------------------------------------
Alanna Koser, David Li, Jonah Kolar
CSS 535 A
March 23, 2025
Final Project: Optical Flow

Description:
Implements temporal convolution operations across video frames for optical flow processing with
support for both CPU and GPU implementations
------------------------------------------------------------------------------------------------ */
#include "temporalConvolution.h"
#include <opencv2/core/cuda.hpp>
#include "cuda_runtime.h"

/**
 * @brief Constructor for TemporalConvolution
 *
 * @param impl: The implementation strategy to use
 * @param kernelSize: The number of frames to use in the temporal convolution
 */
TemporalConvolution::TemporalConvolution(Implementation impl, int kernelSize) : 
    Convolution(impl), cpuBuffer(kernelSize), gpuBuffer(kernelSize) {}

/**
 * @brief Destructor for cleaning up GPU resources
 */
TemporalConvolution::~TemporalConvolution() {
    // Free CUDA device memory
    if (d_frames) cudaFree(d_frames);
    if (d_convolved) cudaFree(d_convolved);
    if (d_kernel) cudaFree(d_kernel);
    // Reset pointers to avoid double-free issues
    d_frames = nullptr;
    d_convolved = nullptr;
    d_kernel = nullptr;
}

/**
 * @brief CPU-based temporal convolution implementation
 *
 * @param frame: The current input frame
 * @param kernel: The convolution kernel to apply
 * @return Convolved result frame
 */
Mat TemporalConvolution::cpuConvolve(const Mat& frame, const Kernel& kernel) {
    // Add to buffer
    cpuBuffer.enqueue(frame);

    Mat convolved;
    if (cpuBuffer.isFull()) {
        // Initialize output frame with zeros
        convolved = Mat::zeros(frame.size(), frame.type());

        // Apply weighted sum across the frame buffer
        int index = 0;
        for (int i = 0; i < cpuBuffer.getSize(); i++) {
            convolved += cpuBuffer[i] * kernel.getElement(index++);
        }
    }

    return convolved;
}

/**
 * @brief Calls convolve method with custom block size
 *
 * @param frame: The current input frame
 * @param kernel: The convolution kernel to apply
 * @param blockSize: CUDA block size for GPU implementations
 * @return Convolved result frame
 */
Mat TemporalConvolution::convolve(const Mat& frame, const Kernel& kernel) {
    return convolve(frame, kernel, 0);
}

/**
 * @brief Selects which convolution method to use: CPU naive, GPU naive, or GPU shared memory
 *
 * @param frame: The current input frame
 * @param kernel: The convolution kernel to apply
 * @param blockSize: CUDA block size for GPU implementations
 * @return Convolved result frame
 */
Mat TemporalConvolution::convolve(const Mat& frame, const Kernel& kernel, int blockSize) {

    if (frame.empty()) return frame;

    // Convert to grayscale and float format
    Mat converted = convert(frame);

    // Dispatch to appropriate implementation
    switch (implementation) {
    case Implementation::CPU_NAIVE:
        return cpuConvolve(converted, kernel);
    case Implementation::GPU_NAIVE:
        return gpuConvolve(converted, kernel, blockSize, false, getTemporalConvolveNaiveKernel());
    case Implementation::GPU_SHARED_MEMORY:
        return gpuConvolve(converted, kernel, blockSize, true,
            getTemporalConvolveSharedMemKernel());
    default:
        throw std::invalid_argument("Unknown implementation");
    }
}

/**
 * @brief Wrapper function for kernel launcher, GPU implementation
 *
 * @param frame: The current input frame
 * @param kernel: The convolution kernel to apply
 * @param blockSize: CUDA block size
 * @param useSharedMemory: Flag indicating whether to use shared memory optimizations
 * @param convolveKernel: Function pointer to the CUDA kernel implementation
 * @return Convolved result frame
 */
Mat TemporalConvolution::gpuConvolve(const Mat& frame, const Kernel& kernel, 
    int blockSize, const bool useSharedMemory, 
    void (*convolveKernel)(float*, float*, float*, int, int, int)) {

    // Kernel launcher
    return launchConvolveKernel(frame, kernel, blockSize, useSharedMemory, convolveKernel);
}