/* ------------------------------------------------------------------------------------------------
Alanna Koser, David Li, Jonah Kolar
CSS 535 A
March 23, 2025
Final Project: Optical Flow

Description:
Implements 1D spatial convolution operations across video frames for optical flow processing with
support for both CPU and GPU implementations. This includes CUDA kernels for efficient parallel 
processing on GPU hardware.
------------------------------------------------------------------------------------------------ */
#include "spatialConvolution.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace cv;

/**
 * @brief Calls convolve method with appropriate inputs for X convolution
 *   Used for either CPU implementation or GPU implementation with default block size
 *
 * @param frame: Input image frame
 * @param kernel: Convolution kernel to apply
 * @return Mat: Convolved image
 */
Mat SpatialConvolution::convolveX(const Mat& frame, const Kernel& kernel) {
    return convolve(frame, kernel, true, 0);
}

/**
 * @brief Calls convolve method with appropriate inputs for X convolution
 *   Used for GPU implementation with specified block size
 *
 * @param frame: Input image frame
 * @param kernel: Convolution kernel to apply
 * @param blockSize: CUDA thread block size
 * @return Mat: Convolved image
 */
Mat SpatialConvolution::convolveX(const Mat& frame, const Kernel& kernel, int blockSize) {
    return convolve(frame, kernel, true, blockSize);
}

/**
 * @brief Calls convolve method with appropriate inputs for Y convolution
 *   Used for either CPU implementation or GPU implementation with default block size
 *
 * @param frame: Input image frame
 * @param kernel: Convolution kernel to apply
 * @return Mat: Convolved image
 */
Mat SpatialConvolution::convolveY(const Mat& frame, const Kernel& kernel) {
    return convolve(frame, kernel, false, 0);
}

/**
 * @brief Calls convolve method with appropriate inputs for Y convolution
 *   Used for GPU implementation with specified block size
 *
 * @param frame: Input image frame
 * @param kernel: Convolution kernel to apply
 * @param blockSize: CUDA thread block size
 * @return Mat: Convolved image
 */
Mat SpatialConvolution::convolveY(const Mat& frame, const Kernel& kernel, int blockSize) {
    return convolve(frame, kernel, false, blockSize);
}

/**
 * @brief CPU implementation of 1D convolution
 *
 * @param frame: Input image frame
 * @param kernel: Convolution kernel to apply
 * @param isX: Direction flag (true for X-axis, false for Y-axis)
 * @return Mat: Convolved image
 */
Mat SpatialConvolution::cpuConvolve(const Mat& frame, const Kernel& kernel, const bool isX) {

    int rows = frame.rows;
    int cols = frame.cols;
    int halfKernel = kernel.getSize() / 2; // For indexing

    Mat inputFrame = frame;
    // Transpose for Y convolution
    if (!isX) {
        transpose(inputFrame, inputFrame);
        std::swap(rows, cols);
    }

    // Output matrix
    Mat convolved(rows, cols, frame.type());

    // Iterate over all pixels in the image
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float sum = 0.0f;
            // Apply convolution kernel
            for (int k = -halfKernel; k <= halfKernel; ++k) {
                int idx = j + k; // Compute the shifted index
                if (idx >= 0 && idx < cols) { // Make sure index is within bounds
                    sum += inputFrame.at<float>(i, idx) * kernel.getElement(k + halfKernel);
                }
            }
            convolved.at<float>(i, j) = sum; // Store result
        }
    }

    // Transpose back if needed
    if (!isX) {
        transpose(convolved, convolved);
    }

    return convolved;
}

/**
 * @brief Wrapper function for kernel launcher, GPU implementation
 *
 * @param frame: Input image frame
 * @param kernel: Convolution kernel to apply
 * @param isX: Direction flag (true for X-axis, false for Y-axis)
 * @param blockSize: CUDA thread block size
 * @param convolveKernel: Function pointer to the CUDA kernel implementation
 * @return Mat: Convolved image
 */
Mat SpatialConvolution::gpuConvolve(const Mat& frame, const Kernel& kernel, bool isX, 
    int blockSize, void (*convolveKernel)(float*, float*, float*, int, int, int, bool))
{
    Mat result;

    if (isX) {
        // Call the wrapper function
        result = launchConvolveKernel(frame, kernel, true, blockSize, convolveKernel);
    }
    else {
        // Transpose the frame, convolve in X (now acting as Y), then transpose back
        Mat originalTransposed;
        cv::transpose(frame, originalTransposed);  // Swap rows and columns

        Mat convolvedTransposed = launchConvolveKernel(originalTransposed, kernel, true, blockSize,
            convolveKernel);

        // Transpose back to restore original orientation
        cv::transpose(convolvedTransposed, result);
    }
    return result;
}

/**
 * @brief Selects which convolution method to use: CPU naive, GPU naive, or GPU shared memory
 *
 * @param frame: Input image frame
 * @param kernel: Convolution kernel to apply
 * @param isX: Direction flag (true for X-axis, false for Y-axis)
 * @param blockSize: CUDA thread block size
 * @return Mat: Convolved image
 */
Mat SpatialConvolution::convolve(const Mat& frame, const Kernel& kernel, bool isX, int blockSize) {
    if (frame.empty()) return frame;

    // convert to grayscale and float
    Mat converted = convert(frame);

    // Select implementation based on the configured strategy
    switch (implementation) {
    case Implementation::CPU_NAIVE:
        return cpuConvolve(converted, kernel, isX);
    case Implementation::GPU_NAIVE:
        return gpuConvolve(converted, kernel, isX, blockSize, getSpatialConvolveNaiveKernel());
    case Implementation::GPU_SHARED_MEMORY:
        return gpuConvolve(converted, kernel, isX, blockSize, getSpatialConvolveSharedMemKernel());
    default:
        throw std::invalid_argument("Unknown implementation");
    }
}