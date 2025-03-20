#include "spatialConvolution.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace cv;

Mat SpatialConvolution::convolveX(const Mat& frame, const Kernel& kernel) {
    Mat converted = convert(frame);
    return convolve(converted, kernel, true, 0);
}

Mat SpatialConvolution::convolveX(const Mat& frame, const Kernel& kernel, int blockSize) {
    Mat converted = convert(frame);
    return convolve(converted, kernel, true, blockSize);
}

Mat SpatialConvolution::convolveY(const Mat& frame, const Kernel& kernel) {
    Mat converted = convert(frame);
    return convolve(converted, kernel, false, 0);
}

Mat SpatialConvolution::convolveY(const Mat& frame, const Kernel& kernel, int blockSize) {
    Mat converted = convert(frame);
    return convolve(converted, kernel, false, blockSize);
}

Mat SpatialConvolution::cpuConvolve(const Mat& frame, const Kernel& kernel, const bool isX) {

    if (frame.empty()) return frame;

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
            convolved.at<float>(i, j) = sum; // Store
        }
    }

    // Transpose back if needed
    if (!isX) {
        transpose(convolved, convolved);
    }

    return convolved;
}

Mat SpatialConvolution::gpuConvolveNaive(const Mat& frame, const Kernel& kernel, bool isX, int blockSize){
    if (frame.empty()) return frame;

    Mat result;

    if (isX) {
        // Apply convolution directly in the X direction
        result = launchConvolveNaiveKernel(frame, kernel, true, blockSize);
    }
    else {
        // Transpose the frame, convolve in X (now acting as Y), then transpose back
        Mat transposed;
        cv::transpose(frame, transposed);  // Swap rows and columns

        Mat convolvedTransposed = launchConvolveNaiveKernel(transposed, kernel, true, blockSize);

        // Transpose back to restore original orientation
        cv::transpose(convolvedTransposed, result);
    }

    // Call the wrapper function
    return result;
}

Mat SpatialConvolution::convolve(const Mat& frame, const Kernel& kernel, bool isX, int blockSize) {
    switch (implementation) {
    case Implementation::CPU_NAIVE:
        return cpuConvolve(frame, kernel, isX);
    case Implementation::GPU_NAIVE:
        return gpuConvolveNaive(frame, kernel, isX, blockSize);
        //case Implementation::GPU_SHARED_MEMORY:
        //    return gpuConvolveSharedMemory(frame, kernel, isX);
    default:
        throw std::invalid_argument("Unknown implementation");
    }
}