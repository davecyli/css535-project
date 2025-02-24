#include "spatialConvolution.h"

Mat SpatialConvolution::convolveX(const Mat& frame, const Kernel& kernel) {
    Mat converted = convert(frame);
    return convolve(converted, kernel, true);
}

Mat SpatialConvolution::convolveY(const Mat& frame, const Kernel& kernel) {
    Mat converted = convert(frame);
    return convolve(converted, kernel, false);
}

Mat SpatialConvolution::convolve(const Mat& frame, const Kernel& kernel, const bool isX) {
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
                    sum += inputFrame.at<float>(i, idx) * kernel.get(k + halfKernel);
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