#include "convolution.h"
#include "cuda_runtime.h"

Mat Convolution::convert(const Mat& frame) const {
    if (frame.empty()) return frame;

    // Copy and convert to grayscale and float if needed
    Mat converted;

    if (frame.channels() == 3) {
        cvtColor(frame, converted, COLOR_BGR2GRAY);
        converted.convertTo(converted, CV_32F, 1.0 / 255.0);
    }
    else {
        converted = frame.clone();
    }

    return converted;
}

void Convolution::freeGPUResources() {
    if (d_frameData) {
        cudaFree(d_frameData);
        d_frameData = nullptr;
    }
    if (d_convolved) {
        cudaFree(d_convolved);
        d_convolved = nullptr;
    }
    if (d_kernel) {
        cudaFree(d_kernel);
        d_kernel = nullptr;
    }
    isInitialized = false;
}