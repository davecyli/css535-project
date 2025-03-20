#include "temporalConvolution.h"
#include <opencv2/core/cuda.hpp>
#include "cuda_runtime.h"

TemporalConvolution::TemporalConvolution(Implementation impl, int kernelSize) : 
    Convolution(impl), cpuBuffer(kernelSize), gpuBuffer(kernelSize) {}

TemporalConvolution::~TemporalConvolution() {
    if (d_frames) cudaFree(d_frames);
    if (d_convolved) cudaFree(d_convolved);
    if (d_kernel) cudaFree(d_kernel);
    d_frames = nullptr;
    d_convolved = nullptr;
    d_kernel = nullptr;
}

Mat TemporalConvolution::cpuConvolve(const Mat& frame, const Kernel& kernel) {
    if (frame.empty()) return frame;

    Mat converted = convert(frame);

    // Add to buffer
    cpuBuffer.enqueue(converted);

    Mat convolved;
    if (cpuBuffer.isFull()) {
        convolved = Mat::zeros(converted.size(), converted.type());
        int index = 0;
        for (int i = 0; i < cpuBuffer.getSize(); i++) {
            convolved += cpuBuffer[i] * kernel.getElement(index++);
        }
    }

    return convolved;
}

Mat TemporalConvolution::convolve(const Mat& frame, const Kernel& kernel) {
    switch (implementation) {
    case Implementation::CPU_NAIVE:
        return cpuConvolve(frame, kernel);
    case Implementation::GPU_NAIVE:
        return gpuConvolveNaive(frame, kernel);
        //case Implementation::GPU_SHARED_MEMORY:
        //    return gpuConvolveSharedMemory(frame, kernel, isX);
    default:
        throw std::invalid_argument("Unknown implementation");
    }
}

Mat TemporalConvolution::gpuConvolveNaive(const Mat& frame, const Kernel& kernel) {
    if (frame.empty()) return frame;

    // Call the wrapper function
    return launchConvolveNaiveKernel(frame, kernel);
}