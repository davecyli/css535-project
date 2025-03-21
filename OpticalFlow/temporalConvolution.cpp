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
    return convolve(frame, kernel, 0, 0);
}

Mat TemporalConvolution::convolve(const Mat& frame, const Kernel& kernel, int blockSize) {
    return convolve(frame, kernel, blockSize, 0);
}

Mat TemporalConvolution::convolve(const Mat& frame, const Kernel& kernel, 
    int blockSize, int tileSize) {

    switch (implementation) {
    case Implementation::CPU_NAIVE:
        return cpuConvolve(frame, kernel);
    case Implementation::GPU_NAIVE:

        return gpuConvolve(frame, kernel, blockSize, 0, false, getTemporalConvolveNaiveKernel());
    case Implementation::GPU_SHARED_MEMORY:
        return gpuConvolve(frame, kernel, blockSize, tileSize, true,
            getTemporalConvolveSharedMemKernel());
    default:
        throw std::invalid_argument("Unknown implementation");
    }
}

Mat TemporalConvolution::gpuConvolve(const Mat& frame, const Kernel& kernel, 
    int blockSize, const int tileSize, const bool useSharedMemory, 
    void (*convolveKernel)(float*, float*, float*, int, int, int, int)) {

    if (frame.empty()) return frame;

    // Call the wrapper function
    return launchConvolveKernel(frame, kernel, blockSize, tileSize, useSharedMemory, convolveKernel);
}