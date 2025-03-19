#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <opencv2/opencv.hpp>
#include "kernel.h"

using namespace cv;

enum class Implementation {
    CPU_NAIVE,
    GPU_NAIVE,
    GPU_SHARED_MEMORY
};

class Convolution {
public:
    Convolution() : implementation(Implementation::GPU_SHARED_MEMORY) {}
    virtual ~Convolution() { freeGPUResources(); }
    explicit Convolution(Implementation impl) : implementation(impl) {}
protected:
    Mat convert(const Mat& frame) const;
    Implementation implementation;

    float* d_frameData = nullptr;
    float* d_convolved = nullptr;
    float* d_kernel = nullptr;
    bool isInitialized = false;

    void freeGPUResources();
};

#endif // CONVOLUTION_H