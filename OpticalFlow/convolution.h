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
    explicit Convolution(Implementation impl) : implementation(impl) {}
protected:
    Mat convert(const Mat& frame) const;
    Implementation implementation;

    void freeGPUResources();
};

#endif // CONVOLUTION_H