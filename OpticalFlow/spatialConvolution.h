#ifndef SPATIALCONVOLUTION_H
#define SPATIALCONVOLUTION_H

#include <opencv2/opencv.hpp>
#include "convolution.h"
#include "kernel.h"

using namespace cv;

class SpatialConvolution : public Convolution {
public:
    explicit SpatialConvolution(Implementation impl) : Convolution(impl) {}
    Mat convolveX(const Mat& frame, const Kernel& kernel);
    Mat convolveY(const Mat& frame, const Kernel& kernel);
private:
    Mat convolve(const Mat& frame, const Kernel& kernel, bool isX);
    Mat cpuConvolve(const Mat& frame, const Kernel& kernel, bool isX);
    Mat gpuConvolveNaive(const Mat& frame, const Kernel& kernel, bool isX);
    Mat launchConvolveNaiveKernel(const Mat& frame, const Kernel& kernel);
    // Mat gpuConvolveSharedMemory(const Mat& frame, const Kernel& kernel, bool isX) const override;
};

#endif // SPATIALCONVOLUTION_H