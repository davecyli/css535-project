#ifndef TEMPORALCONVOLUTION_H
#define TEMPORALCONVOLUTION_H

#include <opencv2/opencv.hpp>
#include "convolution.h"
#include "kernel.h"
#include "circularBuffer.h"

using namespace cv;

class TemporalConvolution : public Convolution {
public:
    TemporalConvolution(Implementation impl, int kernelSize);
    ~TemporalConvolution();

    Mat convolve(const Mat& frame, const Kernel& kernel);
private:
    CircularBuffer<Mat> cpuBuffer;
    CircularBuffer<float*> gpuBuffer;

    float* d_frames = nullptr;
    float* d_convolved = nullptr;
    float* d_kernel = nullptr;

    bool isFirstPass = true;
    int frameIndex = 0;

    Mat cpuConvolve(const Mat& frame, const Kernel& kernel);
    Mat gpuConvolveNaive(const Mat& frame, const Kernel& kernel);

    Mat launchConvolveNaiveKernel(const Mat& frame, const Kernel& kernel);
};

#endif // TEMPORALCONVOLUTION_H