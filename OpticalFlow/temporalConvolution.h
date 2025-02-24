#ifndef TEMPORALCONVOLUTION_H
#define TEMPORALCONVOLUTION_H

#include <opencv2/opencv.hpp>
#include "convolution.h"
#include "kernel.h"
#include "circularBuffer.h"

using namespace cv;

class TemporalConvolution : public Convolution {
public:
    TemporalConvolution(int kernelSize);
    Mat convolve(const Mat& frame, const Kernel& kernel);
private:
    CircularBuffer buffer;
};

#endif // TEMPORALCONVOLUTION_H