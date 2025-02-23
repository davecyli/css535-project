#ifndef SPATIALCONVOLUTION_H
#define SPATIALCONVOLUTION_H

#include <opencv2/opencv.hpp>
#include "convolution.h"
#include "kernel.h"

using namespace cv;

class SpatialConvolution : public Convolution {
public:
    Mat convolveX(const Mat& frame, const Kernel& kernel);
    Mat convolveY(const Mat& frame, const Kernel& kernel);
private:
    Mat convolve(const Mat& frame, const Kernel& kernel, const bool isX);
};

#endif // SPATIALCONVOLUTION_H