#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <opencv2/opencv.hpp>
#include "kernel.h"

using namespace cv;

class Convolution {
protected:
    Mat convert(const Mat& frame) const;
};

#endif // CONVOLUTION_H