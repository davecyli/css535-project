#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <opencv2/opencv.hpp>
#include "kernel.h"

using namespace cv;
using namespace std;

class Convolution {
public:
    Mat convolveXY(const Mat& frame, const Kernel& kernel);
    Mat convolveT(const Kernel& kernel, const deque<Mat>& frames);
};

#endif // CONVOLUTION_H