#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <opencv2/opencv.hpp>
#include "kernel.h"
#include "circularBuffer.h"

using namespace cv;
using namespace std;

class Convolution {
public:
    Convolution(const Kernel& kernel);

    Mat convolveXY(const Mat& frame);
    Mat convolveT(const Mat& frame);
private:
    CircularBuffer buffer;
    const Kernel kernel;
};

#endif // CONVOLUTION_H