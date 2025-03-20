#include "convolution.h"
#include "cuda_runtime.h"

Mat Convolution::convert(const Mat& frame) const {
    if (frame.empty()) return frame;

    // Copy and convert to grayscale and float if needed
    Mat converted;

    if (frame.channels() == 3) {
        cvtColor(frame, converted, COLOR_BGR2GRAY);
        converted.convertTo(converted, CV_32F, 1.0 / 255.0);
    }
    else {
        converted = frame.clone();
    }

    return converted;
}