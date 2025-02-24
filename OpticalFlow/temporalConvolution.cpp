#include "temporalConvolution.h"

TemporalConvolution::TemporalConvolution(int kernelSize) : buffer(kernelSize) {}

Mat TemporalConvolution::convolve(const Mat& frame, const Kernel& kernel) {
    if (frame.empty()) return frame;

    Mat converted = convert(frame);

    // Add to buffer
    buffer.enqueue(converted);

    // If there are enough frames for a convolution in t, convolve
    Mat convolved;
    if (buffer.isFull()) {
        convolved = Mat::zeros(converted.rows, converted.cols, converted.type());

        int index = 0;
        for (int i = 0; i < buffer.getSize(); i++) {
            convolved += buffer[i] * kernel.get(index++);
        }
    }
    return convolved;
}