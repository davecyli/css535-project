#include "convolution.h"

Convolution::Convolution(const Kernel& kernel) : kernel(kernel), buffer(kernel.getSize()) {}

Mat Convolution::convolveXY(const Mat& frame) {
    if (frame.empty()) return frame;

    int rows = frame.rows;
    int cols = frame.cols;
    int halfKernel = kernel.getSize() / 2;

    Mat temp(rows, cols, frame.type());  // Temporary image for X pass
    Mat convolved(rows, cols, frame.type()); // Final result

    // First pass: Convolve X
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float sum = 0.0f;

            for (int k = -halfKernel; k <= halfKernel; ++k) {
                int x = j + k;
                if (x >= 0 && x < cols) {
                    sum += frame.at<float>(i, x) * kernel.get(k + halfKernel);
                }
            }

            temp.at<float>(i, j) = sum;  // Store first pass result
        }
    }

    // Transpose for convolve Y for cache locality
    // INVESTIGATE BEFORE USING THIS FOR GPU IMPLEMENTATION!!!
    Mat transposedTemp;
    transpose(temp, transposedTemp);

    // Second pass: Convolve Y
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            float sum = 0.0f;

            for (int k = -halfKernel; k <= halfKernel; ++k) {
                int y = j + k;
                if (y >= 0 && y < rows) {
                    sum += transposedTemp.at<float>(i, y) * kernel.get(k + halfKernel);
                }
            }

            convolved.at<float>(j, i) = sum;  // Store result transposed back
        }
    }

    return convolved;
}

Mat Convolution::convolveT(const Mat& frame) {

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