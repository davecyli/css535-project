#include "convolution.h"

Mat Convolution::convolveXY(const Mat& frame, const Kernel& kernel) {
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

Mat Convolution::convolveT(const Kernel& kernel, const deque<Mat>& frames) {
    int rows = frames.front().rows;
    int cols = frames.front().cols;
    Mat convolved = Mat::zeros(rows, cols, frames.front().type());

    int index = 0;
    for (const Mat& frame : frames) {
        convolved += frame * kernel.get(index++);
    }

    return convolved;
}