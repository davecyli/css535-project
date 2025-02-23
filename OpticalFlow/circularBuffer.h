// Better than dequeue for locality and GPU

#ifndef CIRCULARBUFFER_H
#define CIRCULARBUFFER_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class CircularBuffer {
public:
    CircularBuffer(int capacity);
    // ~CircularBuffer(); will be needed for GPU version

    void enqueue(const Mat& frame);

    bool isFull() const;
    int getSize() const;

    const Mat& operator[](int index) const;

private:
    vector<Mat> buffer; // Change to Mat* buffer with cudaMalloc for GPU
    int end = 0;
    int size = 0;
    int capacity;
};
#endif // CIRCULARBUFFER_H
