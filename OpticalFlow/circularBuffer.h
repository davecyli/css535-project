// Better than dequeue for locality and GPU

#ifndef CIRCULARBUFFER_H
#define CIRCULARBUFFER_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

template <typename T>
class CircularBuffer {
public:
    CircularBuffer(int capacity);
    ~CircularBuffer();

    void enqueue(const T& frame);
    void enqueue(const float* frame, size_t size);

    bool isFull() const;
    int getSize() const;

    const T& operator[](int index) const;

private:
    vector<T> buffer;
    int end = 0;
    int size = 0;
    int capacity;
};

// Explicit template instantiations (needed for separate compilation)
extern template class CircularBuffer<Mat>;
extern template class CircularBuffer<float*>;

#endif // CIRCULARBUFFER_H
