/* ------------------------------------------------------------------------------------------------
Alanna Koser, David Li, Jonah Kolar
CSS 535 A
March 23, 2025
Final Project: Optical Flow

Description:

Constructs a lightweight circular buffer with O(1) index-based access, specialized for maintaining
a fixed window of frames without needing to shift data in memory. It avoids frequent allocations/
deallocations that might occur with other container types.

Original intention was for better cache locality than dequeue, but the frames are large, and so
there is no locality between the pixels of interest, which are the corresponding pixels across
consecutive frames. They will be too far apart in memory.
------------------------------------------------------------------------------------------------ */

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
