// Better than dequeue for locality and GPU

#include <opencv2/opencv.hpp>
#include "circularBuffer.h"

template <typename T>
CircularBuffer<T>::CircularBuffer(int capacity) : capacity(capacity), buffer(capacity) {}

template <typename T>
CircularBuffer<T>::~CircularBuffer() {}

template <>
CircularBuffer<float*>::~CircularBuffer() {
	for (int i = 0; i < capacity; ++i) {
		if (buffer[i]) {
			delete[] buffer[i];
		}
	}
}

template <typename T>
void CircularBuffer<T>::enqueue(const T& frame) {
	buffer[end] = frame;
	end = (end + 1) % capacity;
	if (size < capacity) size++;
}

template <>
void CircularBuffer<float*>::enqueue(const float* frame, size_t frameSize) {
	if (buffer[end]) {  // Free existing memory if necessary
		delete[] buffer[end];
	}

	// Allocate memory and copy the contents of frame
	buffer[end] = new float[frameSize];
	memcpy(buffer[end], frame, frameSize * sizeof(float));

	end = (end + 1) % capacity;
	if (size < capacity) size++;
}

template <typename T>
bool CircularBuffer<T>::isFull() const {
	return (size == capacity);
}

template <typename T>
int CircularBuffer<T>::getSize() const {
	return size;
}

template <typename T>
const T& CircularBuffer<T>::operator[](int index) const {
	if (index >= size) {
		throw std::out_of_range("Index out of bounds");
	}
	return buffer[(end + capacity - size + index) % capacity];
}

// Explicit instantiations for Mat and GpuMat
template class CircularBuffer<Mat>;
template class CircularBuffer<float*>;