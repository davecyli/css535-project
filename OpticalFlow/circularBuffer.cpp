// Better than dequeue for locality and GPU

#include "circularBuffer.h"

CircularBuffer::CircularBuffer(int capacity) : capacity(capacity), buffer(capacity) {}

void CircularBuffer::enqueue(const Mat& frame) {
	buffer[end] = frame;
	end = (end + 1) % capacity;
	if (size < capacity) size++;
}

bool CircularBuffer::isFull() const {
	return (size == capacity);
}

int CircularBuffer::getSize() const {
	return size;
}

const Mat& CircularBuffer::operator[](int index) const {
	if (index >= size) {
		throw std::out_of_range("Index out of bounds");
	}
	return buffer[(end + capacity - size + index) % capacity];
}