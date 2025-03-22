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

#include <opencv2/opencv.hpp>
#include "circularBuffer.h"

/**
* @brief Constructor
* 
* @param capacity: The maximum number of frames the buffer can hold
*/
template <typename T>
CircularBuffer<T>::CircularBuffer(int capacity) : capacity(capacity), buffer(capacity) {}

/**
* @brief Dummy destructor for buffer of Mat data type
*   Required because its declared in .h file, even though custom destructor is only necessary
*   for buffer of float* data type
*/
template <typename T>
CircularBuffer<T>::~CircularBuffer() {}

/**
* @brief Specialized destructor for buffer of float* type
*   Properly frees dynamically allocated memory for each element in the buffer
*/
template <>
CircularBuffer<float*>::~CircularBuffer() {
	// Free all dynamically allocated memory
	for (int i = 0; i < capacity; ++i) {
		if (buffer[i]) {
			delete[] buffer[i];
		}
	}
}

/**
* @brief Adds a new frame to the circular buffer
*
* @param frame: The frame to be added to the buffer
*/
template <typename T>
void CircularBuffer<T>::enqueue(const T& frame) {
	// Place the frame at the current end position
	buffer[end] = frame;

	// Update the end position, wrapping around if necessary
	end = (end + 1) % capacity;

	// Increase size if buffer is not full
	if (size < capacity) size++;
}

/**
* @brief Specialized enqueue method for float* data type
*   Handles memory allocation and copying for raw float arrays
*
* @param frame: Pointer to the float array to be added
* @param frameSize: Size of the float array in elements
*/
template <>
void CircularBuffer<float*>::enqueue(const float* frame, size_t frameSize) {
	// Free existing memory if necessary
	if (buffer[end]) {  
		delete[] buffer[end];
	}

	// Allocate new memory for the frame
	buffer[end] = new float[frameSize];

	// Copy the contents of the input frame
	memcpy(buffer[end], frame, frameSize * sizeof(float));

	// Update the end position, wrapping around if necessary
	end = (end + 1) % capacity;

	// Increase size if buffer is not full
	if (size < capacity) size++;
}

/**
* @brief Checks if the buffer has reached its maximum capacity
*
* @return true if the buffer is full, false otherwise
*/
template <typename T>
bool CircularBuffer<T>::isFull() const {
	return (size == capacity);
}

/**
* @brief Returns the current number of frames in the buffer
*
* @return The number of frames currently stored in the buffer
*/
template <typename T>
int CircularBuffer<T>::getSize() const {
	return size;
}

/**
* @brief Provides indexed access to buffer frames, with 0 being the oldest element
*
* @param index: The position of the frame to access, relative to the oldest element
* 
* @return Reference to the frame at the specified position
* 
* @throws std::out_of_range if index is out of bounds
*/
template <typename T>
const T& CircularBuffer<T>::operator[](int index) const {
	// Check if the index is valid
	if (index >= size) {
		throw std::out_of_range("Index out of bounds");
	}

	// Calculate the actual position, accounting for circular wrapping
	return buffer[(end + capacity - size + index) % capacity];
}

// Explicit instantiations for Mat and float*
template class CircularBuffer<Mat>;
template class CircularBuffer<float*>;