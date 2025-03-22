/* ------------------------------------------------------------------------------------------------
Alanna Koser, David Li, Jonah Kolar
CSS 535 A
March 23, 2025
Final Project: Optical Flow

Description:
Parent class for TemporalConvolution and SpatialConvolution. Provides constructor behavior and 
defines the implementation to be used by derived classes. Also provides convert() method for 
common preprocessing.
------------------------------------------------------------------------------------------------ */

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <opencv2/opencv.hpp>
#include "kernel.h"

using namespace cv;

/**
 * @brief Enumeration of available convolution implementation strategies
 *
 * CPU_NAIVE: Naive CPU implementation without optimization
 * GPU_NAIVE: Naive GPU implementation without optimizations
 * GPU_SHARED_MEMORY: GPU implementation using shared memory for better performance
 */
enum class Implementation {
    CPU_NAIVE,
    GPU_NAIVE,
    GPU_SHARED_MEMORY
};

class Convolution {
public:
    Convolution() : implementation(Implementation::GPU_SHARED_MEMORY) {} // default constructor
    explicit Convolution(Implementation impl) : implementation(impl) {}
protected:
    Mat convert(const Mat& frame) const; // converts to greyscale and float format
    Implementation implementation;
};

#endif // CONVOLUTION_H