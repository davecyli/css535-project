/* ------------------------------------------------------------------------------------------------
Alanna Koser, David Li, Jonah Kolar
CSS 535 A
March 23, 2025
Final Project: Optical Flow

Description:
Implements 1D spatial convolution operations across video frames for optical flow processing with
support for both CPU and GPU implementations. This includes CUDA kernels for efficient parallel 
processing on GPU hardware.
------------------------------------------------------------------------------------------------ */

#ifndef SPATIALCONVOLUTION_H
#define SPATIALCONVOLUTION_H

#include <opencv2/opencv.hpp>
#include "convolution.h"
#include "kernel.h"

using namespace cv;

class SpatialConvolution : public Convolution {
public:
    explicit SpatialConvolution(Implementation impl) : Convolution(impl) {}

    Mat convolveX(const Mat& frame, const Kernel& kernel); // For CPU implementation
    Mat convolveX(const Mat& frame, const Kernel& kernel, int blockSize); // For GPU implementation
    Mat convolveY(const Mat& frame, const Kernel& kernel); // For CPU implementation
    Mat convolveY(const Mat& frame, const Kernel& kernel, int blockSize); // For GPU implementation
private:
    // Dispatcher that selects appropriate implementation
    Mat convolve(const Mat& frame, const Kernel& kernel, const bool isX, int blockSize); 

    Mat cpuConvolve(const Mat& frame, const Kernel& kernel, const bool isX);
    Mat gpuConvolve(const Mat& frame, const Kernel& kernel, const bool isX, int blockSize,  
        void (*convolveKernel)(float*, float*, float*, int, int, int, bool));

    Mat launchConvolveKernel(const Mat& frame, const Kernel& kernel, bool isX, int blockSize,
        void (*convolveKernel)(float*, float*, float*, int, int, int, bool));
};

/**
 * @brief Function pointer type for spatial convolution kernels
 */
typedef void (*SpatialConvolveKernelPtr)(float*, float*, float*, int, int, int, bool);

#ifdef __CUDACC__  // Compile with nvcc
extern "C" {
    // CUDA kernel for basic parallel convolution
    __global__ void spatialConvolveNaiveKernel(float* input, float* output, float* kernel,
        int width, int height, int kernelSize, bool isX);

    // CUDA kernel for optimized parallel convolution using shared memory
    __global__ void spatialConvolveSharedMemKernel(float* input, float* output, float* kernel,
        int width, int height, int kernelSize, bool isX);
}
#endif // __CUDACC__

// Wrapper function visible to all compilers
SpatialConvolveKernelPtr getSpatialConvolveNaiveKernel();
SpatialConvolveKernelPtr getSpatialConvolveSharedMemKernel();

#endif // SPATIALCONVOLUTION_H