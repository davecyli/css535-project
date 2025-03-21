/* ------------------------------------------------------------------------------------------------
Alanna Koser, David Li, Jonah Kolar
CSS 535 A
March 23, 2025
Final Project: Optical Flow

Description:
Implements temporal convolution operations across video frames for optical flow processing with 
support for both CPU and GPU implementations
------------------------------------------------------------------------------------------------ */
#ifndef TEMPORALCONVOLUTION_H
#define TEMPORALCONVOLUTION_H

#include <opencv2/opencv.hpp>
#include "convolution.h"
#include "kernel.h"
#include "circularBuffer.h"

using namespace cv;

class TemporalConvolution : public Convolution {
public:
    TemporalConvolution(Implementation impl, int kernelSize);
    ~TemporalConvolution();

    Mat convolve(const Mat& frame, const Kernel& kernel); // For CPU implementation
    Mat convolve(const Mat& frame, const Kernel& kernel, int blockSize); // For GPU implementation
private:
    CircularBuffer<Mat> cpuBuffer; // Stores frames for temporal convolution, CPU implementation
    CircularBuffer<float*> gpuBuffer; // Stores frames for temporal convolution, GPU implementation

    float* d_frames = nullptr; // Points to the circular buffer of frames
    float* d_convolved = nullptr; // Points to the convolution output
    float* d_kernel = nullptr; // Points to the convolution kernel

    bool isFirstPass = true;
    int frameIndex = 0; // Current position in the circular buffer

    Mat cpuConvolve(const Mat& frame, const Kernel& kernel);
    Mat gpuConvolve(const Mat& frame, const Kernel& kernel,
        int blockSize, const bool useSharedMemory,
        void (*convolveKernel)(float*, float*, float*, int, int, int));

    Mat launchConvolveKernel(const Mat& frame, const Kernel& kernel, int blockSize, 
        const bool useSharedMemory, 
        void (*convolveKernel)(float*, float*, float*, int, int, int));
};


/**
 * @brief Type definition for temporal convolution CUDA kernel function pointers
 */
typedef void (*TemporalConvolveKernelPtr)(float*, float*, float*, int, int, int);

#ifdef __CUDACC__  // Compile with nvcc
extern "C" { 
    // CUDA kernel for naive temporal convolution
    __global__ void temporalConvolveNaiveKernel(float* d_frames, float* d_output,
        float* d_kernel, int frameSize, int bufferSize, int currentIndex);

    // CUDA kernel for shared memory optimized temporal convolution
    __global__ void temporalConvolveSharedMemKernel(float* d_frames, float* d_output, 
        float* d_kernel, int frameSize, int bufferSize, int currentIndex);
}
#endif // __CUDACC__

// Wrapper functions visible to all compilers
TemporalConvolveKernelPtr getTemporalConvolveNaiveKernel();
TemporalConvolveKernelPtr getTemporalConvolveSharedMemKernel();

#endif // TEMPORALCONVOLUTION_H