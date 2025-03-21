#ifndef SPATIALCONVOLUTION_H
#define SPATIALCONVOLUTION_H

#include <opencv2/opencv.hpp>
#include "convolution.h"
#include "kernel.h"

using namespace cv;

class SpatialConvolution : public Convolution {
public:
    explicit SpatialConvolution(Implementation impl) : Convolution(impl) {}

    Mat convolveX(const Mat& frame, const Kernel& kernel);
    Mat convolveX(const Mat& frame, const Kernel& kernel, int blockSize);
    Mat convolveY(const Mat& frame, const Kernel& kernel);
    Mat convolveY(const Mat& frame, const Kernel& kernel, int blockSize);
private:
    Mat convolve(const Mat& frame, const Kernel& kernel, const bool isX, int blockSize);

    Mat cpuConvolve(const Mat& frame, const Kernel& kernel, const bool isX);
    Mat gpuConvolve(const Mat& frame, const Kernel& kernel, const bool isX, int blockSize,  
        void (*convolveKernel)(float*, float*, float*, int, int, int, bool));

    Mat launchConvolveKernel(const Mat& frame, const Kernel& kernel, bool isX, int blockSize,
        void (*convolveKernel)(float*, float*, float*, int, int, int, bool));
};

typedef void (*SpatialConvolveKernelPtr)(float*, float*, float*, int, int, int, bool);

#ifdef __CUDACC__  // Compile with nvcc
extern "C" {
    __global__ void spatialConvolveNaiveKernel(float* input, float* output, float* kernel,
        int width, int height, int kernelSize, bool isX);

    __global__ void spatialConvolveSharedMemKernel(float* input, float* output, float* kernel,
        int width, int height, int kernelSize, bool isX);
}
#endif // __CUDACC__

// Wrapper function visible to all compilers
SpatialConvolveKernelPtr getSpatialConvolveNaiveKernel();
SpatialConvolveKernelPtr getSpatialConvolveSharedMemKernel();

#endif // SPATIALCONVOLUTION_H