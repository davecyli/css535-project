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

    Mat convolve(const Mat& frame, const Kernel& kernel);
    Mat convolve(const Mat& frame, const Kernel& kernel, int blockSize);
    Mat convolve(const Mat& frame, const Kernel& kernel, int blockSize, int tileSize);
private:
    CircularBuffer<Mat> cpuBuffer;
    CircularBuffer<float*> gpuBuffer;

    float* d_frames = nullptr;
    float* d_convolved = nullptr;
    float* d_kernel = nullptr;

    bool isFirstPass = true;
    int frameIndex = 0;

    Mat cpuConvolve(const Mat& frame, const Kernel& kernel);
    Mat gpuConvolve(const Mat& frame, const Kernel& kernel,
        int blockSize, const int tileSize, const bool useSharedMemory,
        void (*convolveKernel)(float*, float*, float*, int, int, int, int));

    Mat launchConvolveKernel(const Mat& frame, const Kernel& kernel, int blockSize, 
        const int tileSize, const bool useSharedMemory, 
        void (*convolveKernel)(float*, float*, float*, int, int, int, int));
};

typedef void (*TemporalConvolveKernelPtr)(float*, float*, float*, int, int, int, int);

#ifdef __CUDACC__  // Compile with nvcc
extern "C" { 
    __global__ void temporalConvolveNaiveKernel(float* d_frames, float* d_output,
        float* d_kernel, int frameSize, int bufferSize, int currentIndex, int tileSize);

    __global__ void temporalConvolveSharedMemKernel(float* d_frames, float* d_output, 
        float* d_kernel, int frameSize, int bufferSize, int currentIndex, int tileSize);
}
#endif // __CUDACC__

// Wrapper function visible to all compilers
TemporalConvolveKernelPtr getTemporalConvolveNaiveKernel();
TemporalConvolveKernelPtr getTemporalConvolveSharedMemKernel();

#endif // TEMPORALCONVOLUTION_H