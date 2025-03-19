#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "temporalConvolution.h"

__global__ void convolveTemporalNaiveKernel(
    float* d_frames, float* d_output, float* d_kernel,
    int width, int height, int bufferSize, int currentIndex)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    float sum = 0.0f;

    for (int i = 0; i < bufferSize; i++) {
        // Compute correct index in circular buffer
        int frameIdx = (currentIndex + bufferSize - i) % bufferSize;
        sum += d_frames[frameIdx * width * height + idx] * d_kernel[i];
    }

    d_output[idx] = sum;
}

Mat TemporalConvolution::launchConvolveNaiveKernel(const Mat& frame, const Kernel& kernel) {
    if (frame.empty()) return frame;

    Mat converted = convert(frame);
    size_t frameSize = converted.rows * converted.cols;  // Number of float elements

    float* floatFrame = new float[frameSize];
    memcpy(floatFrame, converted.ptr<float>(), frameSize * sizeof(float));

    gpuBuffer.enqueue(floatFrame, frameSize);

    if (!gpuBuffer.isFull()) {
        delete[] floatFrame;  // Clean up temporary buffer
        return frame;
    }

    // Allocate GPU memory only once
    static float* d_frameData = nullptr;
    static float* d_convolved = nullptr;
    static float* d_kernel = nullptr;
    static int frameIndex = 0;  // Circular buffer index
    static bool isFirstPass = true;

    if (isFirstPass) {
        size_t totalSize = gpuBuffer.getSize() * frameSize * sizeof(float);
        cudaMalloc(&d_frameData, totalSize);
        cudaMalloc(&d_convolved, frameSize * sizeof(float));
        cudaMalloc(&d_kernel, kernel.getSize() * sizeof(float));
        isFirstPass = false;
    }

    // Copy kernel data to device
    cudaMemcpy(d_kernel, kernel.getRawData(), kernel.getSize() * sizeof(float), cudaMemcpyHostToDevice);

    // Compute the offset where the new frame should go
    int offset = frameIndex * frameSize;

    // Copy the new frame into the correct position in circular buffer
    cudaMemcpy(d_frameData + offset, floatFrame, frameSize * sizeof(float), cudaMemcpyHostToDevice);

    // Increment the frame index circularly
    frameIndex = (frameIndex + 1) % gpuBuffer.getSize();

    // Launch kernel
    int totalPixels = frame.cols * frame.rows;
    dim3 blockDim(256);
    dim3 gridDim((totalPixels + blockDim.x - 1) / blockDim.x);

    convolveTemporalNaiveKernel << <gridDim, blockDim >> > (
        d_frameData, d_convolved, d_kernel,
        frame.cols, frame.rows, gpuBuffer.getSize(), frameIndex
        );

    // Download result
    Mat result(converted.size(), CV_32F);
    cudaMemcpy(result.ptr<float>(), d_convolved, frameSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] floatFrame;

    return result;
}