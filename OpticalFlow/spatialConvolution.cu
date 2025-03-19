/*

*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "spatialConvolution.h"

__global__ void convolveNaiveKernel(float* input, float* output, float* kernel, 
    int width, int kernelSize) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int halfKernel = kernelSize / 2;

    if (idx >= width) return;

    float sum = 0.0f;
    for (int k = -halfKernel; k <= halfKernel; ++k) {
        int index = idx + k;
        if (index >= 0 && index < width) {
            sum += input[index] * kernel[k + halfKernel];
        }
    }
    output[idx] = sum;
}

Mat SpatialConvolution::launchConvolveNaiveKernel(const Mat& frame, const Kernel& kernel) {
    if (frame.empty()) return frame;

    // Allocate memory on GPU for the input frame
    float* d_input;
    float* d_output;
    float* d_kernel;

    int width = frame.cols;
    int height = frame.rows;
    int kernelSize = kernel.getSize();

    size_t frameSize = width * height * sizeof(float);
    size_t kernelSizeBytes = kernelSize * sizeof(float);

    // Allocate memory on GPU
    cudaMalloc(&d_input, frameSize);
    cudaMalloc(&d_output, frameSize); // Output will have the same size
    cudaMalloc(&d_kernel, kernelSizeBytes);

    // Copy frame data and kernel to GPU
    cudaMemcpy(d_input, frame.ptr<float>(), frameSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.getRawData(), kernelSizeBytes, cudaMemcpyHostToDevice);

    // Configure and launch the kernel
    dim3 blockSize(32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x);

    convolveNaiveKernel << <gridSize, blockSize >> > (d_input, d_output, d_kernel, width, kernelSize);
    cudaDeviceSynchronize();

    // Copy the result back to host
    Mat result(height, width, frame.type());
    cudaMemcpy(result.ptr<float>(), d_output, frameSize, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return result;
}
