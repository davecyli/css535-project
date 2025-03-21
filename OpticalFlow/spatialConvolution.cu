/*

*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "spatialConvolution.h"

__global__ void spatialConvolveNaiveKernel(float* input, float* output, float* kernel, 
    int width, int height, int kernelSize, bool isX) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int halfKernel = kernelSize / 2;

    if (idx >= width * height) return;

    int x, y;
    if (isX) {
        x = idx % width;  // Column index
        y = idx / width;  // Row index
    }
    else {
        y = idx % height;  // Row index
        x = idx / height;  // Column index
    }

    float sum = 0.0f;
    for (int k = -halfKernel; k <= halfKernel; ++k) {
        int index = isX ? (x + k) : (y + k);
        if (index >= 0 && (isX ? index < width : index < height) < width) {
            sum += input[(isX ? y * width + index : index * width + x)] * kernel[k + halfKernel];
        }
    }
    output[idx] = sum;
}

Mat SpatialConvolution::launchConvolveNaiveKernel(const Mat& frame, 
    const Kernel& kernel, bool isX, int blockSize) {

    if (frame.empty()) return frame;

    Mat converted = convert(frame);

    // Allocate memory on GPU for the input frame
    float* d_input;
    float* d_output;
    float* d_kernel;

    int width = converted.cols;
    int height = converted.rows;
    int frameSize = width * height;
    int kernelSize = kernel.getSize();

    size_t frameBytes = frameSize * sizeof(float);
    size_t kernelBytes = kernelSize * sizeof(float);

    // Allocate memory on GPU
    cudaMalloc(&d_input, frameBytes);
    cudaMalloc(&d_output, frameBytes); // Output will have the same size
    cudaMalloc(&d_kernel, kernelBytes);

    // Copy frame data and kernel to GPU
    cudaMemcpy(d_input, converted.ptr<float>(), frameBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.getRawData(), kernelBytes, cudaMemcpyHostToDevice);

    if (blockSize == 0) {
        blockSize = 256;
    }
    spatialConvolveNaiveKernel << <(frameSize + blockSize - 1) / blockSize, blockSize >> > (d_input, d_output, d_kernel, width, height, kernelSize, isX);

    cudaDeviceSynchronize();

    // Copy the result back to host
    Mat result(height, width, converted.type());
    cudaMemcpy(result.ptr<float>(), d_output, frameBytes, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return result;
}
