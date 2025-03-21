/* ------------------------------------------------------------------------------------------------
Alanna Koser, David Li, Jonah Kolar
CSS 535 A
March 23, 2025
Final Project: Optical Flow

Description:
This is the main program that demonstrates optical flow processing on video input.
It showcases three different implementations (CPU naive, GPU naive, and GPU shared memory)
of spatial and temporal convolution operations with various block sizes for performance comparison.
------------------------------------------------------------------------------------------------ */
#include "spatialConvolution.h"
#include "temporalConvolution.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/**
 * @brief Main entry point for the optical flow demonstration program
 *
 * @param argc: Number of command line arguments
 * @param argv: Array of command line argument strings, must be video file path
 * @return int: Program exit code (0 for success, non-zero for errors)
 */
int main(int argc, char* argv[]) {
    // Check input validity
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video file path>" << std::endl;
        return 1;
    }

    // Open the input video file
    string video = argv[1];
    VideoCapture capture(video);
    if (!capture.isOpened()) {
        cerr << "Error: Unable to open video file " << video << endl;
        return -1;
    }

    // Create a 1D Gaussian kernel for smoothing operations
    int kernelSizeSmoothing = 25;
    float sigma = 3.2f;
    Kernel* gaussianKernel = Kernel::generateGaussian(kernelSizeSmoothing, sigma);

    // Create a 1D Derivative kernel for computing image gradients
    int kernelSizeDerivative = 5;
    float derivativeKernelArray[] = { 1.0f / 12, -8.0f / 12, 0.0f, 8.0f / 12, -1.0f / 12 };
    Kernel derivativeKernel(derivativeKernelArray, kernelSizeDerivative);

    // Initialize variables for frame processing and visualization
    Mat frame, derivativeDisplay;

    //---------------------------------------------------------------------------------------------
    // CPU Implementation
    //---------------------------------------------------------------------------------------------

    // Initialize CPU-based convolution objects
    SpatialConvolution cpuXY(Implementation::CPU_NAIVE);
    TemporalConvolution cpuSmoothT(Implementation::CPU_NAIVE, kernelSizeSmoothing);
    TemporalConvolution cpuDerivativeT(Implementation::CPU_NAIVE, kernelSizeDerivative);

    // Variables to store intermediate and final results
    Mat cpuSmoothedT, cpuSmoothedTX, cpuSmoothedTXY, cpuI_t, cpuI_x, cpuI_y;

    // Process video frames
    while (capture.read(frame)) {
        // Display original image
        imshow("Original image", frame);
        waitKey(1);

        // Apply temporal smoothing (T dimension)
        cpuSmoothedT = cpuSmoothT.convolve(frame, *gaussianKernel);
        if (!cpuSmoothedT.empty()) {
            imshow("Smoothed T: CPU", cpuSmoothedT);
            waitKey(1);
        }

        // Apply spatial smoothing in X dimension
        cpuSmoothedTX = cpuXY.convolveX(cpuSmoothedT, *gaussianKernel);
        if (!cpuSmoothedTX.empty()) {
            imshow("Smoothed TX: CPU", cpuSmoothedTX);
            waitKey(1);
        }

        // Apply spatial smoothing in Y dimension
        cpuSmoothedTXY = cpuXY.convolveY(cpuSmoothedTX, *gaussianKernel);
        if (!cpuSmoothedTXY.empty()) {
            imshow("Smoothed TXY: CPU", cpuSmoothedTXY);
            waitKey(1);
        }

        // Calculate temporal derivative (I_t)
        cpuI_t = cpuDerivativeT.convolve(cpuSmoothedTXY, derivativeKernel);
        if (!cpuI_t.empty()) {
            // Normalize and convert for display
            normalize(cpuI_t, derivativeDisplay, 0, 255, NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("cpuI_t: CPU", derivativeDisplay);
            waitKey(1);
        }

        // Calculate spatial derivative in X dimension (I_x)
        cpuI_x = cpuXY.convolveX(cpuSmoothedTXY, derivativeKernel);
        if (!cpuI_x.empty()) {
            // Normalize and convert for display
            normalize(cpuI_x, derivativeDisplay, 0, 255, NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("cpuI_x: CPU", derivativeDisplay);
            waitKey(1);
        }

        // Calculate spatial derivative in Y dimension (I_y)
        cpuI_y = cpuXY.convolveY(cpuSmoothedTXY, derivativeKernel);
        if (!cpuI_y.empty()) {
            // Normalize and convert for display
            normalize(cpuI_y, derivativeDisplay, 0, 255, NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("cpuI_y: CPU", derivativeDisplay);
            waitKey(1);
        }

    }

    // Reset the video to begin GPU processing
    capture.release();
    capture.open(video); 

    //---------------------------------------------------------------------------------------------
    // GPU Naive Implementation with different block sizes
    //---------------------------------------------------------------------------------------------

    // Initialize GPU naive convolution objects
    SpatialConvolution gpuNaiveXY(Implementation::GPU_NAIVE);
    TemporalConvolution gpuNaiveSmoothT(Implementation::GPU_NAIVE, kernelSizeSmoothing);
    TemporalConvolution gpuNaiveDerivativeT(Implementation::GPU_NAIVE, kernelSizeDerivative);

    // Variables to store intermediate and final results
    Mat gpuNaiveSmoothedT, gpuNaiveSmoothedTX, gpuNaiveSmoothedTXY, gpuNaiveI_t, gpuNaiveI_x, gpuNaiveI_y;

    // Define block sizes to test (powers of 2 from 32 to 1024)
    int blockSizes[] = { 1 << 10, 1 << 9, 1 << 8, 1 << 7, 1 << 6, 1 << 5 };
    int numBlockSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);

    // Test each block size
    for (int i = 0; i < numBlockSizes; i++) {
        while (capture.read(frame)) {
            int blockSize = blockSizes[i];

            // Apply temporal smoothing (T dimension)
            gpuNaiveSmoothedT = gpuNaiveSmoothT.convolve(frame, *gaussianKernel, blockSize);
            if (!gpuNaiveSmoothedT.empty()) {
                string windowName = "Smoothed T, GPU Naive, Block Size: " + to_string(blockSize);
                imshow(windowName, gpuNaiveSmoothedT);
                waitKey(1);
            }

            // Apply spatial smoothing in X dimension
            gpuNaiveSmoothedTX = gpuNaiveXY.convolveX(gpuNaiveSmoothedT, *gaussianKernel, blockSize);
            if (!gpuNaiveSmoothedTX.empty()) {
                string windowName = "Smoothed TX, GPU Naive, Block Size: " + to_string(blockSize);
                imshow(windowName, gpuNaiveSmoothedTX);
                waitKey(1);
            }

            // Apply spatial smoothing in Y dimension
            gpuNaiveSmoothedTXY = gpuNaiveXY.convolveY(gpuNaiveSmoothedTX, *gaussianKernel, blockSize);
            if (!gpuNaiveSmoothedTXY.empty()) {
                string windowName = "Smoothed TXY, GPU Naive, Block Size: " + to_string(blockSize);
                imshow(windowName, gpuNaiveSmoothedTXY);
                waitKey(1);
            }

            // Calculate temporal derivative (I_t)
            gpuNaiveI_t = gpuNaiveDerivativeT.convolve(gpuNaiveSmoothedTXY, derivativeKernel, blockSize);
            if (!gpuNaiveI_t.empty()) {
                // Normalize and convert for display
                normalize(gpuNaiveI_t, derivativeDisplay, 0, 255, NORM_MINMAX);
                derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
                string windowName = "gpuNaiveI_t, GPU Naive, Block Size: " + to_string(blockSize);
                imshow(windowName, derivativeDisplay);
                waitKey(1);
            }

            // Calculate spatial derivative in X dimension (I_x)
            gpuNaiveI_x = gpuNaiveXY.convolveX(gpuNaiveSmoothedTXY, derivativeKernel, blockSize);
            if (!gpuNaiveI_x.empty()) {
                // Normalize and convert for display
                normalize(gpuNaiveI_x, derivativeDisplay, 0, 255, NORM_MINMAX);
                derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
                string windowName = "gpuNaiveI_x, GPU Naive, Block Size: " + to_string(blockSize);
                imshow(windowName, derivativeDisplay);
                waitKey(1);
            }

            // Calculate spatial derivative in Y dimension (I_y)
            gpuNaiveI_y = gpuNaiveXY.convolveY(gpuNaiveSmoothedTXY, derivativeKernel, blockSize);
            if (!gpuNaiveI_y.empty()) {
                // Normalize and convert for display
                normalize(gpuNaiveI_y, derivativeDisplay, 0, 255, NORM_MINMAX);
                derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
                string windowName = "gpuNaiveI_y, GPU Naive, Block Size: " + to_string(blockSize);
                imshow(windowName, derivativeDisplay);
                waitKey(1);
            }
        }
        // Reset the video for the next block size
        capture.release();
        capture.open(video);
    }

    //---------------------------------------------------------------------------------------------
    // GPU Shared Memory Implementation with different block sizes
    //---------------------------------------------------------------------------------------------

    // Initialize GPU shared memory convolution objects
    SpatialConvolution gpuSharedMemXY(Implementation::GPU_SHARED_MEMORY);
    TemporalConvolution gpuSharedMemSmoothT(Implementation::GPU_SHARED_MEMORY, 
        kernelSizeSmoothing);
    TemporalConvolution gpuSharedMemDerivativeT(Implementation::GPU_SHARED_MEMORY, 
        kernelSizeDerivative);

    // Variables to store intermediate and final results
    Mat gpuSharedMemSmoothedT, gpuSharedMemSmoothedTX, gpuSharedMemSmoothedTXY, 
        gpuSharedMemI_t, gpuSharedMemI_x, gpuSharedMemI_y;

    // Test each block size
    for (int i = 0; i < numBlockSizes; i++) {
        while (capture.read(frame)) {
            int blockSize = blockSizes[i];

            // Maximum block size for temporal convolution (due to shared memory limitations)
            int maxBlockSizeT = 1 << 8;

            // Apply temporal smoothing (T dimension) with appropriate block size
            if (blockSize > maxBlockSizeT) {
                gpuSharedMemSmoothedT = gpuSharedMemSmoothT.convolve(
                    frame, *gaussianKernel, maxBlockSizeT);
                if (!gpuSharedMemSmoothedT.empty()) {
                    string windowName = "Smoothed T, GPU Shared Memory, Block Size: "
                        + to_string(maxBlockSizeT);
                    imshow(windowName, gpuSharedMemSmoothedT);
                    waitKey(1);
                }
            } else {
                gpuSharedMemSmoothedT = gpuSharedMemSmoothT.convolve(
                    frame, *gaussianKernel, blockSize);
                if (!gpuSharedMemSmoothedT.empty()) {
                    string windowName = "Smoothed T, GPU Shared Memory, Block Size: "
                        + to_string(blockSize);
                    imshow(windowName, gpuSharedMemSmoothedT);
                    waitKey(1);
                }
            }

            // Apply spatial smoothing in X dimension
            gpuSharedMemSmoothedTX = gpuSharedMemXY.convolveX(
                gpuSharedMemSmoothedT, *gaussianKernel, blockSize);
            if (!gpuSharedMemSmoothedTX.empty()) {
                string windowName = "Smoothed TX, GPU Shared Memory, Block Size: "
                    + to_string(blockSize);
                imshow(windowName, gpuSharedMemSmoothedTX);
                waitKey(1);
            }
            
            // Apply spatial smoothing in Y dimension
            gpuSharedMemSmoothedTXY = gpuSharedMemXY.convolveY(
                gpuSharedMemSmoothedTX, *gaussianKernel, blockSize);
            if (!gpuSharedMemSmoothedTXY.empty()) {
                string windowName = "Smoothed TXY, GPU Shared Memory, Block Size: "
                    + to_string(blockSize);
                imshow(windowName, gpuSharedMemSmoothedTXY);
                waitKey(1);
            }
            
            // Calculate temporal derivative (I_t)
            gpuSharedMemI_t = gpuSharedMemDerivativeT.convolve(
                gpuSharedMemSmoothedTXY, derivativeKernel, blockSize);
            if (!gpuSharedMemI_t.empty()) {
                // Normalize and convert for display
                normalize(gpuSharedMemI_t, derivativeDisplay, 0, 255, NORM_MINMAX);
                derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
                string windowName = "gpuSharedMemI_t, GPU Shared Memory, Block Size: "
                    + to_string(blockSize);
                imshow(windowName, derivativeDisplay);
                waitKey(1);
            }
            
            // Calculate spatial derivative in X dimension (I_x)
            gpuSharedMemI_x = gpuSharedMemXY.convolveX(
                gpuSharedMemSmoothedTXY, derivativeKernel, blockSize);
            if (!gpuSharedMemI_x.empty()) {
                // Normalize and convert for display
                normalize(gpuSharedMemI_x, derivativeDisplay, 0, 255, NORM_MINMAX);
                derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
                string windowName = "gpuSharedMemI_x, GPU Shared Memory, Block Size: "
                    + to_string(blockSize);
                imshow(windowName, derivativeDisplay);
                waitKey(1);
            }

            // Calculate spatial derivative in Y dimension (I_y)
            gpuSharedMemI_y = gpuSharedMemXY.convolveY(
                gpuSharedMemSmoothedTXY, derivativeKernel, blockSize);
            if (!gpuSharedMemI_y.empty()) {
                // Normalize and convert for display
                normalize(gpuSharedMemI_y, derivativeDisplay, 0, 255, NORM_MINMAX);
                derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
                string windowName = "gpuSharedMemI_y, GPU Shared Memory, Block Size: "
                    + to_string(blockSize);
                imshow(windowName, derivativeDisplay);
                waitKey(1);
            }
            
        }
    // Reset the video for the next block size
    capture.release();
    capture.open(video);
    }
    // Cleanup
    delete gaussianKernel;
    gaussianKernel = nullptr;
    capture.release();
    destroyAllWindows();

    return 0;
}