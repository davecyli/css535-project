/*

Changes to make prior to GPU implementation
* Investigate transposing Y prior to convolve for cache locality
* Use cv::cuda::GpuMat instead of cv::Mat

*/

#include "spatialConvolution.h"
#include "temporalConvolution.h"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    // Check input validity
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video file path>" << std::endl;
        return 1;
    }

    string video = argv[1];
    VideoCapture capture(video);
    if (!capture.isOpened()) {
        cerr << "Error: Unable to open video file " << video << endl;
        return -1;
    }

    // Create a 1D Gaussian kernel
    int kernelSizeSmoothing = 25;
    float sigma = 3.2f;
    Kernel* gaussianKernel = Kernel::generateGaussian(kernelSizeSmoothing, sigma);

    // Create a 1D Derivative kernel
    int kernelSizeDerivative = 5;
    float derivativeKernelArray[] = { 1.0f / 12, -8.0f / 12, 0.0f, 8.0f / 12, -1.0f / 12 };
    Kernel derivativeKernel(derivativeKernelArray, kernelSizeDerivative);

    // Process frames
    SpatialConvolution spatial;
    TemporalConvolution temporalSmooth(kernelSizeSmoothing);
    TemporalConvolution temporalDerivative(kernelSizeDerivative);
    Mat frame, smoothedT, smoothedTX, smoothedTXY, I_t, I_x, I_y, derivativeDisplay;
    while (capture.read(frame)) {
        // Display original image
        imshow("Original image", frame);
        waitKey(1);

        smoothedT = temporalSmooth.convolve(frame, *gaussianKernel);
        if (!smoothedT.empty()) {
            imshow("Smoothed T", smoothedT);
            waitKey(1);
        }

        smoothedTX = spatial.convolveX(smoothedT, *gaussianKernel);
        if (!smoothedTX.empty()) {
            imshow("Smoothed TX", smoothedTX);
            waitKey(1);
        }

        smoothedTXY = spatial.convolveY(smoothedTX, *gaussianKernel);
        if (!smoothedTXY.empty()) {
            imshow("Smoothed TXY", smoothedTXY);
            waitKey(1);
        }
        
        I_t = temporalDerivative.convolve(smoothedTXY, derivativeKernel);
        if (!I_t.empty()) {
            normalize(I_t, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("I_t", derivativeDisplay);
            waitKey(1);
        }

        I_x = spatial.convolveX(smoothedTXY, derivativeKernel);
        if (!I_x.empty()) {
            normalize(I_x, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("I_x", derivativeDisplay);
            waitKey(1);
        }

        I_y = spatial.convolveY(smoothedTXY, derivativeKernel);
        if (!I_y.empty()) {
            normalize(I_y, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("I_y", derivativeDisplay);
            waitKey(1);
        }

    }

    // Cleanup
    delete gaussianKernel;
    gaussianKernel = nullptr;
    capture.release();
    destroyAllWindows();

    return 0;
}