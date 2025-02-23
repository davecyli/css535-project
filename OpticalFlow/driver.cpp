/*

Changes to make prior to GPU implementation
* Investigate transposing Y prior to convolve for cache locality
* Use cv::cuda::GpuMat instead of cv::Mat

*/
#include "convolution.h"
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
    Convolution smooth(*gaussianKernel);
    Convolution derivative(derivativeKernel);
    Mat frame, smoothedT, smoothedTXY, derivativeT, derivativeTXY, derivativeDisplay;
    while (capture.read(frame)) {
        // Display original image
        imshow("Original image", frame);
        waitKey(1);

        smoothedT = smooth.convolveT(frame);
        if (!smoothedT.empty()) {
            imshow("Smoothed T", smoothedT);
            waitKey(1);
        }

        smoothedTXY = smooth.convolveXY(smoothedT);
        if (!smoothedTXY.empty()) {
            imshow("Smoothed TXY", smoothedTXY);
            waitKey(1);
        }

        derivativeT = derivative.convolveT(smoothedTXY);
        if (!derivativeT.empty()) {
            normalize(derivativeT, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("Derivative T", derivativeDisplay);
            waitKey(1);
        }

        derivativeTXY = derivative.convolveXY(derivativeT);
        if (!derivativeTXY.empty()) {
            normalize(derivativeTXY, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            imshow("Derivative TXY", derivativeDisplay);
            waitKey(1);
        }
    }

    // Cleanup
    capture.release();
    destroyAllWindows();

    return 0;
}