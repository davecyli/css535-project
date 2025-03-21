/*

Changes to make prior to GPU implementation
* Investigate transposing Y prior to convolve for cache locality
* Use cv::cuda::GpuMat instead of cv::Mat

*/
#include "profiler.h"
#include "least_squares_solver.h"
#include "spatialConvolution.h"
#include "temporalConvolution.h"
#include "collection_adapters.hpp"
#include "least_squares_solver_cuda.h"

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

#include <rerun.hpp>
#include <rerun/demo_utils.hpp>
#include <fstream>

using namespace std;
using namespace cv;


int main(int argc, char* argv[]) {
    CudaProfiler profiler = CudaProfiler();
    LeastSquaresSolver solver;
	LeastSquaresSolverCUDA solverCUDA;

    string videoFilePath = "";
    // Check input validity
    if (argc < 2) {
        std::cout << "Enter video file path (..\\OpticalFlow\\data\\Squishies.mp4 if none entered): ";
        std::getline(std::cin, videoFilePath);
        if (videoFilePath == "") {
            videoFilePath = "..\\OpticalFlow\\data\\Squishies.mp4";
        }
    } else {
        videoFilePath = argv[1];
    }

    // Create a new `RecordingStream` which sends data over TCP to the viewer process.
    const rerun::RecordingStream rec = rerun::RecordingStream("OpticalFlow");
    // Try to spawn a new viewer instance.
    rerun::Error error = rec.spawn();

    VideoCapture capture(videoFilePath);
    if (!capture.isOpened()) {
        cerr << "Error: Unable to open video file " << videoFilePath << endl;
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
    int index = 0;

    // Process frames
    SpatialConvolution cpuXY(Implementation::CPU_NAIVE);
    TemporalConvolution cpuSmoothT(Implementation::CPU_NAIVE, kernelSizeSmoothing);
    TemporalConvolution cpuDerivativeT(Implementation::CPU_NAIVE, kernelSizeDerivative);
    Mat frame, cpuSmoothedT, cpuSmoothedTX, cpuSmoothedTXY, cpuI_t, cpuI_x, cpuI_y;
    Mat derivativeDisplay, smoothedDisplay, frameRGBA;

	ofstream resultsFile("results.txt");
    while (capture.read(frame)) {

        // Display original image
        // Convert from BGR to RGBA
        cv::cvtColor(frame, frameRGBA, cv::COLOR_BGR2RGBA);
        // Log image to rerun using the tensor buffer adapter defined in `collection_adapters.hpp`.
        rec.log("0.input", rerun::Image::from_rgba32(frameRGBA, { uint32_t(frameRGBA.cols), uint32_t(frameRGBA.rows) }));

        // CPU Implementation ---------------------------------------------------------------------
        profiler.startCPUTimer();
        cpuSmoothedT = cpuSmoothT.convolve(frame, *gaussianKernel);
        if (!cpuSmoothedT.empty()) {
            rec.log("1.T.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(cpuSmoothedT, smoothedDisplay, 0, 255, cv::NORM_MINMAX);
            smoothedDisplay.convertTo(smoothedDisplay, CV_8U); // Convert to 8-bit
            rec.log("1.T", rerun::Image::from_greyscale8(smoothedDisplay, { uint32_t(smoothedDisplay.cols), uint32_t(smoothedDisplay.rows) }));
        }

        profiler.startCPUTimer();
        cpuSmoothedTX = cpuXY.convolveX(cpuSmoothedT, *gaussianKernel);
        if (!cpuSmoothedTX.empty()) {
            rec.log("2.TX.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(cpuSmoothedTX, smoothedDisplay, 0, 255, cv::NORM_MINMAX);
            smoothedDisplay.convertTo(smoothedDisplay, CV_8U); // Convert to 8-bit
            rec.log("2.TX", rerun::Image::from_greyscale8(smoothedDisplay, { uint32_t(smoothedDisplay.cols), uint32_t(smoothedDisplay.rows) }));
        }

        profiler.startCPUTimer();
        cpuSmoothedTXY = cpuXY.convolveY(cpuSmoothedTX, *gaussianKernel);
        if (!cpuSmoothedTXY.empty()) {
            rec.log("3.TXY.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(cpuSmoothedTXY, smoothedDisplay, 0, 255, cv::NORM_MINMAX);
            smoothedDisplay.convertTo(smoothedDisplay, CV_8U); // Convert to 8-bit
            rec.log("3.TXY", rerun::Image::from_greyscale8(smoothedDisplay, { uint32_t(smoothedDisplay.cols), uint32_t(smoothedDisplay.rows) }));
        }

        profiler.startCPUTimer();
        cpuI_t = cpuDerivativeT.convolve(cpuSmoothedTXY, derivativeKernel);
        if (!cpuI_t.empty()) {
            rec.log("4.I_t.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(cpuI_t, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            rec.log("4.I_t", rerun::Image::from_greyscale8(derivativeDisplay, { uint32_t(derivativeDisplay.cols), uint32_t(derivativeDisplay.rows) }));
        }

        profiler.startCPUTimer();
        cpuI_x = cpuXY.convolveX(cpuSmoothedTXY, derivativeKernel);
        if (!cpuI_x.empty()) {
            rec.log("5.I_x.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(cpuI_x, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            rec.log("5.I_x", rerun::Image::from_greyscale8(derivativeDisplay, { uint32_t(derivativeDisplay.cols), uint32_t(derivativeDisplay.rows) }));
        }

        profiler.startCPUTimer();
        cpuI_y = cpuXY.convolveY(cpuSmoothedTXY, derivativeKernel);
        if (!cpuI_y.empty()) {
            rec.log("6.I_y.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(cpuI_y, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            rec.log("6.I_y", rerun::Image::from_greyscale8(derivativeDisplay, { uint32_t(derivativeDisplay.cols), uint32_t(derivativeDisplay.rows) }));
        }

        Mat flowX = Mat::zeros(uint32_t(cpuI_x.rows), uint32_t(cpuI_x.cols), CV_32F);
        Mat flowY = Mat::zeros(uint32_t(cpuI_x.rows), uint32_t(cpuI_x.cols), CV_32F);
        Mat flowX_8u = Mat::zeros(uint32_t(cpuI_x.rows), uint32_t(cpuI_x.cols), CV_8U);
        Mat flowY_8u = Mat::zeros(uint32_t(cpuI_y.rows), uint32_t(cpuI_y.cols), CV_8U);

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;

        cv::minMaxLoc(cpuI_x, &minVal, &maxVal, &minLoc, &maxLoc);
        resultsFile << "I_x Minimum value: " << minVal << " at position " << minLoc << std::endl;
        resultsFile << "I_x Maximum value: " << maxVal << " at position " << maxLoc << std::endl;
        cv::minMaxLoc(cpuI_y, &minVal, &maxVal, &minLoc, &maxLoc);
        resultsFile << "I_y Minimum value: " << minVal << " at position " << minLoc << std::endl;
        resultsFile << "I_y Maximum value: " << maxVal << " at position " << maxLoc << std::endl;
        cv::minMaxLoc(cpuI_t, &minVal, &maxVal, &minLoc, &maxLoc);
        resultsFile << "I_t Minimum value: " << minVal << " at position " << minLoc << std::endl;
        resultsFile << "I_t Maximum value: " << maxVal << " at position " << maxLoc << std::endl;

        Mat I_x_32F;
        Mat I_y_32F;
        Mat I_t_32F;

        normalize(cpuI_x, I_x_32F, -1, 1, cv::NORM_MINMAX);
        I_x_32F.convertTo(I_x_32F, CV_32F);
        normalize(cpuI_y, I_y_32F, -1, 1, cv::NORM_MINMAX);
        I_x_32F.convertTo(I_x_32F, CV_32F);
        normalize(cpuI_t, I_t_32F, -1, 1, cv::NORM_MINMAX);
        I_x_32F.convertTo(I_x_32F, CV_32F);
        int index = 0;
        if (!cpuI_x.empty() && !cpuI_y.empty() && !cpuI_t.empty()) {
            //solver.computeOpticalFlow(I_x_32F, I_y_32F, I_t_32F, flowX, flowY);
			solverCUDA.computeOpticalFlow(cpuI_x, cpuI_y, cpuI_t, flowX, flowY);
            // Declare what you need
            imwrite("flowX_" + std::to_string(index) + ".jpg", flowX);
            imwrite("flowY_" + std::to_string(index) + ".jpg", flowY);

            cv::minMaxLoc(flowX, &minVal, &maxVal, &minLoc, &maxLoc);
            resultsFile << "flowX Minimum value: " << minVal << " at position " << minLoc << std::endl;
            resultsFile << "flowX Maximum value: " << maxVal << " at position " << maxLoc << std::endl;

            cv::minMaxLoc(flowY, &minVal, &maxVal, &minLoc, &maxLoc);
            resultsFile << "flowY Minimum value: " << minVal << " at position " << minLoc << std::endl;
            resultsFile << "flowY Maximum value: " << maxVal << " at position " << maxLoc << std::endl;

            flowX.convertTo(flowX_8u, CV_8U); // Convert to 8-bit
            flowY.convertTo(flowY_8u, CV_8U); // Convert to 8-bit

            rec.log("&.flowX", rerun::Image::from_greyscale8(flowX_8u, { uint32_t(flowX_8u.cols), uint32_t(flowX_8u.rows) }));
            rec.log("&.flowY", rerun::Image::from_greyscale8(flowY_8u, { uint32_t(flowY_8u.cols), uint32_t(flowY_8u.rows) }));
            index++;
        }
    }

    // Cleanup
    delete gaussianKernel;
    gaussianKernel = nullptr;
    capture.release();
    //destroyAllWindows();

    return 0;
}