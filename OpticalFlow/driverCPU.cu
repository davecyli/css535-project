/* -----------------------------------------------------------------------------
Alanna Koser, David Li, Jonah Kolar
CSS 535 A
March 23, 2025

Final Project: Optical Flow
Description:
CUDA kernel implementations for CPU only
----------------------------------------------------------------------------- */
#include "collection_adapters.hpp"
#include "least_squares_solver.h"
#include "least_squares_solver_cuda.h"
#include "profiler.h"
#include "spatialConvolution.h"
#include "temporalConvolution.h"
#include "utilities.h"

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
    bool streamToRerun = true;

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

    if (argc > 2) {
        streamToRerun = false;
    }

    // Create a new `RecordingStream` which sends data over TCP to the viewer process.
    const rerun::RecordingStream rec = rerun::RecordingStream("OpticalFlow-driverCPU");
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
        rec.log("0.input", rerun::Image::from_rgba32(frameRGBA,
            { uint32_t(frameRGBA.cols), uint32_t(frameRGBA.rows) }));

        // CPU Implementation --------------------------------------------------
        profiler.startCPUTimer();
        cpuSmoothedT = cpuSmoothT.convolve(frame, *gaussianKernel);
        if (!cpuSmoothedT.empty()) {
            rec.log("1.T.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(cpuSmoothedT, smoothedDisplay, 0, 255, cv::NORM_MINMAX);
            smoothedDisplay.convertTo(smoothedDisplay, CV_8U); // Convert to 8-bit
            rec.log("1.T", rerun::Image::from_greyscale8(smoothedDisplay,
                { uint32_t(smoothedDisplay.cols), uint32_t(smoothedDisplay.rows) }));
        }

        profiler.startCPUTimer();
        cpuSmoothedTX = cpuXY.convolveX(cpuSmoothedT, *gaussianKernel);
        if (!cpuSmoothedTX.empty()) {
            rec.log("2.TX.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(cpuSmoothedTX, smoothedDisplay, 0, 255, cv::NORM_MINMAX);
            smoothedDisplay.convertTo(smoothedDisplay, CV_8U); // Convert to 8-bit
            rec.log("2.TX", rerun::Image::from_greyscale8(smoothedDisplay,
                { uint32_t(smoothedDisplay.cols), uint32_t(smoothedDisplay.rows) }));
        }

        profiler.startCPUTimer();
        cpuSmoothedTXY = cpuXY.convolveY(cpuSmoothedTX, *gaussianKernel);
        if (!cpuSmoothedTXY.empty()) {
            rec.log("3.TXY.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(cpuSmoothedTXY, smoothedDisplay, 0, 255, cv::NORM_MINMAX);
            smoothedDisplay.convertTo(smoothedDisplay, CV_8U); // Convert to 8-bit
            rec.log("3.TXY", rerun::Image::from_greyscale8(smoothedDisplay,
                { uint32_t(smoothedDisplay.cols), uint32_t(smoothedDisplay.rows) }));
        }

        profiler.startCPUTimer();
        cpuI_t = cpuDerivativeT.convolve(cpuSmoothedTXY, derivativeKernel);
        if (!cpuI_t.empty()) {
            rec.log("4.I_t.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(cpuI_t, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            rec.log("4.I_t", rerun::Image::from_greyscale8(derivativeDisplay,
                { uint32_t(derivativeDisplay.cols), uint32_t(derivativeDisplay.rows) }));
        }

        profiler.startCPUTimer();
        cpuI_x = cpuXY.convolveX(cpuSmoothedTXY, derivativeKernel);
        if (!cpuI_x.empty()) {
            rec.log("5.I_x.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(cpuI_x, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            rec.log("5.I_x", rerun::Image::from_greyscale8(derivativeDisplay,
                { uint32_t(derivativeDisplay.cols), uint32_t(derivativeDisplay.rows) }));
        }

        profiler.startCPUTimer();
        cpuI_y = cpuXY.convolveY(cpuSmoothedTXY, derivativeKernel);
        if (!cpuI_y.empty()) {
            rec.log("6.I_y.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));
            normalize(cpuI_y, derivativeDisplay, 0, 255, cv::NORM_MINMAX);
            derivativeDisplay.convertTo(derivativeDisplay, CV_8U); // Convert to 8-bit
            rec.log("6.I_y", rerun::Image::from_greyscale8(derivativeDisplay,
                { uint32_t(derivativeDisplay.cols), uint32_t(derivativeDisplay.rows) }));
        }

        Mat flowX = Mat::zeros(uint32_t(cpuI_x.rows), uint32_t(cpuI_x.cols), CV_32F);
        Mat flowY = Mat::zeros(uint32_t(cpuI_x.rows), uint32_t(cpuI_x.cols), CV_32F);
        Mat flowX_8u = Mat::zeros(uint32_t(cpuI_x.rows), uint32_t(cpuI_x.cols), CV_8U);
        Mat flowY_8u = Mat::zeros(uint32_t(cpuI_y.rows), uint32_t(cpuI_y.cols), CV_8U);

        Mat I_x_32F, I_y_32F, I_t_32F;

        int index = 0;
        if (!cpuI_x.empty() && !cpuI_y.empty() && !cpuI_t.empty()) {
            profiler.startCPUTimer();
            solver.computeOpticalFlow(cpuI_x, cpuI_y, cpuI_t, flowX, flowY);
            rec.log("6.5.LSF.CPUTime", rerun::Scalar(profiler.stopCPUTimer()));

            Mat bgr;
            flowToHSV(flowX, flowY, bgr);

            flowX.convertTo(flowX_8u, CV_8U); // Convert to 8-bit
            flowY.convertTo(flowY_8u, CV_8U); // Convert to 8-bit

            rec.log("7.flowX", rerun::Image::from_greyscale8(flowX_8u,
                { uint32_t(flowX_8u.cols), uint32_t(flowX_8u.rows) }));
            rec.log("8.flowY", rerun::Image::from_greyscale8(flowY_8u,
                { uint32_t(flowY_8u.cols), uint32_t(flowY_8u.rows) }));
            rec.log("9.OpticalFlow", rerun::Image::from_rgb24(bgr,
                { uint32_t(bgr.cols), uint32_t(bgr.rows) }));
            index++;
        }
    }

    // Cleanup
    delete gaussianKernel;
    gaussianKernel = nullptr;
    capture.release();

    return 0;
}