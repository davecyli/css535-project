#ifndef LEAST_SQUARES_SOLVER_CUDA_H
#define LEAST_SQUARES_SOLVER_CUDA_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class LeastSquaresSolverCUDA {
public:
    LeastSquaresSolverCUDA();   // Constructor
    ~LeastSquaresSolverCUDA();  // Destructor

    void computeOpticalFlow(const cv::Mat& Ix, const cv::Mat& Iy, const cv::Mat& It, cv::Mat& flowX, cv::Mat& flowY);

private:
    float *d_Ix, *d_Iy, *d_It, *d_flowX, *d_flowY; // Device pointers
    int width, height;

    void allocateMemory(int w, int h);
    void freeMemory();
};

#endif