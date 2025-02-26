#ifndef LEAST_SQUARES_SOLVER_H
#define LEAST_SQUARES_SOLVER_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class LeastSquaresSolver {
public:
    void computeOpticalFlow(const Mat& Ix, const Mat& Iy, const Mat& It, Mat& flowX, Mat& flowY);
};

#endif