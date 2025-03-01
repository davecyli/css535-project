#include <least_squares_solver.h>

using namespace std;
using namespace cv;

int main() {
	// Define matrix size
    int rows = 5, cols = 5;

    // Create sample derivative matrices
    Mat Ix = (Mat_<float>(rows, cols) <<
         1, 2, 1, 0, -1,
         1, 2, 1, 0, -1,
         1, 2, 1, 0, -1,
         1, 2, 1, 0, -1,
         1, 2, 1, 0, -1);

    Mat Iy = (Mat_<float>(rows, cols) <<
         1, 1, 1, 1, 1,
         2, 2, 2, 2, 2,
         1, 1, 1, 1, 1,
         -1, -1, -1, -1, -1,
         -2, -2, -2, -2, -2);

    Mat It = (Mat_<float>(rows, cols) <<
         -1, -1, -1, -1, -1,
         -2, -2, -2, -2, -2,
         -1, -1, -1, -1, -1,
          1,  1,  1,  1,  1,
          2,  2,  2,  2,  2);

    // Initialize flow matrices to zeros
    Mat flowX = Mat::zeros(rows, cols, CV_32F);
    Mat flowY = Mat::zeros(rows, cols, CV_32F);

    // Create an instance of LeastSquaresSolver and call the function
    LeastSquaresSolver solver;
    solver.computeOpticalFlow(Ix, Iy, It, flowX, flowY);

    // Print the resulting optical flow
    cout << "FlowX:\n" << flowX << endl;
    cout << "FlowY:\n" << flowY << endl;
	
	return 0;
}