#include "least_squares_solver.h"

/* -----------------------------------------------------------------------------------------------
 *	computeOpticalFlow
 *  Ix, Iy, It are matrices representing the partial derivatives on x, y, and time
 *  flowX and flowY are used to return the resulting velocity vectors.
		They must be initialized outside the function and contain all 0s.
	Ix, Iy, It, flowX, and flowY must all be 2d matrices and have the same dimensions.
 * ----------------------------------------------------------------------------------------------- */

void LeastSquaresSolver::computeOpticalFlow(const Mat& Ix, const Mat& Iy, const Mat& It, Mat& flowX, Mat& flowY) {
    // Define constants
    const float NOISE_THRESHOLD = 0.05;  // Threshold for reliable eigenvalues
    const float MIN_DETERMINANT = 1e-4;  // Threshold for matrix invertibility
    const float SIGMA1 = 0.08;  // Noise parameter 1
    const float SIGMA2 = 1.0;   // Noise parameter 2
    const float SIGMAP = 2.0;   // Noise parameter for optical flow reliability
	
	int windowSize = 5; // Grid size of pixels to consider (n x n grid to calculate the pixel at the center)
	vector<float> weights = {0.0625, 0.25, 0.375, 0.25, 0.0625}; // Pixel weights
	
	
	int halfWin = windowSize / 2;

    // Compute optical flow using weighted least squares method
    for (int y = halfWin; y < Ix.rows - halfWin; y++) {
        for (int x = halfWin; x < Ix.cols - halfWin; x++) {
            Mat A = Mat::zeros(2, 2, CV_32F);
            Mat B = Mat::zeros(2, 1, CV_32F);

            // Compute weighted A and B
            for (int i = -halfWin; i <= halfWin; i++) {
                for (int j = -halfWin; j <= halfWin; j++) {
                    float w = weights[halfWin + i] * weights[halfWin + j]; // Compute 2D weight

                    float ix = Ix.at<float>(y + i, x + j);
                    float iy = Iy.at<float>(y + i, x + j);
                    float it = It.at<float>(y + i, x + j);
                    
                    A.at<float>(0, 0) += w * (ix * ix);
                    A.at<float>(0, 1) += w * (ix * iy);
                    A.at<float>(1, 0) += w * (ix * iy);
                    A.at<float>(1, 1) += w * (iy * iy);

                    B.at<float>(0, 0) -= w * (ix * it);
                    B.at<float>(1, 0) -= w * (iy * it);
                }
            }

            // Compute eigenvalues of A
            Mat eigenvalues;
            eigen(A, eigenvalues);
            float lambda_max = eigenvalues.at<float>(0); // Largest eigenvalue

            // Discard unreliable flow
            if (lambda_max < NOISE_THRESHOLD) {
                flowX.at<float>(y, x) = 0;
                flowY.at<float>(y, x) = 0;
                continue;
            }

            // Solve for (Vx, Vy) if A is invertible
            if (determinant(A) > MIN_DETERMINANT) {
                Mat flow = A.inv() * B;
                flowX.at<float>(y, x) = flow.at<float>(0, 0);
                flowY.at<float>(y, x) = flow.at<float>(1, 0);
            }
        }
    }
}

int main(){}