#include "function.h"

void createGaussianKernel(Mat& kernel, int ksize, float sigma) {
	

	// initial kernel matrix
	Mat dest(ksize, ksize,CV_32F);
	int range = (ksize - 1) / 2;

	// initial the needed variable
	float sum = 0.0, r;
	float s = 2.0 * sigma * sigma;
	for (int x = -range; x <= range; ++x) {
		for (int y = -range; y <= range; ++y) {
			r = x * x + y * y;

			// Apply gaussian distribution
			dest.at<float>(x + range, y + range) = (exp(-(r / s))) / (M_PI * s);

			// Calculate sum to normalize
			sum += dest.at<float>(x + range, y + range);
		}
	}

	// Normalize the value of kernel
	for (int i = 0; i < ksize; ++i) {
		for (int j = 0; j < ksize; ++j) {
			dest.at<float>(i, j) /= sum;
		}
	}
	kernel = dest.clone();
}


void convolve(const Mat& src, Mat& dest, const Mat& kernel) {

	// initial destination matrix
	Mat result(src.rows, src.cols, CV_32F);

	int ksize = kernel.rows;

	// compute the center of matrix
	const int dx = ksize / 2;
	const int dy = ksize / 2;

	//loop height
	for (int i = 0; i < src.rows; ++i) {
		// loop width
		for (int j = 0; j < src.cols; ++j) {
			float temp = 0.0;
			for (int k = 0; k < ksize; ++k) {
				for (int l = 0; l < ksize; ++l) {
					int x = j - dx + l;
					int y = i - dy + k;

					// check position
					if (x >= 0 && x < src.cols && y >= 0 && y < src.rows) {
						if (kernel.type() == CV_32F && src.type() == CV_8U) {
							// reduce noise
							temp += src.at<uchar>(y, x) * kernel.at<float>(k, l);
						}
						else {
							temp += src.at<float>(y, x) * kernel.at<float>(k, l);
						}
					}
				}
			}

			//mapping to [0, 1]
			result.at<float>(i, j) = temp;
		}
	}
	//int temp = 0;
	//for (int i = 0; i < src.rows; ++i) {
	//	for (int j = 0; j < src.rows; ++j) {
	//		if (result.at<float>(i, j) < 0) temp++;
	//	}
	//}
	//cout << temp << endl;
	dest = result.clone();
}


void applyGaussianBlur(const Mat& src, Mat& dest, int ksize, float sigma) {
	Mat kernel;
	createGaussianKernel(kernel, ksize, sigma);
	convolve(src, dest, kernel);
}



void add(const Mat& mat1, const Mat& mat2, Mat& dest) {
	// check the shape of two matrices
	if (mat1.cols == mat2.cols && mat1.rows == mat2.rows) {
		// inital result matrix
		Mat result(mat1.rows, mat1.cols, CV_32F);
		for (int i = 0; i < mat1.rows; ++i) {
			for (int j = 0; j < mat1.cols; ++j) {
				result.at<float>(i, j) = mat1.at<float>(i, j) + mat2.at<float>(i, j);
			}
		}

		dest = result.clone();
	}
}


void computeHypotenuse(const Mat& mat1, const Mat& mat2, Mat& dest) {
	// check the shape of two matrices
	if (mat1.cols == mat2.cols && mat1.rows == mat2.rows) {
		Mat result(mat1.rows, mat1.cols, CV_32F);
		for (int i = 0; i < mat1.rows; ++i) {
			for (int j = 0; j < mat1.cols; ++j) {
				result.at<float>(i, j) = sqrt(mat1.at<float>(i, j) * mat1.at<float>(i, j) + mat2.at<float>(i, j) * mat2.at<float>(i, j));
			}
		}
		dest = result.clone();
	}
}

void computeTheta(const Mat& mat1, const Mat& mat2, Mat& dest) {

	Mat result(mat1.rows, mat1.cols, CV_32F);
	for (int i = 0; i < mat1.rows; ++i) {
		for (int j = 0; j < mat1.cols; ++j) {
			result.at<float>(i, j) = atan2(mat1.at<float>(i, j), mat2.at<float>(i, j));
		}
	}
	dest = result.clone();
}


void scale(Mat& mat, float value) {
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			mat.at<float>(i, j) = mat.at<float>(i, j) * value ;
		}
	}
}

void computeGradient(const Mat& image, Mat& grad, Mat& theta) {
	float xFilters[3][3] = { {-1,0, 1}, {-2,0,2},{-1,0,1} };
	float yFilters[3][3] = { {1, 2, 1} ,{0, 0, 0},{-1, -2, -1} };
	Mat Kx(3, 3, CV_32F, xFilters);
	Mat Ky(3, 3, CV_32F, yFilters);
	Mat Ix, Iy;

	convolve(image, Ix, Kx);
	convolve(image, Iy, Ky);
	computeHypotenuse(Ix, Iy, grad);
	computeTheta(Iy, Ix, theta);
}



void convertRadianToDegree(const Mat& src, Mat& dest) {
	dest = Mat(src.rows, src.cols, src.type());
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			dest.at<float>(i, j) = src.at<float>(i, j) * 180.0 / M_PI;
			if (dest.at<float>(i, j) < 0) {
				dest.at<float>(i, j) += 180.0;
			}
		}
	}
}

float findMaxPixel(const Mat& mat) {
	float max = LLONG_MIN;
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			if (mat.at<float>(i, j) > max) {
				max = mat.at<float>(i, j);
			}
		}
	}
	return max;
}


void applyNonMaxSupression(const Mat& src, Mat& dest, const Mat& degree) {
	dest = Mat(src.rows, src.cols, src.type());

	for (int i = 1; i < src.rows - 1; ++i) {
		for (int j = 1; j < src.cols - 1; ++j) {
			float value = degree.at<float>(i, j);
			float r = 255.0, q = 255.0;

			// angle 0
			if ((value >= 0 && value < 22.5) || (value >= 157.5 && value <= 180.0)) {
				q = src.at<float>(i, j - 1);
				r = src.at<float>(i, j + 1);
			}

			// angle 45
			else if (value >= 22.5 && value < 67.5) {
				q = src.at<float>(i + 1, j - 1);
				r = src.at<float>(i - 1, j + 1);
			}

			// angle 90
			else if (value <= 67.5 && value < 112.5) {
				q = src.at<float>(i + 1, j);
				r = src.at<float>(i - 1, j);
			}

			// angle 135
			else if (value >= 112.5 && value < 157.5) {
				q = src.at<float>(i - 1, j - 1);
				r = src.at<float>(i + 1, j + 1);
			}

			// check value 
			if (src.at<float>(i, j) >= q && src.at<float>(i, j) >= r) {
				dest.at<float>(i, j) = src.at<float>(i, j);
			}
			else {
				dest.at<float>(i, j) = 0.0;
			}
		}
	}
}

void applyThresholdAndHysteresis(const Mat& src, Mat& dest, float lowThreshold, float highThreshold, float strongPixel, float weakPixel) {
	dest = Mat(src.rows, src.cols, src.type());
	float maxPixel = findMaxPixel(src);
	float lowPixel = lowThreshold * maxPixel;
	float highPixel = lowPixel* highThreshold;
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {

			if (src.at<float>(i, j) < lowPixel) {
				dest.at<float>(i, j) = 0;
			}
			else if (src.at<float>(i, j) <= highPixel) {
				dest.at<float>(i, j) = weakPixel;
			}
			else {
				dest.at<float>(i, j) = strongPixel;
			}
		}
	}


	for (int i = 1; i < src.rows - 1; ++i) {
		for (int j = 1; j < src.cols - 1; ++j) {
			float pixel = dest.at<float>(i, j);
			if (pixel == weakPixel) {
				try {
					if (dest.at<float>(i - 1, j - 1) == strongPixel || dest.at<float>(i - 1, j) == strongPixel || dest.at<float>(i - 1, j + 1) == strongPixel ||
						dest.at<float>(i, j - 1) == strongPixel || dest.at<float>(i, j + 1) == strongPixel ||
						dest.at<float>(i + 1, j - 1) == strongPixel || dest.at<float>(i + 1, j) == strongPixel || dest.at<float>(i + 1, j + 1) == strongPixel) {
						dest.at<float>(i, j) = strongPixel;
					}
					else {
						dest.at<float>(i, j) = 0;
					}
				}
				catch (Exception e) {
				}
			}

		}
	}
}

int detectByCanny(const Mat& sourceImage, Mat& destinationImage, int ksize, float sigma, float lowThreshold, float highThreshold, float strongPixel, float weakPixel)
{
	try {
		Mat imageBlur, grads, theta, angle, nonMaxSuprression;

		// Step 1: reduce noise or blur image
		// create kernel to apply reduce noise
		applyGaussianBlur(sourceImage, imageBlur, ksize, sigma);
		// reduce noise
		// Step 2: compute gradient and theta
		computeGradient(imageBlur, grads, theta);
		// Step 3: apply non max supression
		//convert radion to dregree after applying
		convertRadianToDegree(theta, angle);
		applyNonMaxSupression(grads, nonMaxSuprression, angle);

		// Step 4: apply double threshold and hyteresis
		applyThresholdAndHysteresis(nonMaxSuprression, destinationImage, lowThreshold, highThreshold, strongPixel, weakPixel);
	}
	catch (Exception& e) {
		cout << e.msg << endl;
		return 0;
	}
	return 1;
}



float calculateLaplacianOfGaussian(int x, int y, float sigma) {
	float value1 = -((x * x + y * y) / (2.0 * sigma * sigma));
	float value2 = -1.0 / (M_PI * sigma * sigma * sigma * sigma);
	return value2 * (1 + value1) * exp(value1);
}

Mat createLaplacianOfGaussian(int ksize, float sigma) {
	
	if (ksize % 2 == 0) {
		ksize += 1;
	}
	Mat dest(ksize, ksize, CV_32F);
	int range = (ksize - 1) / 2;
	// initial the needed variable
	float sum = 0.0, r;
	for (int x = -range; x <= range; ++x) {
		for (int y = -range; y <= range; ++y) {

			// Apply gaussian distribution
			dest.at<float>(x + range, y + range) = calculateLaplacianOfGaussian(x,y,sigma);

			// Calculate sum to normalize
			sum += dest.at<float>(x + range, y + range);
		}
	}

	 //Normalize the value of kernel
	for (int i = 0; i < ksize; ++i) {
		for (int j = 0; j < ksize; ++j) {
			dest.at<float>(i, j) /= sum;
			//if (dest.at<float>(i, j) < 0) temp++;
		}
	}
	return dest.clone();
}


void applyZeroCrossing(const Mat& src, Mat& dest) {
	Mat result(src.rows, src.cols, CV_32F);

	for (int i = 1; i < src.rows - 1; ++i) {
		for (int j = 1; j < src.cols - 1; ++j) {
			int negCounter = 0;
			int posCounter = 0;
			for (int k = -1; k <= 1; ++k) {
				for (int l = -1; l <= 1; ++l) {
					if (k != 0 && l != 0) {
						if (src.at<float>(i + k, j + l) > 0) posCounter++;
						else if (src.at<float>(i + k, j + l) < 0) negCounter++;
					}
				}
			}
			if (negCounter > 0 && posCounter > 0) {
				result.at<float>(i, j) = 1.0;
			}
			else {
			}
		}
	}

	dest = result.clone();
}
int detectBySobel(const Mat& sourceImage, Mat& destinationImage_X, Mat& destinationImage_Y, Mat& destinationImage_XY, int ksize, float sigma)
{
	try {
		Mat imageBlur;
		float xFilters[3][3] = { {-1,0, 1}, {-2,0,2},{-1,0,1} };
		float yFilters[3][3] = { {-1, -2, -1} ,{0, 0, 0},{1, 2, 1} };
		Mat Kx(3, 3, CV_32F, xFilters);
		Mat Ky(3, 3, CV_32F, yFilters);

		applyGaussianBlur(sourceImage, imageBlur, ksize, sigma);
		convolve(imageBlur, destinationImage_X, Kx);
		scale(destinationImage_X, 1.0 / 255);
		convolve(imageBlur, destinationImage_Y, Ky);
		scale(destinationImage_Y, 1.0 / 255);
		computeHypotenuse(destinationImage_X, destinationImage_Y, destinationImage_XY);
	}
	catch (Exception& e) {
		cout << e.msg << endl;
		return 0;
	}
	return 1;
}

void detectByLaplace(const Mat& sourceImage, Mat& destinationImage) {
	float xFilters[3][3] = { {0,-1, 0}, {-1,4,-1},{0,-1,0} };
	//float yFilters[3][3] = { {-1, -1, -1} ,{-1, 8, -1},{-1, -1, -1} };
	Mat Kx(3, 3, CV_32F, xFilters);
	//Mat Ky(3, 3, CV_32F, yFilters);
	Mat Ix;
	convolve(sourceImage, destinationImage, Kx);
	//convolve(src, dest, Ky);
	//computeHypotenuse(Ix, Iy, dest);
}