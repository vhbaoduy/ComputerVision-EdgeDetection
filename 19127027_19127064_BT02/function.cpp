#include "function.h"


//////////////////////////////////////////////////////////////////////////////////
/*
* 
* IMPLEMENT SOME SUB FUNCTION TO SUPPORT DETECTION'S FUNCTION
* 
* 
*/
//////////////////////////////////////////////////////////////////////////////////
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
	dest = result.clone();
}


void applyGaussianBlur(const Mat& src, Mat& dest, int ksize, float sigma) {
	Mat kernel;

	// create gaussian kernel
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

	if (mat1.cols == mat2.cols && mat1.rows == mat2.rows) {
		Mat result(mat1.rows, mat1.cols, CV_32F);
		for (int i = 0; i < mat1.rows; ++i) {
			for (int j = 0; j < mat1.cols; ++j) {
				result.at<float>(i, j) = atan2(mat1.at<float>(i, j), mat2.at<float>(i, j));
			}
		}
		dest = result.clone();
	}
}


void normalize(Mat& mat, float value) {
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
	float highPixel = maxPixel* highThreshold;
	float lowPixel = lowThreshold * highPixel;
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
				result.at<float>(i, j) = 255.0;
			}
			else {
			}
		}
	}

	dest = result.clone();
}


//////////////////////////////////////////////////////////////////////////////////
/*
*
* IMPLEMENT MAIN FUNCTION OF DETECTION
*
*
*/
//////////////////////////////////////////////////////////////////////////////////

int detectByCanny(const Mat& sourceImage, Mat& destinationImage, int ksize, float sigma, float lowThreshold, float highThreshold, float strongPixel, float weakPixel)
{
	try {
		Mat imageBlur, grads, theta, angle, nonMaxSuprression;

		// Step 1: reduce noise or blur image
		//// reduce noise
		applyGaussianBlur(sourceImage, imageBlur, ksize, sigma);
		
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


int detectBySobel(const Mat& sourceImage, Mat& destinationImage_X, Mat& destinationImage_Y, Mat& destinationImage_XY, int ksize, float sigma)
{
	try {
		Mat imageBlur;

		// initial sobel filter
		float xFilters[3][3] = { {-1,0, 1}, {-2,0,2},{-1,0,1} };
		float yFilters[3][3] = { {-1, -2, -1} ,{0, 0, 0},{1, 2, 1} };
		Mat Kx(3, 3, CV_32F, xFilters);
		Mat Ky(3, 3, CV_32F, yFilters);


		// Apply Gaussian kernel to reduce noise 
		applyGaussianBlur(sourceImage, imageBlur, ksize, sigma);


		convolve(imageBlur, destinationImage_X, Kx);
		// Normailize the matrix of image to show
		normalize(destinationImage_X, 1.0 / 255);

		convolve(imageBlur, destinationImage_Y, Ky);
		// Normailize the matrix of image to show
		normalize(destinationImage_Y, 1.0 / 255);


		computeHypotenuse(destinationImage_X, destinationImage_Y, destinationImage_XY);
	}
	catch (Exception& e) {
		cout << e.msg << endl;
		return 0;
	}
	return 1;
}

int detectByPrewitt(const Mat& sourceImage, Mat& destinationImage_X, Mat& destinationImage_Y, Mat& destinationImage_XY, int ksize, float sigma)
{
	try {
		Mat imageBlur;

		// initial filters
		float xFilters[3][3] = { {-1,0,1}, {-1,0,1},{-1,0,1} };
		float yFilters[3][3] = { {-1,-1,-1}, {0, 0, 0},{1, 1, 1} };
		Mat Kx(3, 3, CV_32F, xFilters);
		Mat Ky(3, 3, CV_32F, yFilters);


		// Apply Gaussian kernel to reduce noise 
		applyGaussianBlur(sourceImage, imageBlur, ksize, sigma);

		convolve(imageBlur, destinationImage_X, Kx);
		// Normalize the matrix of image to show
		normalize(destinationImage_X, 1.0 / 255);

		convolve(imageBlur, destinationImage_Y, Ky);
		// Normalize the matrix of image to show
		normalize(destinationImage_Y, 1.0 / 255);

		computeHypotenuse(destinationImage_X, destinationImage_Y, destinationImage_XY);
	}
	catch (Exception& e) {
		cout << e.msg << endl;
		return 0;
	}
	return 1;
}

int detectByLaplace(const Mat& sourceImage, Mat& destinationImage, int ksize, float sigma, bool isZeroCrossing) {


	try {
		// initial Laplace kernel
		float filters[3][3] = { {0,1, 0}, {1,-4,1},{0,1,0} };
		//float filters[3][3] = { {-1, -1, -1} ,{-1, 8, -1},{-1, -1, -1} };
		//float filters[3][3] = { {1, 1, 1} ,{1, -8, 1},{1, 1, 1} };
		Mat laplacianKernel(3, 3, CV_32F, filters);

		Mat imageBlur;
		// Apply Gaussian kernel to reduce noise 
		applyGaussianBlur(sourceImage, imageBlur, ksize, sigma);

		// Apply kernel to image
		convolve(imageBlur, destinationImage, laplacianKernel);


		// Check and apply zero crossing
		if (isZeroCrossing) {
			//normalize(destinationImage, 1 / 255.0);
			applyZeroCrossing(destinationImage, destinationImage);
		}

		return 1;
	}
	catch (Exception& e) {
		cout << e.msg << endl;
		return 0;
	}
}

void sobelMethod(const Mat& sourceImage)
{
	Mat destX, destY, destXY;
	namedWindow("Sobel", 1);
	int ksize, sigma, lowThreshold, highThreshold;

	createTrackbar("ksize", "Sobel", &ksize, 9);
	createTrackbar("sigma", "Sobel", &sigma, 100);

	setTrackbarPos("ksize", "Sobel", 5);
	setTrackbarPos("sigma", "Sobel", 10);

	while (true) {
		if (ksize % 2 != 0) {
			int check = detectBySobel(sourceImage, destX, destY, destXY, ksize, sigma);
			imshow("Sobel", destXY);

			// OpenCV
			/*Mat grad_x, grad_y;
			Sobel(originalImage, grad_x, CV_8U, 1, 0);
			Sobel(originalImage, grad_y, CV_8U, 0, 1);
			imshow("Sobel CV by X", grad_x);
			imshow("Sobel CV by Y", grad_y);*/
		}
		int iKey = waitKey(50);
		if (iKey == 27)
			break;
	}
}

void prewittMethod(const Mat& sourceImage)
{
	Mat destX, destY, destXY, grad_x, grad_y;
	namedWindow("Prewitt", 1);
	int ksize, sigma, lowThreshold, highThreshold;

	createTrackbar("ksize", "Prewitt", &ksize, 9);
	createTrackbar("sigma", "Prewitt", &sigma, 100);

	setTrackbarPos("ksize", "Prewitt", 5);
	setTrackbarPos("sigma", "Prewitt", 10);

	while (true) {
		if (ksize % 2 != 0) {
			int check = detectByPrewitt(sourceImage, destX, destY, destXY, ksize, sigma);
			imshow("Prewitt", destXY);
		}
		int iKey = waitKey(50);
		if (iKey == 27)
			break;
	}
}

void laplaceMethod(const Mat& sourceImage)
{
	Mat dest, zeroCrossing, imageLib, imageBlur;
	//namedWindow("Laplace", 1);

	while (true) {
		detectByLaplace(sourceImage, dest,5, 1.0,false);
		//normalize(dest,255.0);
		//cout << dest;
		imshow("Laplace", dest);

		//detectByLaplace(sourceImage, zeroCrossing);
		//imshow("Laplace with using ZeroCrossing", zeroCrossing);

		applyGaussianBlur(sourceImage, imageBlur,5,1.0);
		Laplacian(sourceImage, imageLib, CV_32F,5);
		normalize(imageLib, 1/255.0);
		imshow("Laplace with OpenCV", imageLib);
		int iKey = waitKey(50);
		if (iKey == 27)
			break;
	}
}

void cannyMethod(const Mat& sourceImage)
{
	Mat dest;
	namedWindow("Canny", 1);
	int ksize, sigma, lowThreshold, highThreshold;

	createTrackbar("ksize", "Canny", &ksize, 9);
	createTrackbar("sigma", "Canny", &sigma, 100);
	createTrackbar("low\nthreshold", "Canny", &lowThreshold, 100);
	createTrackbar("high\nthreshold", "Canny", &highThreshold, 100);

	setTrackbarPos("ksize", "Canny", 5);
	setTrackbarPos("sigma", "Canny", 10);
	setTrackbarPos("low\nthreshold", "Canny", 10);
	setTrackbarPos("high\nthreshold", "Canny", 30);
	
	while (true) {
		if (ksize % 2 != 0) {
			int check = detectByCanny(sourceImage, dest, ksize, sigma*1.0/100, lowThreshold*1.0/100, highThreshold*1.0/100);
			imshow("Canny", dest);
			
			// OpenCV
			Mat imageCanny;
			Canny(sourceImage, imageCanny, 25, 75,3,true);
			imshow("Canny CV", imageCanny);
		}
		int iKey = waitKey(50);
		if (iKey == 27)
			break;
	}
}
