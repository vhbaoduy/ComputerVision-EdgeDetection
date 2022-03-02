#ifndef  _FUNCTION_H_
#define _FUNCTION_H_
#define _USE_MATH_DEFINES
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <map>


#include <assert.h>
// include library using M_PI in math
#include <cmath>

using namespace std;
using namespace cv;

/**
 * Create Gaussian Kernel
 *
 * @param kernel - The kernel matrix destination.
 * @param ksize - The size of kernel with matrix ksize x ksize
 * @param sigma - The sigma of gaussian distribution.
 */
void createGaussianKernel(Mat& kernel,int ksize, float sigma);

/**
* Conovle source matrix with kernel
* 
* @param src - The matrix of source image
* @param dest - The matrix of destination image
* @param kernel - The kernel matrix
*/
void convolve(const Mat& src, Mat& dest, const Mat& kernel);


/**
* Apply GaussianBlur kernel to image
*
* @param src - The matrix of source image (gray scale)
* @param dest - The matrix of destination image
* @param kernel - The gaussian kernel
*
*/
void applyGaussianBlur(const Mat& src, Mat& dest, int ksize, float sigma);


/**
* Add two matrices
* 
* @param mat1 - The first matrix
* @param mat2 - The second matrix
* @param dest - The result matrix
*/
void add(const Mat& mat1, const Mat& mat2, Mat& dest);

/**
* Compute the length of hypotenuse. 
* Two matrices must be the same shape
* 
* @param mat1 - The first matrix
* @param mat2 - The second matrix
* @param dest - The destination matrix having same shape with two matrix
*/
void computeHypotenuse(const Mat& mat1, const Mat& mat2, Mat& dest);

/**
* Compute the theta between x and y axis
* Two matrices must be the same shape
*
* @param mat1 - The first matrix
* @param mat2 - The second matrix
* @param dest - The destination matrix having same shape with two matrix
*/
void computeTheta(const Mat& mat1, const Mat& mat2, Mat& dest);


/**
* Scale matrix after computing gradient
* @param mat - The input matrix
* @param value - the value of scaling
*/
void scale(Mat& mat, float value);


/**
* Compute gradient and theta (angle) of image
* @param image - The input matrix of image
* @param grad - The matrix of gradient
* @param theta - The matrix of angle
*/
void computeGradient(const Mat& image, Mat& grad, Mat& theta); // function of Canny of algorithms

/**
* Convert radian matrix to degree matrix
* 
* @param src - The input matrix (radian)
* @param dest - The output matrix (degree)
*/
void convertRadianToDegree(const Mat& src, Mat& dest); // function of Canny of algorithms


/**
* Find max pixel of image
* @param mat - The matrix of image
* @return max - The max pixel in the image
*/
float findMaxPixel(const Mat& mat); // function of Canny of algorithms


/**
* Apply non max supression to the image
* 
* @param src - The matrix of image (input)
* @param dest - The destination matrix (output)
* @param degree - The matrix of degree (input)
*/
void applyNonMaxSupression(const Mat& src, Mat& dest, const Mat& degree); // function of Canny of algorithms



/*
* Apply double thresholding and hysteresis to the image
*
* @param src - The matrix of image (input)
* @param dest - The destination matrix (output)
* @param lowThreshold - The low threshold (input), default 0.05
* @param highThreshold - The high thresold (input), default 0.1
* @param strongPixel - The pixel that assigned to strong position on image, default 255
* @param weakPixel - The pixel that assigned to weak position on image, default 75
*/
void applyThresholdAndHysteresis(const Mat& src, Mat& dest, float lowThreshold, float highThreshold, float strongPixel = 255.0, float weakPixel = 75.0); // function of Canny of algorithms

/**
* Detect the egdge of image by Sobel method
* @param sourceImage - The matrix contains source image
* @param destination - The matrix contains destination image
* @param ksize - The size of kernel with matrix ksize x ksize
* @param sigma - The sigma of gaussian distribution.
* @param lowThreshold - The low threshold (input), default 0.05
* @param highThreshold - The high thresold (input), default 0.1
* @return 1 - if detecting successfully, otherwise 0.
*/
int detectBySobel(const Mat& sourceImage, Mat& destinationImage_X, Mat& destinationImage_Y, Mat& destinationImage_XY, int ksize = 5, float sigma = 1.0);


/**
* Detect the egdge of image by Prewitt method
* @param sourceImage - The matrix contains source image
* @param destination - The matrix contains destination image
* @param ksize - The size of kernel with matrix ksize x ksize
* @param sigma - The sigma of gaussian distribution.
* @param lowThreshold - The low threshold (input), default 0.05
* @param highThreshold - The high thresold (input), default 0.1
* @return 1 - if detecting successfully, otherwise 0.
*/
int detectByPrewitt(const Mat& sourceImage, Mat& destinationImage_X, Mat& destinationImage_Y, Mat& destinationImage_XY, int ksize = 5, float sigma = 1.0);

/**
* Detect the egdge of image by Canny method
* @param sourceImage - The matrix contains source image
* @param destination - The matrix contains destination image
* @param ksize - The size of kernel with matrix ksize x ksize
* @param sigma - The sigma of gaussian distribution.
* @param lowThreshold - The low threshold (input), default 0.05
* @param highThreshold - The high thresold (input), default 0.1
* @param strongPixel - The pixel that assigned to strong position on image, default 255
* @param weakPixel - The pixel that assigned to weak position on image, default 75
* @return 1 - if detecting successfully, otherwise 0.
*/
int detectByCanny(const Mat& sourceImage, Mat& destinationImage, int ksize = 5, float sigma = 1.0, float lowThreshold = 0.05, float highThreshold = 0.1, float strongPixel = 255.0, float weakPixel = 75.0);




/*
* Calculate 2D - Laplacian of Gaussian.
* 
* Reference: https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
* 
* @param x - the value at position x
* @param y - the value at position y
* @param sigma - Gaussian standard deviation
* @return The value of 2D Laplacian of Gaussian
*/
float calculateLaplacianOfGaussian(int x, int y, float sigma);


/*
* Create LoG (Laplacian of Gaussian) kernel
* @param ksize - the size of kernel
* @param sigma - Gaussian standard deviation
* @return -  The LogG kernel
*/
Mat createLaplacianOfGaussian(int ksize, float sigma);


/*
* Apply zero crossing to detect edges.
* @param src - The matrix of source image
* @param dest - The matrix of destination image
*/
void applyZeroCrossing(const Mat& src, Mat& dest);

/**
* Detect edges by Laplacian
* @param sourceImage - The matix of source image
* @param destinationImage - The maxtrix of destination image
*/
void detectByLaplace(const Mat& sourceImage, Mat& destinationImage);

/**
* Sobel method with ksize, sigma trackbar
* @param sourceImage - The matix of source image
*/
void sobelMethod(const Mat& sourceImage);

/**
* Prewitt method with ksize, sigma trackbar
* @param sourceImage - The matix of source image
*/
void prewittMethod(const Mat& sourceImage);

/**
* Laplace method
* @param sourceImage - The matix of source image
*/
void laplaceMethod(const Mat& sourceImage);

/**
* Canny method with ksize, sigma, low threshold, high threshold trackbar
* @param sourceImage - The matix of source image
*/
void cannyMethod(const Mat& sourceImage);


#endif // ! _FUNCTION_H_

