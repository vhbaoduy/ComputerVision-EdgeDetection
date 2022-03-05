#ifndef  _FUNCTION_H_
#define _FUNCTION_H_
#define _USE_MATH_DEFINES
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <map>

// include library using M_PI in math
#include <cmath>

using namespace std;
using namespace cv;



//////////////////////////////////////////////////////////////////////////////////
/*
*
* DECLARE SOME SUB FUNCTION TO SUPPORT DETECTION'S FUNCTION
*
*
*/
//////////////////////////////////////////////////////////////////////////////////


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
* Convert matrix of image to new range that supports to show image with cv2::imshow()
* @param mat - The input matrix
* @param value - the value of scaling
* @return result - matrix that normalized
*/
Mat normalize(const Mat& mat, float value);


/**
* Compute gradient and theta (angle) of image
* @param image - The input matrix of image
* @param grad - The matrix of gradient
* @param gradX - The matrix of gradient in direction X
* @param gradY - The matrix of gradientin direction Y
* @param theta - The matrix of angle
*/
void computeGradient(const Mat& image, Mat& grad, Mat& gradX, Mat& gradY, Mat& theta); // function of Canny of algorithms

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
* Compute value of pixel by using interpolation. r = (b-a)*alpha + a :: http://justin-liang.com/tutorials/canny/
* @param a - coefficient of interpolation
* @param b - coefficient of interpolation
* @param c - coefficient of interpolation
* @return r - the result of interpolation
*/
float applyInterpolation(float a, float b, float alpha); // function of non-max supression

/**
* Apply non max supression to the image
* 
* @param grads - The matrix of gradient image (input)
* @param dest - The destination matrix (output)
* @param degree - The matrix of degree (input)
* @param isInterpolation - Apply non-max supression with Interpolation or not
* @param gradX - If using interpolation, it will be needed gradient X
* @param gradY - If using interpolation, it will be needed gradient Y
*/
void applyNonMaxSuppression(const Mat& grads, Mat& dest, const Mat& degree, bool isInterpolation,const Mat& gradX, const Mat& gradY); // function of Canny of algorithms



/*
* Apply double thresholding and hysteresis to the image
*
* @param src - The matrix of image (input)
* @param dest - The destination matrix (output)
* @param lowThreshold - The low threshold (input), default 0.05
* @param highThreshold - The high thresold (input), default 0.1
* @param show - Option that show image
* @param strongPixel - The pixel that assigned to strong position on image, default 255
* @param weakPixel - The pixel that assigned to weak position on image, default 75
*/
void applyThresholdAndHysteresis(const Mat& src, Mat& dest, float lowThreshold, float highThreshold,bool show, float strongPixel = 255.0, float weakPixel = 75.0); // function of Canny of algorithms



//////////////////////////////////////////////////////////////////////////////////
/*
*
* DECLARE MAIN FUNCTION OF DETECTION
*
*
*/
//////////////////////////////////////////////////////////////////////////////////


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
* @param isInterpolation - Use interpolation or not, default true
* @param lowThreshold - The low threshold (input), default 0.05
* @param highThreshold - The high thresold (input), default 0.1
* @param showStep - The option that show step by step.
* @param strongPixel - The pixel that assigned to strong position on image, default 255
* @param weakPixel - The pixel that assigned to weak position on image, default 75
* @return 1 - if detecting successfully, otherwise 0.
*/
int detectByCanny(const Mat& sourceImage, Mat& destinationImage, int ksize = 5, float sigma = 1.0,bool isInterpolation = true, float lowThreshold = 0.05, float highThreshold = 0.15,bool showStep = false, float strongPixel = 255.0, float weakPixel = 75.0);




/**
* Detect edges by Laplacian
* @param sourceImage - The matix of source image
* @param destinationImage - The maxtrix of destination image
* @param ksize - The size of kernel with matrix ksize x ksize
* @param sigma - The sigma of gaussian distribution.
* @param isZeroCrossing - The option of applying zerocrossing
*/
int detectByLaplace(const Mat& sourceImage, Mat& destinationImage, int ksize = 5, float sigma = 1.0);



//////////////////////////////////////////////////////////////////////////////////
/*
*
* FUNCTIONS THAT PROCESS COMMANDLINE
*
*
*/
//////////////////////////////////////////////////////////////////////////////////



/**
* Sobel method with ksize, sigma trackbar
* @param sourceImage - The matix of source image
*/
void sobelMethod(const Mat& sourceImage, String direction);

/**
* Prewitt method with ksize, sigma trackbar
* @param sourceImage - The matix of source image
*/
void prewittMethod(const Mat& sourceImage, String direction);

/**
* Laplace method with ksize, sigma trackbar
* @param sourceImage - The matix of source image
*/
void laplaceMethod(const Mat& sourceImage);

/**
* Canny method with ksize, sigma, low threshold, high threshold trackbar
* @param sourceImage - The matix of source image
* @param interpolation - The option at non-max suppression step
* @param showStep - The option that show step by step
*/
void cannyMethod(const Mat& sourceImage, String interpolation, String showStep);



#endif // ! _FUNCTION_H_

