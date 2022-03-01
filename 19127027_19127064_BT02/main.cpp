#include "function.h"

int main(int argc, char** argv) {
	cv::CommandLineParser parser(argc, argv,
		"{help h usage ? |      | }"
		"{input |D:\\lena.png|input image}"
		"{choice ||choose function    }");
	parser.printMessage();

	String imageSrc = parser.get<String>("input");
	Mat originalImage = imread(imageSrc, IMREAD_COLOR), grayscaleImage;
	resize(originalImage, originalImage, Size(512, 512));
	cvtColor(originalImage, grayscaleImage, COLOR_BGR2GRAY);
	String choice = parser.get<String>("choice");

	if (choice == "Sobel") {
		Mat destX, destY, destXY, grad_x, grad_y;
		namedWindow("Sobel", 1);
		int ksize, sigma, lowThreshold, highThreshold;

		createTrackbar("ksize", "Sobel", &ksize, 9);
		createTrackbar("sigma", "Sobel", &sigma, 100);
		setTrackbarPos("ksize", "Sobel", 5);
		setTrackbarPos("sigma", "Sobel", 10);

		while (true) {
			if (ksize % 2 != 0) {
				int check = detectBySobel(grayscaleImage, destX, destY, destXY, ksize, sigma);

				imshow("Orginal image", originalImage);
				imshow("Sobel", destXY);

				// OpenCV
				/*Sobel(originalImage, grad_x, CV_8U, 1, 0);
				Sobel(originalImage, grad_y, CV_8U, 0, 1);
				imshow("Sobel CV by X", grad_x);
				imshow("Sobel CV by Y", grad_y);*/
			}
			int iKey = waitKey(50);
			if (iKey == 27)
				break;
		}
	}

	if (choice=="Canny") {
		Mat dest;
		namedWindow("Canny", 1);
		int ksize, sigma, lowThreshold, highThreshold;

		createTrackbar("ksize", "Canny", &ksize, 9);
		createTrackbar("sigma", "Canny", &sigma, 100);
		createTrackbar("low\nthreshold", "Canny", &lowThreshold, 100);
		createTrackbar("high\nthreshold", "Canny", &highThreshold, 100);
		setTrackbarPos("ksize", "Canny", 5);
		setTrackbarPos("sigma", "Canny", 10);
		setTrackbarPos("low\nthreshold", "Canny", 5);
		setTrackbarPos("high\nthreshold", "Canny", 10);
		
		while (true) {
			if (ksize % 2 != 0) {
				int check = detectByCanny(grayscaleImage, dest, ksize, sigma*1.0/100, lowThreshold*1.0/100, highThreshold*1.0/100);
				imshow("Orginal", originalImage);
				imshow("Canny", dest);
				
				// OpenCV
				/*Mat imageCanny;
				Canny(originalImage, imageCanny, 50, 100);
				imshow("Canny CV", imageCanny);*/
			}
			int iKey = waitKey(50);
			if (iKey == 27)
				break;
		}
	}
	return 0;
}