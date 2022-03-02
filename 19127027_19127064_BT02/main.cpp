#include "function.h"

int main(int argc, char** argv) {
	cv::CommandLineParser parser(argc, argv,
		"{help h usage ? |      | }"
		"{input |D:\\lena.png|Input image's path}"
		"{method ||Choose method to detect edge of image}");
	// Show help's commandline 
	parser.printMessage();

	// Get image's path
	String imageSrc = parser.get<String>("input");
	Mat originalImage = imread(imageSrc, IMREAD_COLOR), grayscaleImage;

	// Resize image to 512x512
	resize(originalImage, originalImage, Size(512, 512));

	// Convert color's image to grayscale
	cvtColor(originalImage, grayscaleImage, COLOR_BGR2GRAY);

	// Get method option
	String method = parser.get<String>("method");
	
	// Run all methods
	if (method == "All") {
		imshow("Orginal image", originalImage);
		sobelMethod(grayscaleImage);
		prewittMethod(grayscaleImage);
		laplaceMethod(grayscaleImage);
		cannyMethod(grayscaleImage);
	}

	// Sobel method
	if (method == "Sobel") {
		imshow("Orginal image", originalImage);
		sobelMethod(grayscaleImage);
	}

	// Prewitt method
	if (method == "Prewitt") {
		imshow("Orginal image", originalImage);
		prewittMethod(grayscaleImage);
	}

	// Laplace method
	if (method == "Laplace") {
		imshow("Orginal image", originalImage);
		laplaceMethod(grayscaleImage);
	}

	// Canny method
	if (method == "Canny") {
		imshow("Orginal image", originalImage);
		cannyMethod(grayscaleImage);
	}
	return 0;
}