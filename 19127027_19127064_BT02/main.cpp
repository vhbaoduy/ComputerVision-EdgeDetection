#include "function.h"

int main(int argc, char** argv) {
	cv::CommandLineParser parser(argc, argv,
		"{input |Images\\1.png|Input image's path}"
		"{method ||Choose method [Sobel, Prewitt, Laplace, Canny, All] to detect edge of image }"
		"{direction |XY| Choose direction [X, Y, XY] of Sobel, Prewitt method}"
	);
	// Show help's commandline 
	parser.about("\n~~This program detect edge of image~~\n[Press ESC to exit program]");
	parser.printMessage();

	// Get image's path
	String imageSrc = parser.get<String>("input");
	try {
		Mat originalImage = imread(imageSrc, IMREAD_COLOR), grayscaleImage;
		if (originalImage.empty()) {
			cout << "Path doesn't exist";
			return 0;
		}
		// Resize image to 512x512
		resize(originalImage, originalImage, Size(512, 512));

		// Convert color's image to grayscale
		cvtColor(originalImage, grayscaleImage, COLOR_BGR2GRAY);

		// Get method option
		String method = parser.get<String>("method");
		String direction = parser.get<String>("direction");

		// Run all methods
		if (method == "All") {
			imshow("Orginal image", originalImage);
			sobelMethod(grayscaleImage, direction);
			prewittMethod(grayscaleImage, direction);
			laplaceMethod(grayscaleImage);
			cannyMethod(grayscaleImage);
		}

		// Sobel method
		if (method == "Sobel") {
			imshow("Orginal image", originalImage);
			sobelMethod(grayscaleImage, direction);
		}

		// Prewitt method
		if (method == "Prewitt") {
			
			imshow("Orginal image", originalImage);
			prewittMethod(grayscaleImage, direction);
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
	}
	catch (Exception& e) {
		cout << e.msg;
		return 0;
	}
	return 0;
}