#include "function.h"

int main(int argc, char** argv) {
	cv::CommandLineParser parser(argc, argv,
		"{input |Images\\1.png|Input image's path}"
		"{method ||Choose method [Sobel, Prewitt, Laplace, Canny, All] to detect edge of image }"
		"{direction |XY| Direction (X, Y or XY) of Sobel, Prewitt method}"
		"{interpolation |true|Options (true or false) of Canny at Non-max Supression step}"
		"{showStep |false|Options (true or false) of Canny detection, show step by step}"
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
		String interpolation = parser.get<String>("interpolation");
		String showStep = parser.get<String>("showStep");

		// Run all methods
		if (method == "All") {
			imshow("Orginal image", originalImage);
			sobelMethod(grayscaleImage, direction);
			prewittMethod(grayscaleImage, direction);
			laplaceMethod(grayscaleImage);
			cannyMethod(grayscaleImage, interpolation,showStep);
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
			cannyMethod(grayscaleImage, interpolation, showStep);
		}
	}
	catch (Exception& e) {
		cout << e.msg;
		return 0;
	}
	return 0;
}