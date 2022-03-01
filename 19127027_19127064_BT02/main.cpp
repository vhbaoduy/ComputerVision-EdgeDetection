#include "function.h"

int main(int argc, char** argv) {
	cv::CommandLineParser parser(argc, argv,
		"{help h usage ? |      | }"
		"{@input |D:\\lena.png|input image}"
		"{choice ||choose function    }");
	parser.printMessage();

	String imageSrc = parser.get<String>("@input");
	Mat src = imread(imageSrc,IMREAD_GRAYSCALE);

	String choice = parser.get<String>("choice");
	
	if (choice=="Sobel"){
		Mat destX, destY, destXY, grad_x, grad_y;
		int check = detectBySobel(src, destX, destY, destXY);
		imshow("Orginal", src);
		imshow("Sobel by X", destX);
		imshow("Sobel by Y", destY);
		imshow("Sobel by XY", destXY);

		Sobel(src, grad_x, CV_8U, 1, 0);
		Sobel(src, grad_y, CV_8U, 0, 1);
		imshow("Sobel CV by X", grad_x);
		imshow("Sobel CV by Y", grad_y);
		waitKey(0);
	}
    
	if (choice=="Canny") {
		Mat dest;
		int check = detectByCanny(src, dest, 5, 1.0, 0.1, 0.3);
		imshow("Orginal", src);
		Mat imageCanny;
		Canny(src, imageCanny, 50, 100);
		imshow("Canny", dest);
		imshow("Canny CV", imageCanny);
		waitKey(0);
	}
	return 0;
}