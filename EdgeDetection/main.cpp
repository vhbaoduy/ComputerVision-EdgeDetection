#include "function.h"


int main() {
	string path = "F:\\Academic 2021_2022\\Image Testing\\lena.png";
	Mat src, dest;
	src = imread(path, IMREAD_GRAYSCALE);
	//cout << src.step;
	imshow("Source", src);
	int check = detectByCanny(src, dest, 5, 1.4, 20, 30);
	imshow("Detected", dest);

	Mat imageCanny;
	Canny(src, imageCanny, 50, 70);
	imshow("Canny", imageCanny);
	waitKey(0);
	return 0;
}