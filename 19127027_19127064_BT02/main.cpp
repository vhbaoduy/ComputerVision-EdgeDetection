#include "function.h"


int main() {
	string path = "F:\\Academic 2021_2022\\Image Testing\\lena.png";
	Mat src, dest;
	src = imread(path, IMREAD_GRAYSCALE);

	//Laplacian(src, dest,CV_8U,3);
	//resize(src, src, Size(512, 512));
	////cout << src.step;
	//imshow("Source", src);
	int check = detectByCanny(src, dest, 5, 1.0, 0.1, 0.3);
	imshow("Detected", dest);
	Mat imageCanny;
	Canny(src, imageCanny, 50, 100);
	imshow("Canny", imageCanny);
	waitKey(0);
	return 0;
}