#include "function.h"


int main() {


	string path = "F:\\Academic 2021_2022\\Image Testing\\cacn.jpg";
	Mat src, temp, dest;
	src = imread(path, IMREAD_GRAYSCALE);
	resize(src, src, Size(512, 512));
	//Laplacian(src, dest,CV_8U,3);
	//resize(src, src, Size(512, 512));
	////cout << src.step;
	//imshow("Source", src);
	/*int check = detectByCanny(src, dest, 5, 1.0, 0.1, 0.3);
	imshow("Detected", dest);
	Mat imageCanny;
	Canny(src, imageCanny, 50, 100);
	imshow("Canny", imageCanny);
	waitKey(0);*/
	
	Mat imgLap, imgBlur;
	GaussianBlur(src, imgBlur, Size(5,5), 1.4);
	Laplacian(imgBlur, imgLap, CV_32F);
	applyZeroCrossing(imgLap, imgLap);
	imshow("lap CV", imgLap);


	Mat lap;
	detectByLaplace(imgBlur, lap);
	//imshow("Lap", lap);
	applyZeroCrossing(lap, lap);
	imshow("lap", lap);
	
	//applyGaussianBlur(src, src, 5, 1.0);
	//Mat kernel = createLaplacianOfGaussian(7, 1.4);
	//convolve(src, temp, kernel);
	//cout << kernel;
	//scale(src, 1 / 255.0);
	//detectByLaplace(src, dest);
	//imshow("img", temp);
	//scale(temp, 1 / 255.);
	//imshow("Temp", temp);
	//applyZeroCrossing(temp, dest);
	//scale(dest, 1/255.0);
	//imshow("dest", dest);

	

	waitKey(0);
	return 0;
}