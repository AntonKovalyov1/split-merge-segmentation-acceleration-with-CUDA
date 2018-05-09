#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <iostream>
#include <cmath>

#include "RegSeg.h"

using namespace cv;
using namespace std;

float const THRESHOLD_SPLIT = 15.0;
float const THRESHOLD_MERGE = 18.0;

int main(int argc, char** argv)
{
	Mat image;
	image = imread("image9.jpg", IMREAD_COLOR); // Read the file

	if (!image.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	Mat gray_image;
	cvtColor(image, gray_image, CV_BGR2GRAY);

	int width = gray_image.cols;
	int height = gray_image.rows;
	uchar* data = gray_image.data;
	uchar* seg_data = new uchar[width * height];
	cuda_seg s;
	s.run(data, width, height, THRESHOLD_SPLIT, THRESHOLD_MERGE, false);
	s.displaySeg(seg_data);
	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imwrite("test.jpg", Mat(height, width, CV_8UC1, seg_data));
	//imshow("Display window", Mat(height, width, CV_8UC1, data)); // Show our image inside it.
	getchar();
	return 0;
}