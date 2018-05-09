#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;

static uchar** arrayTo2DArray(uchar *array, int height, int width)
{
	// initialize 2d array
	uchar **array2D = new uchar*[height];

	for (int i = 0; i < height; i++)
	{
		array2D[i] = new uchar[width];
	}

	// populate 2d array
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			array2D[i][j] = array[i * width + j];
		}
	}
	return array2D;
}

static uchar* array2DToArray(uchar** array2D, int height, int width)
{
	int size = width * height;
	uchar* array = new uchar[size];

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			array[i * width + j] = array2D[i][j];
		}
	}
	return array;
}

static Mat array2DToMat(uchar **array2D, int height, int width)
{
	return Mat(height, width, CV_8UC1, array2DToArray(array2D, height, width));
}