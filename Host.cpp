#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <cuda_runtime.h>

#include "Device.h"

using namespace std;
using namespace cv;

Mat immerge(const Mat& img , int paddingTop , int paddingLeft , int initValue)
{

    Mat immergedImg;

    immergedImg = Mat(img.rows + 2*paddingTop , img.cols + 2*paddingLeft , CV_8UC1, initValue);

    for(int i = 0; i < img.rows; i++)
    {
        for(int j = 0; j < img.cols; j++)
        {
            immergedImg.at<uchar>(i + paddingTop , j + paddingLeft) = img.at<uchar>(i , j);
        }
    }

    return immergedImg;

};

int main(int argc , char** argv)
{

	// No need for an actual matrix SE, it's only about selecting a min or max.
	int SErows = 7;
	int SEcols = 7;

	int paddingTop = floor(SErows/2);
	int paddingLeft = floor(SEcols/2);

	Mat img = imread("/Users/Mr_Holmes/Development/NsightProjects/cuda-parallel-project/img.jpg" , CV_LOAD_IMAGE_GRAYSCALE);
	Mat immergedImg = immerge(img , paddingTop , paddingLeft , 255); // 255 cause dilation is always executed first

	int choice = 0; // 0 = Dilation, otherwise = Erosion

	launchKernel(img , immergedImg , SErows , SEcols , choice);

	return 0;

}
