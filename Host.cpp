#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <cuda_runtime.h>

#include "Device.h"
#include "Device2.h"
#include "Device3.h"

using namespace std;
using namespace cv;

#define SE_WIDTH 7 // Always odd

Mat immerge(const Mat& img , int padding , int initValue)
{

    Mat immergedImg;

    immergedImg = Mat(img.rows + 2*padding , img.cols + 2*padding , CV_8UC1, initValue);

    for(int i = 0; i < img.rows; i++)
    {
        for(int j = 0; j < img.cols; j++)
        {
            immergedImg.at<uchar>(i + padding , j + padding) = img.at<uchar>(i , j);
        }
    }

    return immergedImg;

};

int main(int argc , char** argv)
{

	int padding = floor(SE_WIDTH/2);

	Mat img = imread("/Users/Mr_Holmes/Development/NsightProjects/cuda-parallel-project/img.jpg" , CV_LOAD_IMAGE_GRAYSCALE);
	//Mat immergedImg = immerge(img , padding , 255); // 255 cause dilation is always executed first
	Mat immergedImg = immerge(img , 0 , 0);

	int choice = 0; // 0 = Dilation, otherwise = Erosion

	img = launchKernel(img , immergedImg , choice);
	//img = launchKernel2(img , immergedImg , choice);

	imshow("Processed Img" , img);
	waitKey(0);

	return 0;

}
