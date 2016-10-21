/*
 * Device.h
 *
 *  Created on: Oct 20, 2016
 *      Author: Mr_Holmes
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#ifndef DEVICE_H_
#define DEVICE_H_

void launchKernel(cv::Mat& img , cv::Mat& immergedImg , int SErows , int SEcols , int choice);

#endif /* DEVICE_H_ */
