//
// Created by nagash on 23/10/16.
//

#include "BenchCuda.h"

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <cuda_runtime.h>

using namespace std;
using namespace cv;

#define SE_WIDTH 7 // It's always odd
#define SE_RADIUS (SE_WIDTH - 1)/2
#define GRID_DIM 32



void BenchCuda::imshow(string message, uchar* img, int rows, int cols){
    cv::Mat cvImg(rows, cols, CV_8U, img);
    cv::imshow(message, cvImg);
}

uchar* BenchCuda::cloneImg(const uchar* img, int rows, int cols)
{
	uchar* clone = new uchar[rows * cols];
	for(int i = 0; i < rows * cols ; i++)
		clone[i] = img[i];
	return clone;
}
// Si assume un kernel con dimensioni m*n, con m,n dispari
uchar* BenchCuda::immerge(const uchar* img, int rows, int cols, int paddingTop , int paddingLeft , uchar initValue)
{
    const int immergRows = rows + 2*paddingTop;
    const int immergCols = cols + 2*paddingLeft;

    uchar* immergedImg = newImg(immergRows, immergCols, initValue);

    for(long i = 0; i < rows; i++)
    {
        for(long j = 0; j < cols; j++)
        {
            // NB: using INDEX( i+paddingTop, j+paddingLeft, immergCols ) will not works because MACRO will be translated as:
            // (col + row * cols) ----> i+paddingTop + j+paddingLeft  * immergCols --->   i + paddingTop + j + ( paddingLeft  * immergCols )  :(
            immergedImg[INDEX( i+paddingTop, j+paddingLeft, immergCols)] = img[INDEX(i, j, cols)];
        }

    }

    return immergedImg;
}



virtual  void BenchCuda::onPreRun() override {
    procImg = cloneImg(img, imgHeight, imgWidth);
    procImg = immerge(benchImage , padding , 255); // 255 for erosion, 0 for dilation



    immergedImgHeight = imgHeight + ceil(seHeight/2) * 2;
    immergedImgWidth  = imgWidth  + ceil(seWidth/2)  * 2;

    int immergedImgSize = immergedImgHeight*immergedImgWidth*sizeof(uchar);
    int imgSize = benchImage.rows*benchImage.cols*sizeof(uchar);

    // Allocating stuff on GPU
    SAFE_CALL(cudaMalloc((void**)&devProcImg , imgSize) , "CUDA Malloc Failed");

    SAFE_CALL(cudaMalloc((void**)&devImmergedImg , immergedImgSize) , "CUDA Malloc Failed");
    SAFE_CALL(cudaMemcpy(devImmergedImg , immergedImg.ptr() , immergedImgSize , cudaMemcpyHostToDevice) , "CUDA Memcpy Host To Device Failed");

    // Launching kernel(s)
    // Mysteriously Dim3 is structured like this (cols , rows , depth)
    dim3 blockDim(GRID_DIM , GRID_DIM , 1); // Using max threads number
    dim3 gridDim((benchImage.cols + blockDim.x - 1)/blockDim.x , (benchImage.rows + blockDim.y - 1)/blockDim.y , 1);


}



void  BenchCuda::run() {
    // TODO check params
    basicDilation<<<gridDim , blockDim>>>(devImmergedImg , devProcImg , immergedImgHeight , immergedImgWidth , imgWidth,  seWidth,  seHeight);
    // Checking for Kernel launch errors and wait for Device job to be done.
    SAFE_CALL(cudaDeviceSynchronize() , "Kernel Launch Failed");
}



virtual void BenchCuda::onPostRun() override {
    imgProcessed = true;


    // TODO: check if ok: put your result in benchImage!
    // Retrieving result
    SAFE_CALL(cudaMemcpy(benchImage , devProcImg , imgSize ,cudaMemcpyDeviceToHost) , "CUDA Memcpy Host To Device Failed");

    // Freeing device
    SAFE_CALL(cudaFree(devProcImg) , "CUDA Free Failed");
    SAFE_CALL(cudaFree(devImmergedImg) , "CUDA Free Failed");

}
















// srcImg is the image with padding, dstImg is without padding
__global__ void basicDilation(uchar* srcImg , uchar* dstImg , int srcImgCols , int dstImgRows , int dstImgCols, int seW, int seH) {

	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if(ty >= dstImgRows || tx >= dstImgCols)return;

	uchar min = srcImg[(ty + seH) * srcImgCols + (tx + seW)]; // Selecting SE central element

	for(int i=0 ; i<seH ; i++)
	{
		for (int j=0 ; j<seW ; j++)
		{
			uchar current = srcImg[(ty+i) * srcImgCols + (tx+j)];
			if (current < min)
				min = current;
		}
	}

	dstImg[ty * dstImgCols + tx] = min;

};


__global__ void basicErosion(uchar* srcImg , uchar* dstImg , int srcImgCols , int dstImgRows , int dstImgCols, int seW, int seH) {

	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if(ty >= dstImgRows || tx >= dstImgCols)return;

	uchar max = srcImg[(ty + seH) * srcImgCols + (tx + seW)]; // Selecting SE central element

	for(int i=0 ; i<seH ; i++)
	{
		for (int j=0 ; j<seW ; j++)
		{
			uchar current = srcImg[(ty+i) * srcImgCols + (tx+j)];
			if (current > max)
				max = current;
		}
	}

	dstImg[ty * dstImgCols + tx] = max;

};









































// TODO: this wrapper have to be splitted in: onPreRun, run, onPostRun methods. (done)
/*
// Wrapper function: choice = 0 -> Dilation
Mat BenchCuda::launchKernel(Mat& img , Mat& immergedImg , int choice) {

	// Allocating stuff on GPU
	uchar* devProcImg;
	uchar* devImmergedImg;
	int imgSize = img.rows*img.cols*sizeof(uchar);
	int immergedImgSize = immergedImg.rows*immergedImg.cols*sizeof(uchar);

	SAFE_CALL(cudaMalloc((void**)&devProcImg , imgSize) , "CUDA Malloc Failed");

	SAFE_CALL(cudaMalloc((void**)&devImmergedImg , immergedImgSize) , "CUDA Malloc Failed");
	SAFE_CALL(cudaMemcpy(devImmergedImg , immergedImg.ptr() , immergedImgSize , cudaMemcpyHostToDevice) , "CUDA Memcpy Host To Device Failed");

	// Launching kernel(s)
	// Mysteriously Dim3 is structured like this (cols , rows , depth)
	dim3 blockDim(GRID_DIM , GRID_DIM , 1); // Using max threads number
	dim3 gridDim((img.cols + blockDim.x - 1)/blockDim.x , (img.rows + blockDim.y - 1)/blockDim.y , 1);

	if(choice == 0)
	{
		// ------------------------------START TIMER HERE------------------------------------

		basicDilation<<<gridDim , blockDim>>>(devImmergedImg ,
											  devProcImg ,
											  immergedImg.cols ,
											  img.rows ,
											  img.cols,
                                                                                          seWidth,
                                                                                          seHeight);
	}
	else
	{
		basicErosion<<<gridDim , blockDim>>>(devImmergedImg ,
				  	  	  	  	  	  	  	 devProcImg ,
				  	  	  	  	  	  	  	 immergedImg.cols ,
				  	  	  	  	  	  	  	 img.rows ,
				  	  	  	  	  	  	  	 img.cols,
                                                                                         seWidth,
                                                                                         seHeight);
                                                    );
	}

	// Checking for Kernel launch errors and wait for Device job to be done.
	SAFE_CALL(cudaDeviceSynchronize() , "Kernel Launch Failed");

	// ----------------------------------END TIMER HERE--------------------------------------

	// Retrieving result
	SAFE_CALL(cudaMemcpy(img.ptr() , devProcImg , imgSize ,cudaMemcpyDeviceToHost) , "CUDA Memcpy Host To Device Failed");

	// Freeing device
	SAFE_CALL(cudaFree(devProcImg) , "CUDA Free Failed");
	SAFE_CALL(cudaFree(devImmergedImg) , "CUDA Free Failed");

	return img;
}



*/






/*
// Cuda error handler
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number) {

	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}

}
*/