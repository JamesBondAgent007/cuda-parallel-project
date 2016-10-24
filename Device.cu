#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <cuda_runtime.h>

using namespace std;
using namespace cv;

// Cuda error handler
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

// srcImg is the image with padding, dstImg is without padding
__global__ void basicDilation(uchar* srcImg , uchar* dstImg , int srcImgCols , int dstImgRows , int dstImgCols ,
							  int SErows , int SEcols)
{

	int paddingTop = (SErows-1)/2; // SErows and SEcols are assumed odd, can't call floor() from Device
	int paddingLeft = (SEcols-1)/2;

	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if(ty >= dstImgRows || tx >= dstImgCols)return;

	uchar min = srcImg[(ty + paddingTop) * srcImgCols + (tx + paddingLeft)]; // Selecting SE central element

	for(int i=0 ; i<SErows ; i++)
	{
		for (int j=0 ; j<SEcols ; j++)
		{
			uchar current = srcImg[(ty+i) * srcImgCols + (tx+j)];
			if (current < min)
				min = current;
		}
	}

	dstImg[ty * dstImgCols + tx] = min;

};

__global__ void basicErosion(uchar* srcImg , uchar* dstImg , int srcImgCols , int dstImgRows , int dstImgCols ,
		  	  	  	  	  	 int SErows , int SEcols)
{

	int paddingTop = (SErows-1)/2; // SErows and SEcols are assumed odd, can't call floor() from Device
	int paddingLeft = (SEcols-1)/2;

	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if(ty >= dstImgRows || tx >= dstImgCols)return;

	uchar max = srcImg[(ty + paddingTop) * srcImgCols + (tx + paddingLeft)]; // Selecting SE central element

	for(int i=0 ; i<SErows ; i++)
	{
		for (int j=0 ; j<SEcols ; j++)
		{
			uchar current = srcImg[(ty+i) * srcImgCols + (tx+j)];
			if (current > max)
				max = current;
		}
	}

	dstImg[ty * dstImgCols + tx] = max;

};

// Wrapper function: choice = 0 -> Dilation
void launchKernel(Mat& img , Mat& immergedImg , int SErows , int SEcols , int choice)
{

	// Allocating stuff on GPU
	uchar* devImgPtr;
	uchar* devImmergedImgPtr;
	int imgSize = img.rows*img.cols*sizeof(uchar);
	int immergedImgSize = immergedImg.rows*immergedImg.cols*sizeof(uchar);

	SAFE_CALL(cudaMalloc((void**)&devImgPtr , imgSize) , "CUDA Malloc Failed");
	SAFE_CALL(cudaMemcpy(devImgPtr , img.ptr() , imgSize , cudaMemcpyHostToDevice) , "CUDA Memcpy Host To Device Failed");

	SAFE_CALL(cudaMalloc((void**)&devImmergedImgPtr , immergedImgSize) , "CUDA Malloc Failed");
	SAFE_CALL(cudaMemcpy(devImmergedImgPtr , immergedImg.ptr() , immergedImgSize , cudaMemcpyHostToDevice) , "CUDA Memcpy Host To Device Failed");

	// Launching kernel(s)
	// Mysteriously Dim3 is structured like this (cols , rows , depth)
	dim3 blockDim(32 , 32 , 1); // Using max threads number
	dim3 gridDim((img.cols + blockDim.x - 1)/blockDim.x , (img.rows + blockDim.y - 1)/blockDim.y , 1);

	if(choice == 0)
	{
		basicDilation<<<gridDim , blockDim>>>(devImmergedImgPtr ,
											  devImgPtr ,
											  immergedImg.cols ,
											  img.rows ,
											  img.cols ,
											  SErows ,
											  SEcols);
	}
	else
	{
		basicErosion<<<gridDim , blockDim>>>(devImmergedImgPtr ,
				  	  	  	  	  	  	  	 devImgPtr ,
				  	  	  	  	  	  	  	 immergedImg.cols ,
				  	  	  	  	  	  	  	 img.rows ,
				  	  	  	  	  	  	  	 img.cols ,
				  	  	  	  	  	  	  	 SErows ,
				  	  	  	  	  	  	  	 SEcols);
	}

	// Checking for Kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize() , "Kernel Launch Failed");

	// Retrieving result
	SAFE_CALL(cudaMemcpy(img.ptr() , devImgPtr , imgSize ,cudaMemcpyDeviceToHost) , "CUDA Memcpy Host To Device Failed");

	// Freeing device
	SAFE_CALL(cudaFree(devImgPtr) , "CUDA Free Failed");
	SAFE_CALL(cudaFree(devImmergedImgPtr) , "CUDA Free Failed");

	imshow("Processed Img" , img);
	waitKey(0);

}
