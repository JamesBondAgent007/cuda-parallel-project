// Convolution using Shared Memory and Halo Element SIMPLIFIED VERSION

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <cuda_runtime.h>

using namespace std;
using namespace cv;

#define SE_WIDTH 7 // It's always odd
#define SE_RADIUS (SE_WIDTH - 1)/2
#define TILE_WIDTH 32
#define w (TILE_WIDTH + SE_WIDTH - 1)
#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

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


// srcImg is the image with padding, dstImg is without padding
__global__ void sharedMemDilationSimple(uchar* srcImg , uchar* dstImg , int srcImgRows, int srcImgCols)
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ uchar srcImg_ds[TILE_WIDTH * TILE_WIDTH];
	srcImg_ds[threadIdx.y * TILE_WIDTH + threadIdx.x] = srcImg[ty * srcImgCols + tx];

	__syncthreads();

	int thisTileStartPointX = blockIdx.x * blockDim.x;
	int thisTileStartPointY = blockIdx.y * blockDim.y;
	int nextTileStartPointX = (blockIdx.x + 1) * blockDim.x;
	int nextTileStartPointY = (blockIdx.y + 1) * blockDim.y;
	int srcImgStartPointX = tx - SE_RADIUS;
	int srcImgStartPointY = ty - SE_RADIUS;

	if(ty >= srcImgRows || tx >= srcImgCols)return;

	uchar min = srcImg_ds[(threadIdx.y) * TILE_WIDTH + (threadIdx.x)];

	for(int i=0 ; i<SE_RADIUS ; i++)
	{
		for(int j=0 ; j<SE_RADIUS ; j++)
		{
			int srcImgX = srcImgStartPointX + j;
			int srcImgY = srcImgStartPointY + i;
			if(srcImgX >= 0 && srcImgX < srcImgCols && srcImgY >= 0 && srcImgY < srcImgRows)
			{
				uchar current;
				if(srcImgX >= thisTileStartPointX && srcImgX < nextTileStartPointX &&
				   srcImgY >= thisTileStartPointY && srcImgY < nextTileStartPointY)
				{
					current = srcImg_ds[(threadIdx.y+i) * TILE_WIDTH + threadIdx.x+j];
				}
				else
				{
					current = srcImg[srcImgY * srcImgCols + srcImgX];
				}
				if(current < min)
					min = current;
			}
		}
	}

	dstImg[ty * srcImgCols + tx] = min;

	__syncthreads();

};


__global__ void sharedMemErosionSimple()
{



};


// Wrapper function: choice = 0 -> Dilation
Mat launchKernel3(Mat& input , Mat& output , int choice)
{

	// Allocating stuff on GPU
	uchar* devInputPtr;
	uchar* devOutputPtr;
	int imgSize = input.rows*input.cols*sizeof(uchar); // input and output size are the same

	SAFE_CALL(cudaMalloc((void**)&devInputPtr , imgSize) , "CUDA Malloc Failed");
	SAFE_CALL(cudaMemcpy(devInputPtr , input.ptr() , imgSize , cudaMemcpyHostToDevice) , "CUDA Memcpy Host To Device Failed");

	SAFE_CALL(cudaMalloc((void**)&devOutputPtr , imgSize) , "CUDA Malloc Failed");
	SAFE_CALL(cudaMemcpy(devOutputPtr , output.ptr() , imgSize , cudaMemcpyHostToDevice) , "CUDA Memcpy Host To Device Failed");

	// Launching kernel(s)
	// Mysteriously Dim3 is structured like this (cols , rows , depth)
	dim3 blockDim(TILE_WIDTH , TILE_WIDTH , 1); // Using max threads number
	dim3 gridDim((input.cols + blockDim.x - 1)/blockDim.x , (input.rows + blockDim.y - 1)/blockDim.y , 1);

	if(choice == 0)
	{
		// ------------------------------START TIMER HERE------------------------------------

		sharedMemDilationSimple<<<gridDim , blockDim>>>(devInputPtr ,
											  	  devOutputPtr ,
											  	  input.rows ,
											  	  input.cols );
	}
//	else
//	{
//		sharedMemErosionSimple<<<gridDim , blockDim>>>();
//	}

	// Checking for Kernel launch errors and wait for Device job to be done.
	SAFE_CALL(cudaDeviceSynchronize() , "Kernel Launch Failed");

	// ----------------------------------END TIMER HERE--------------------------------------

	// Retrieving result
	SAFE_CALL(cudaMemcpy(output.ptr() , devOutputPtr , imgSize ,cudaMemcpyDeviceToHost) , "CUDA Memcpy Host To Device Failed");

	// Freeing device
	SAFE_CALL(cudaFree(devOutputPtr) , "CUDA Free Failed");
	SAFE_CALL(cudaFree(devInputPtr) , "CUDA Free Failed");

	return output;

}
