// Convolution using Shared Memory and Halo Element NO IF VER.

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <cuda_runtime.h>

using namespace std;
using namespace cv;

#define SE_WIDTH 7
#define SE_RADIUS (SE_WIDTH - 1)/2
#define TILE_WIDTH 32
#define w (TILE_WIDTH * 3)
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
__global__ void sharedBlockMemDilation(uchar* srcImg , uchar* dstImg , int srcImgCols , int dstImgCols , int dstImgRows)
{

	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	const int tx_prev = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
	const int ty_prev = (blockIdx.y - 1) * blockDim.y + threadIdx.y;

	const int tx_next = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
	const int ty_next = (blockIdx.y + 1) * blockDim.y + threadIdx.y;

	__shared__ uchar srcImg_ds[w * w];

	srcImg_ds[threadIdx.y * w + threadIdx.x] = srcImg[ty_prev * srcImgCols + tx_prev];
	srcImg_ds[(threadIdx.y + TILE_WIDTH) * w + threadIdx.x] = srcImg[ty * srcImgCols + tx_prev];
	srcImg_ds[(threadIdx.y + TILE_WIDTH * 2) * w + threadIdx.x] = srcImg[ty_next * srcImgCols + tx_prev];
	srcImg_ds[threadIdx.y * w + (threadIdx.x + TILE_WIDTH)] = srcImg[ty_prev * srcImgCols + tx];
	srcImg_ds[threadIdx.y * w + (threadIdx.x + TILE_WIDTH*2)] = srcImg[ty_prev * srcImgCols + tx_next];
	srcImg_ds[(threadIdx.y + TILE_WIDTH) * w + (threadIdx.x + TILE_WIDTH)] = srcImg[ty * srcImgCols + tx];
	srcImg_ds[(threadIdx.y + TILE_WIDTH) * w + (threadIdx.x + TILE_WIDTH*2)] = srcImg[ty * srcImgCols + tx_next];
	srcImg_ds[(threadIdx.y + TILE_WIDTH * 2) * w + (threadIdx.x + TILE_WIDTH)] = srcImg[ty_next * srcImgCols + tx];
	srcImg_ds[(threadIdx.y + TILE_WIDTH * 2) * w + (threadIdx.x + TILE_WIDTH * 2)] = srcImg[ty_next * srcImgCols + tx_next];

	__syncthreads();

	if(ty >= dstImgRows || tx >= dstImgCols)return;

	uchar min = srcImg_ds[(threadIdx.y + TILE_WIDTH + SE_RADIUS) * w + (threadIdx.x + TILE_WIDTH + SE_RADIUS)]; // Selecting SE central element

	for(int i=0 ; i<SE_RADIUS ; i++)
	{
		for (int j=0 ; j<SE_RADIUS ; j++)
		{
			uchar current = srcImg_ds[(threadIdx.y + TILE_WIDTH + i) * w + threadIdx.x + TILE_WIDTH +j];
			if (current < min)
				min = current;
		}
	}

	dstImg[ty * dstImgCols + tx] = min;

	__syncthreads();

//	dstImg[ty * srcImgCols + tx] = srcImg_ds[threadIdx.y + paddingTop][threadIdx.x + paddingLeft];

};


__global__ void sharedBlockMemErosion(uchar* srcImg , uchar* dstImg , int srcImgCols , int dstImgCols , int dstImgRows)
{



};


// Wrapper function: choice = 0 -> Dilation
Mat launchKernel3(Mat& output , Mat& input , int choice)
{

	// Allocating stuff on GPU
	uchar* devInputPtr;
	uchar* devOutputPtr;
	int imgSize = output.rows * output.cols * sizeof(uchar);
	int immergedImgSize = input.rows * input.cols * sizeof(uchar);

	SAFE_CALL(cudaMalloc((void**)&devInputPtr , immergedImgSize) , "CUDA Malloc Failed");
	SAFE_CALL(cudaMemcpy(devInputPtr , input.ptr() , immergedImgSize , cudaMemcpyHostToDevice) , "CUDA Memcpy Host To Device Failed");

	SAFE_CALL(cudaMalloc((void**)&devOutputPtr , imgSize) , "CUDA Malloc Failed");

	// Launching kernel(s)
	// Mysteriously Dim3 is structured like this (cols , rows , depth)
	dim3 blockDim(TILE_WIDTH , TILE_WIDTH , 1); // Using max threads number
	dim3 gridDim((input.cols + blockDim.x - 1)/blockDim.x , (input.rows + blockDim.y - 1)/blockDim.y , 1);

	if(choice == 0)
	{
		// ------------------------------START TIMER HERE------------------------------------

		sharedBlockMemDilation<<<gridDim , blockDim>>>(devInputPtr ,
											  	  devOutputPtr ,
											  	  input.cols ,
											  	  output.cols ,
											  	  output.rows);
	}
	else
	{
		sharedBlockMemErosion<<<gridDim , blockDim>>>(devInputPtr ,
												 devOutputPtr ,
												 input.cols ,
												 output.cols ,
												 output.rows);
	}

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
