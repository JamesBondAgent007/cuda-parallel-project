// Convolution using Shared Memory and Halo Element

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <cuda_runtime.h>

using namespace std;
using namespace cv;

#define SE_WIDTH 7
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
__global__ void sharedMemDilation(uchar* srcImg , uchar* dstImg , int srcImgRows, int srcImgCols)
{

	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ uchar srcImg_ds[w * w];

	int dst = threadIdx.y * TILE_WIDTH + threadIdx.x;
	int dstY = dst / w;
	int dstX = dst % w;
	int srcY = blockIdx.y * TILE_WIDTH + dstY - SE_RADIUS;
	int srcX = blockIdx.x * TILE_WIDTH + dstX - SE_RADIUS;
	int src = srcY * srcImgCols + srcX;

	if (srcY >= 0 && srcY < srcImgRows && srcX >= 0 && srcX < srcImgCols)
		srcImg_ds[dstY * w + dstX] = srcImg[src];
	else
		srcImg_ds[dstY * w + dstX] = 255;

	dst = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
	dstY = dst / w;
	dstX = dst % w;
	srcY = blockIdx.y * TILE_WIDTH + dstY - SE_RADIUS;
	srcX = blockIdx.x * TILE_WIDTH + dstX - SE_RADIUS;
	src = srcY * srcImgCols + srcX;
	if (dstY < w) {
		if (srcY >= 0 && srcY < srcImgRows && srcX >= 0 && srcX < srcImgCols)
			srcImg_ds[dstY * w + dstX] = srcImg[src];
		else
			srcImg_ds[dstY * w + dstX] = 255;
	}

	__syncthreads();

	if(ty >= srcImgRows || tx >= srcImgCols)return;

	uchar min = srcImg_ds[(threadIdx.y + SE_RADIUS) * w + (threadIdx.x + SE_RADIUS)]; // Selecting SE central element

	for(int i=0 ; i<SE_RADIUS ; i++)
	{
		for (int j=0 ; j<SE_RADIUS ; j++)
		{
			uchar current = srcImg_ds[(threadIdx.y+i) * w + threadIdx.x+j];
			if (current < min)
				min = current;
		}
	}

	dstImg[ty * srcImgCols + tx] = min;

	__syncthreads();

//	dstImg[ty * srcImgCols + tx] = srcImg_ds[threadIdx.y + paddingTop][threadIdx.x + paddingLeft];

};


__global__ void sharedMemErosion(uchar* srcImg , uchar* dstImg , int srcImgRows, int srcImgCols)
{

	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ uchar srcImg_ds[w * w];

	int dst = threadIdx.y * TILE_WIDTH + threadIdx.x;
	int dstY = dst / w;
	int dstX = dst % w;
	int srcY = blockIdx.y * TILE_WIDTH + dstY - SE_RADIUS;
	int srcX = blockIdx.x * TILE_WIDTH + dstX - SE_RADIUS;
	int src = srcY * srcImgCols + srcX;

	if (srcY >= 0 && srcY < srcImgRows && srcX >= 0 && srcX < srcImgCols)
		srcImg_ds[dstY * w + dstX] = srcImg[src];
	else
		srcImg_ds[dstY * w + dstX] = 0;

	dst = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
	dstY = dst / w;
	dstX = dst % w;
	srcY = blockIdx.y * TILE_WIDTH + dstY - SE_RADIUS;
	srcX = blockIdx.x * TILE_WIDTH + dstX - SE_RADIUS;
	src = srcY * srcImgCols + srcX;
	if (dstY < w) {
		if (srcY >= 0 && srcY < srcImgRows && srcX >= 0 && srcX < srcImgCols)
			srcImg_ds[dstY * w + dstX] = srcImg[src];
		else
			srcImg_ds[dstY * w + dstX] = 0;
	}

	__syncthreads();

	if(ty >= srcImgRows || tx >= srcImgCols)return;

	uchar max = srcImg_ds[(threadIdx.y + SE_RADIUS) * w + (threadIdx.x + SE_RADIUS)]; // Selecting SE central element

	for(int i=0 ; i<SE_RADIUS ; i++)
	{
		for (int j=0 ; j<SE_RADIUS ; j++)
		{
			uchar current = srcImg_ds[(threadIdx.y+i) * w + threadIdx.x+j];
			if (current > max)
				max = current;
		}
	}

	dstImg[ty * srcImgCols + tx] = max;

	__syncthreads();

};


// Wrapper function: choice = 0 -> Dilation
Mat launchKernel2(Mat& input , Mat& output , int choice)
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

		sharedMemDilation<<<gridDim , blockDim>>>(devInputPtr ,
											  	  devOutputPtr ,
											  	  input.rows ,
											  	  input.cols );
	}
	else
	{
		sharedMemErosion<<<gridDim , blockDim>>>(devInputPtr ,
												 devOutputPtr ,
												 input.rows ,
												 input.cols );
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
