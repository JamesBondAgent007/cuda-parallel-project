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
__global__ void basicDilation(int* srcImg , int* dstImg , int srcImgCols , int dstImgRows , int dstImgCols ,
							  int SErows , int SEcols)
{

	int paddingTop = (SErows-1)/2; // SErows and SEcols are assumed odd
	int paddingLeft = (SEcols-1)/2;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int min = srcImg[(y + paddingTop) * srcImgCols + (x + paddingLeft)];

	if(y < dstImgRows && x < dstImgCols) // See professor's slides to understand this check
	{
		for(int i=0 ; i<SErows ; i++)
		{
			for (int j=0 ; j<SEcols ; j++)
			{
				int current = srcImg[(y+i) * srcImgCols + (x+j)];
				if (current < min)
					min = current;
			}
		}
	}

	dstImg[y * dstImgCols + x] = min;

};

__global__ void basicErosion(int* srcImg , int* dstImg , int srcImgCols , int dstImgCols ,
							 int SErows , int SEcols)
{

	int paddingTop = (SErows-1)/2; // SErows and SEcols are assumed odd
	int paddingLeft = (SEcols-1)/2;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int max = srcImg[(y + paddingTop) * srcImgCols + (x + paddingLeft)];

	for(int i=0 ; i<SErows ; i++)
	{
		for (int j=0 ; j<SEcols ; j++)
		{
			int current = srcImg[(y+i+paddingTop) * srcImgCols + (x+j+paddingLeft)];
			if (current > max)
				max = current;
		}
	}

	dstImg[y * dstImgCols + x] = max;

};

// Wrapper function: choice = 0 -> Dilation
void launchKernel(Mat& img , Mat& immergedImg , int SErows , int SEcols , int choice)
{

	// Allocating stuff on GPU
	int* devImgPtr;
	int* devImmergedImgPtr;
	int imgSize = img.rows*img.cols*sizeof(int);
	int immergedImgSize = immergedImg.rows*immergedImg.cols*sizeof(int);

	SAFE_CALL(cudaMalloc((void**)&devImgPtr , imgSize) , "CUDA Malloc Failed");
	SAFE_CALL(cudaMemcpy(devImgPtr , img.ptr() , imgSize , cudaMemcpyHostToDevice) , "CUDA Memcpy Host To Device Failed");

	SAFE_CALL(cudaMalloc((void**)&devImmergedImgPtr , immergedImgSize) , "CUDA Malloc Failed");
	SAFE_CALL(cudaMemcpy(devImmergedImgPtr , immergedImg.ptr() , immergedImgSize , cudaMemcpyHostToDevice) , "CUDA Memcpy Host To Device Failed");

	// Launching kernel(s)
	dim3 gridDim(ceil(immergedImg.rows/32.0) , ceil(immergedImg.cols/32.0) , 1);
	dim3 blockDim(32 , 32 , 1); // Using max thread number

	if(choice == 0)
	{
		basicDilation<<<gridDim , blockDim>>>(devImmergedImgPtr ,
											  devImgPtr ,
											  immergedImg.cols ,
											  img.rows ,
											  img.cols ,
											  SErows ,
											  SEcols);

		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
	}
	else
	{
		basicErosion<<<gridDim , blockDim>>>(devImmergedImgPtr ,
				  	  	  	  	  	  	  	 devImgPtr ,
				  	  	  	  	  	  	  	 immergedImg.cols ,
				  	  	  	  	  	  	  	 img.cols ,
				  	  	  	  	  	  	  	 SErows ,
				  	  	  	  	  	  	  	 SEcols);

		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
	}

	// Use devImgPtr here to display result

	SAFE_CALL(cudaMemcpy(img.ptr() , devImgPtr , imgSize ,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

	imshow("Processed Img" , img);
	waitKey(0);

	// Freeing device
	SAFE_CALL(cudaFree(devImgPtr) , "CUDA Free Failed");
	SAFE_CALL(cudaFree(devImmergedImgPtr) , "CUDA Free Failed");

}
