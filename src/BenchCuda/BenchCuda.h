//
// Created by nagash on 21/10/16.
//

#ifndef STRUCTURED_BENCHOOP_H
#define STRUCTURED_BENCHOOP_H

#include "../utils/utils.h"
#include "../convolutionBench/IConvBench.h"
#include "../imProc/Image.h"
#include "../imProc/StructuringElement.h"


using namespace imProc;


// Curiously recurring template pattern
template <class T>
class BenchCUDA : public convolutionBench::IConvBench<T>
{
    
private:
     
    // Cuda error handler
    static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number);
    #define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

    __global__ void BenchCUDA::basicDilation(uchar* srcImg , uchar* dstImg , int srcImgCols , int dstImgRows , int dstImgCols, int seW, int seH);
    __global__ void BenchCUDA::basicErosion(uchar* srcImg , uchar* dstImg , int srcImgCols , int dstImgRows , int dstImgCols, int seW, int seH);
    
    bool imgLoaded = false;
    bool imgProcessed = false;
    uint nThreads = 0;
    std::string imgPath;



protected:
    StructuringElement SE = StructuringElement(1,1);
    Image originalImage = Image(1,1,0);
    Image benchImage = Image(1,1,0);
    Image immergedImg = Image(1,1,0);


public:

    BenchCUDA() {}

    virtual void init(std::string imgPath, uint threads, uint se_width, uint se_height, bool useThreadsAsDivisor) override {
        SE = StructuringElement(se_width, se_height);

        originalImage = Image(imgPath);
        if(useThreadsAsDivisor)
            originalImage.setThreads(threads / originalImage.rows());
        else originalImage.setThreads(threads);
        this->nThreads = originalImage.getThreads();
        benchImage = Image(originalImage);
        this->imgPath = imgPath;

        imgLoaded = true;
    }

    virtual void showOriginalImg()  {
        if(imgLoaded) originalImage.imshow("Original Image");
    };
    virtual void showProcessedImg() override {
        if(imgLoaded)
        {
            // enabling this check, we will show the image after ALL the processing.
            // disabling this check, we always make a new processing af a copy of the original image before showing
            // if(imgProcessed == false)
            {
                benchImage = Image(originalImage);
                run();
            }
            benchImage.imshow("Processed Image");
        }
    }

   


    virtual const uint getSeWidth() const override {
        return SE.cols();
    }
    virtual const uint getSeHeight() const override {
        return SE.rows();
    }
    virtual const uint getImgWidth() const override {
        if(imgLoaded)
            return originalImage.cols();
        else return 0;
    }
    virtual const uint getImgHeight() const override {
        if(imgLoaded)
            return originalImage.rows();
        else return 0;
    }

    virtual const uint getThreads() const override {
        return nThreads;
    }
    virtual const std::string getImgPath() const override {
        return imgPath;
    }

    
    
    
     virtual  void onPreRun() override {
        benchImage = Image(originalImage);
        // TODO: check if ok: get your data from benchImage to CUDA memory!
        
        immergedImg = immerge(benchImage , padding , 255); // 255 cause dilation is always executed first
	
        // Allocating stuff on GPU
	uchar* devImgPtr;
	uchar* devImmergedImgPtr;
	int imgSize = benchImage.rows*benchImage.cols*sizeof(uchar);
	int immergedImgSize = immergedImg.rows*immergedImg.cols*sizeof(uchar);

	SAFE_CALL(cudaMalloc((void**)&devImgPtr , imgSize) , "CUDA Malloc Failed");

	SAFE_CALL(cudaMalloc((void**)&devImmergedImgPtr , immergedImgSize) , "CUDA Malloc Failed");
	SAFE_CALL(cudaMemcpy(devImmergedImgPtr , immergedImg.ptr() , immergedImgSize , cudaMemcpyHostToDevice) , "CUDA Memcpy Host To Device Failed");

	// Launching kernel(s)
	// Mysteriously Dim3 is structured like this (cols , rows , depth)
	dim3 blockDim(GRID_DIM , GRID_DIM , 1); // Using max threads number
	dim3 gridDim((benchImage.cols + blockDim.x - 1)/blockDim.x , (benchImage.rows + blockDim.y - 1)/blockDim.y , 1);

        
    }
   
    void run() {
        // TODO check params
        basicDilation<<<gridDim , blockDim>>>(devImmergedImgPtr ,
                                                devImgPtr ,
                                                immergedImg.cols ,
                                                benchImage.rows ,
                                                benchImage.cols,
                                                seWidth,
                                                seHeight);
        // Checking for Kernel launch errors and wait for Device job to be done.
	SAFE_CALL(cudaDeviceSynchronize() , "Kernel Launch Failed");
    }
    
     virtual void onPostRun() override {
        imgProcessed = true;
        
        
        // TODO: check if ok: put your result in benchImage!
	// Retrieving result
	SAFE_CALL(cudaMemcpy(benchImage.ptr() , devImgPtr , imgSize ,cudaMemcpyDeviceToHost) , "CUDA Memcpy Host To Device Failed");

	// Freeing device
	SAFE_CALL(cudaFree(devImgPtr) , "CUDA Free Failed");
	SAFE_CALL(cudaFree(devImmergedImgPtr) , "CUDA Free Failed");

    }

};


#endif //STRUCTURED_BENCHOOP_H





