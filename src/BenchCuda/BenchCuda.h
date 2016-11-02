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

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

// Curiously recurring template pattern
class BenchCuda : public convolutionBench::IConvBench<BenchCuda>
{
    
private:
     
    // Cuda error handler
    //static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number);
    uchar* immerge(const uchar* img, int rows, int cols, int paddingTop , int paddingLeft , uchar initValue);
    uchar* cloneImg(const uchar* img, int rows, int cols);
    void imshow(string message, uchar* img, int rows, int cols);


    bool imgLoaded = false;
    bool imgProcessed = false;
    uint nThreads = 0;
    std::string imgPath;




protected:

    uint seWidth;
    uint seHeight;


    std::string imgPath;

    uchar* devProcImg; // on GPU
    uchar* procImg = nullptr;
    uchar* img = nullptr;
    uint imgWidth;
    uint imgHeight;


    uchar* devImmergedImg;  // on GPU
    uchar* immergedImg = nullptr;
    uint immergedImgWidth;
    uint immergedImgHeight;



    uint nThreads = 0;

public:

    BenchCuda() {}

    virtual void init(std::string imgPath, uint threads, uint se_width, uint se_height, bool useThreadsAsDivisor) override {
        seWidth = se_width;
        seHeight = se_height;

        // useThreadsAsDivisor unused: deprecated
        originalImage.setThreads(threads);
        this->nThreads = originalImage.getThreads();
        this->imgPath = imgPath;

        cv::mat imgCV = imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE);
        imgWidth = imgCV.cols;
        imgHeight = imgCV.rows;
        img = imgCV.data;
        imgLoaded = true;
    }

    virtual void showOriginalImg()  {
        if(img != nullptr)
            imshow("Original Image", img, imgHeight, imgWidth);
    };
    virtual void showProcessedImg() override {
        if(img != nullptr)
        {
            if(procImg != null)
                delete[] procImg;

            procImg = cloneImg(img, imgHeight, imgWidth);
            run();
            imshow("Processed Image", procImg, imgHeight, imgWidth);
        }
    }


    virtual const uint getSeWidth() const override {
        if(img != nullptr) return seWidth;
        else return 0;
    }
    virtual const uint getSeHeight() const override {
        if(img != nullptr) return seHeight;
        else return 0;
    }
    virtual const uint getImgWidth() const override {
        if(img != nullptr) return imgWidth;
        else return 0;
    }
    virtual const uint getImgHeight() const override {
        if(img != nullptr) return imgHeight;
        else return 0;
    }

    virtual const uint getThreads() const override {
        return nThreads;
    }
    virtual const std::string getImgPath() const override {
        return imgPath;
    }

    
    
    
    virtual  void onPreRun() override;
    void run();
	virtual void onPostRun() override;

};


#endif //STRUCTURED_BENCHOOP_H





