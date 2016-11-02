//
// Created by Nagash on 09/05/16.
//

#ifndef DISPLAYIMAGE_IMAGE_H
#define DISPLAYIMAGE_IMAGE_H


#include "../utils/utils.h"

#include "Matrix.h"
#include "Point.h"
#include "StructuringElement.h"

#include <opencv2/opencv.hpp>
#include <iostream>


#define RECTANGLE_KERNEL_OPTIMIZATION


namespace imProc
{

    template <typename T> class GenericImage
    {
    private:
        Matrix<T> _mat;
        uint rows_thread_divisor = 8;
        uint threads = 0;
    protected:

    public:

        GenericImage(int rows, int cols, const T& initVal) : _mat(rows, cols, initVal){}
        GenericImage(const Matrix<T>& matrix) : _mat(matrix) { } // deep copy of matrix
        GenericImage(Matrix<T>&& matrix) : _mat(std::move(matrix)) { } // move constructor of matrix
        GenericImage( std::string imgPath) : GenericImage(readImageCV(imgPath)) {}


        // Copy, Assignmnet, Move
        __forceinline GenericImage( const GenericImage& copy ) : _mat(copy._mat){
            rows_thread_divisor = copy.rows_thread_divisor;
            threads = copy.threads;
        }
        __forceinline GenericImage<T>& operator= (const GenericImage &im) {
            _mat = im._mat;
            rows_thread_divisor = im.rows_thread_divisor;
            threads = im.threads;
        }
        __forceinline GenericImage(GenericImage&& move ) :  _mat(std::move(move._mat)) {
            rows_thread_divisor = std::move(move.rows_thread_divisor);
            threads = std::move(move.threads);
        }
        __forceinline GenericImage<T>& operator= (GenericImage&& move) {
            _mat = std::move(move._mat);
            rows_thread_divisor = std::move(move.rows_thread_divisor);
            threads = std::move(move.threads);
        }


//        /**
//         * Set the number of threads that erosion/dilation will use in case of
//         * fixed_thread disabled (fixed_thread=0, default).
//         * This value must be greater than 0.
//         */
//        __forceinline void setRowsThreadDivisor(const uint  &value) {
//            if(value < 1)
//                rows_thread_divisor = 1;
//            rows_thread_divisor = value;
//        }
        /**
         * Select the number of thread to use with erosion/dilation.
         * Disabling fixed thread, the number of threads will be the number of rows of the image
         * divided by the rows_thread_divisor.
         * 0: disable fixed thread;
         */
        __forceinline void setThreads(const uint &value) {
//            if(value < 0)
//                fixed_thread = 0;
//            fixed_thread = value;
            threads = value;
        }
        __forceinline const uint& getThreads() {
            return threads;
        }
//        __forceinline void disableFixedThread() {
//            fixed_thread = 0;
//        }
//        __forceinline bool isFixedThreadEnabled() {
//            return fixed_thread>0;
//        }

//        __forceinline const uint& getRowsThreadDivisor() {
//            return rows_thread_divisor;
//        }
//        __forceinline uint getNumberOfThreads() {
//            uint nThreads;
//            if(isFixedThreadEnabled())
//                nThreads = fixed_thread;
//            else nThreads = rows() / this->rows_thread_divisor;
//            return nThreads;
//        }


        __forceinline const int&   rows()  const { return _mat.rows(); }
        __forceinline const int&   cols()  const { return _mat.cols(); }


        // ~ ~ GETTERS ~ ~
        __forceinline const T&     get (int row, int col)  const { return _mat.get(row, col);   }
        __forceinline const T&     get (int index)         const { return _mat.get(index);      }
        __forceinline       T      getV(int row, int col)  const { return _mat.get(row, col);   }
        __forceinline       T      getV(int index)         const { return _mat.get(index);      }
        __forceinline Matrix<T>    getMatrix()             const { return _mat;                 }


        // ~ ~ SETTERS ~ ~
        __forceinline void         set(int row, int col, const T& value)   { _mat.set(row, col, value); }
        __forceinline void         set(int index, const T& value)          { _mat.set(index, value); }







        __forceinline void immerge(int topPadding, int rightPadding, int bottomPadding, int leftPadding, T borderValue) {
            //this->_mat = _mat.makeImmersion(topPadding, rightPadding, bottomPadding, leftPadding, borderValue);
           _mat.immerge(topPadding, rightPadding, bottomPadding,leftPadding, borderValue);
        }
        __forceinline GenericImage<T> makeImmersion(int topPadding, int rightPadding, int bottomPadding, int leftPadding, T borderValue) {
            return GenericImage(_mat.makeImmersion(topPadding, rightPadding, bottomPadding, leftPadding, borderValue));;
        }




        __forceinline void imshow(std::string msg = "") {
            cv::Mat m = _mat.toMatCV();
            cv::imshow(msg, m);
        }




        void erosion(const StructuringElement& SE )
        {

            //CV_Assert(bw_SE.depth() == CV_8U);  // accept only uchar Structuring Elements
            //CV_Assert(img.depth() == CV_8U);  // accept only uchar images
            //int nThreads = getNumberOfThreads();

            const Point<int>& SE_center = SE.getCenter();
            const int TOP_PADDING = SE_center.y();
            const int BOTTOM_PADDING = SE.rows() - SE_center.y();
            const int LEFT_PADDING = SE_center.x();
            const int RIGHT_PADDING = SE.cols() - SE_center.x();;

            GenericImage immersion = makeImmersion(TOP_PADDING, RIGHT_PADDING, BOTTOM_PADDING, LEFT_PADDING, 255);
            //immersion.imshow("dilation immersion"); //debug

            // num_threads(20)
            // #pragma omp parallel for schedule(dynamic, 4)
            #pragma omp parallel for num_threads(threads)
            for (int y = 0; y < rows(); y++)
            {
                int y_TOP_PADD = y+TOP_PADDING;//optimization

                for (int x = 0; x < cols(); x++)
                {
                    uchar min = immersion.get(y_TOP_PADD, x+LEFT_PADDING); // optimization //uchar max = copy.get(y + TOP_PADDING, x+LEFT_PADDING);

                    for (int i = 0; i < SE.rows(); i++)
                    {
                        int yi = y + i;
                        for (int j = 0; j < SE.cols(); j++)
                        {

                            #ifndef RECTANGLE_KERNEL_OPTIMIZATION
                            if (SE.get(i, j) == true)// if SE[i][j]==1, apply the mask:
                            {
                            #endif
                                //const uchar& current = copy.get(y + (i-SE_center.y()) + TOP_PADDING, x +  (j-SE_center.x()) + LEFT_PADDING);
                                const uchar current = immersion.get(yi, x + j);
                                if (current < min)
                                    min = current;
                            #ifndef RECTANGLE_KERNEL_OPTIMIZATION
                            }
                            #endif
                        }
                    }

                    set(y, x, min);
                }
            }

        }


        void dilation(const StructuringElement& SE )
        {
            //CV_Assert(bw_SE.depth() == CV_8U);  // accept only uchar Structuring Elements
            //CV_Assert(img.depth() == CV_8U);  // accept only uchar images

            const Point<int>& SE_center = SE.getCenter();
            const int TOP_PADDING = SE_center.y();
            const int BOTTOM_PADDING = SE.rows() - SE_center.y();
            const int LEFT_PADDING = SE_center.x();
            const int RIGHT_PADDING = SE.cols() - SE_center.x();;

            GenericImage immersion = makeImmersion(TOP_PADDING, RIGHT_PADDING, BOTTOM_PADDING, LEFT_PADDING, 255);
            //immersion.imshow("erosion immersion"); // debug


            //num_threads(20)num_threads(this->rows()/8)
            // #pragma omp parallel for num_threads(this->rows()/8)
            #pragma omp parallel for  num_threads(threads)
            for (int y = 0; y < rows(); y++)
            {
                int y_TOP_PADD = y+TOP_PADDING;//optimization
                for (int x = 0; x < cols(); x++)
                {
                    uchar max = immersion.get(y_TOP_PADD, x+LEFT_PADDING); // optimization //uchar max = copy.get(y + TOP_PADDING, x+LEFT_PADDING);

                    for (int i = 0; i < SE.rows(); i++)
                    {
                        int yi = y+i; // optimization

                        for (int j = 0; j < SE.cols(); j++)
                        {

                            #ifndef RECTANGLE_KERNEL_OPTIMIZATION
                            if (SE.get(i, j) == true)// if SE[i][j]==1, apply the mask:
                            {
                            #endif
                                //const uchar& current = copy.get(y + (i-SE_center.y()) + TOP_PADDING, x +  (j-SE_center.x()) + LEFT_PADDING);
                                const uchar current = immersion.get(yi, x + j); //optimization
                                if (current > max)
                                    max = current;
                            #ifndef RECTANGLE_KERNEL_OPTIMIZATION
                            }
                            #endif
                        }
                    }

                    set(y, x, max);
                }
            }

        }



        __forceinline void opening(const StructuringElement& SE )
        {
            erosion(SE);
            dilation(SE);
//            erosion_old(SE, SE_center);
//            dilation_old(SE, SE_center);
            // with erosion/dilation old, on imgBig.jpg - NagashLaptop - core i5 @ 2.38ghz ---> 10/11 seconds
            // with erosion/dilation optimized (immersion), on imgBig.jpg - NagashLaptop - core i5 @ 2.38ghz ---> 6/7 seconds
        }

        __forceinline void closure(const StructuringElement& SE )
        {
            dilation(SE);
            erosion(SE);
//            dilation_old(SE, SE_center);
//            erosion_old(SE, SE_center);
        }





        // shallow converter
        __forceinline cv::Mat toMatCV() const  {  return _mat.toMatCV();  }





// * * * static:      * * *


    public:
        static GenericImage<T> readImageCV(std::string path)
        {
            cv::Mat cvImg =  cv::imread( path, CV_LOAD_IMAGE_GRAYSCALE); //CV_LOAD_IMAGE_GRAYSCALE);
            using namespace cv;
            if ( !cvImg.data )
            {
                // printf("No image data \n");
                // TODO: generate exception
            }
            GenericImage<T> im = GenericImage<T>( Matrix<T>(cvImg.data, cvImg.rows, cvImg.cols) );
            return im;

        }



    };

    typedef GenericImage<uchar> Image;




}

#endif //DISPLAYIMAGE_IMAGE_H







































//
//void dilation_old(const Matrix<bool>& SE, const Point<int>& SE_center)
//{
//    //CV_Assert(bw_SE.depth() == CV_8U);  // accept only uchar Structuring Elements
//    //CV_Assert(img.depth() == CV_8U);  // accept only uchar images
//
//    long double sysTime1 = time(0);
//
//    Image copy = Image(*this);
//
//    for (int y = 0; y < this->_mat.rows(); y++)
//    {
//        for (int x = 0; x < this->_mat.cols(); x++)
//        {
//            uchar min = copy.get(y, x);
//
//            for (int i = 0; i < SE.rows(); i++)
//            {
//                for (int j = 0; j < SE.cols(); j++)
//                {
//                    if ((y - SE_center.y() + i) >= 0 && (x - SE_center.x() + j) >= 0 &&
//                        (y - SE_center.y() + i) < copy.rows() &&
//                        (x - SE_center.x() + j) < copy.cols())
//                    { // CHECK if SE is inside the image edges
//                        if (SE.get(i, j) == true)// if SE[i][j]==1, apply the mask:
//                        {
//                            const uchar &current = copy.get(y - SE_center.y() + i, x - SE_center.x() + j);
//                            if (current < min)
//                                min = current;
//                        }
//                    }
//                }
//            }
//
//            this->set(y, x, min);
//        }
//    }
//    long double sysTime2 = time(0);
//    std::cout << "\nDilation Profiling, elapsed: " << sysTime2-sysTime1 << std::endl;
//
//}
//
//inline void erosion_old(const Matrix<bool>& SE, const Point<int>& SE_center)
//{
//    //CV_Assert(bw_SE.depth() == CV_8U);  // accept only uchar Structuring Elements
//    //CV_Assert(img.depth() == CV_8U);  // accept only uchar images
//
//    long double sysTime1 = time(0);
//
//    Image copy = Image(*this);
//
//    for (int y = 0; y < copy.rows(); y++)
//    {
//        for (int x = 0; x < copy.cols(); x++)
//        {
//            uchar max = copy.get(y, x);
//            for (int i = 0; i < SE.rows(); i++)
//            {
//                for (int j = 0; j < SE.cols(); j++)
//                {
//
//                    if( (y - SE_center.y() + i) >= 0 && (x - SE_center.x() + j) >= 0 &&
//                        (y - SE_center.y() + i) < copy.rows() && (x - SE_center.x() + j) < copy.cols())
//                    { // CHECK if SE is inside the image edges
//
//                        if (SE.get(i, j) == true)// if SE[i][j]==1, apply the mask:
//                        {
//                            const uchar& current = copy.get(y - SE_center.y() + i, x - SE_center.x() + j);
//                            if (current > max)
//                                max = current;
//                        }
//                    }
//                }
//            }
//
//            this->set(y, x, max);
//        }
//    }
//    long double sysTime2 = time(0);
//    std::cout << "\nErosion Profiling, elapsed: " << sysTime2-sysTime1 << std::endl;
//
//}
