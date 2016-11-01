#ifndef STRUCTURED_BENCH_H
#define STRUCTURED_BENCH_H

#include <iostream>
#include "../utils/utils.h"
#include "../utils/TimeProfiler.h"


namespace convolutionBench
{
    // Curiously recurring template pattern
    template <class T>
    class IConvBench
    {
    private:



    protected:
        virtual void onPreRun() = 0;
        void run() { static_cast<T*>(this)->run(); };  // Curiously recurring template pattern: Static polymorphism
        virtual void onPostRun() = 0;


    public:

        IConvBench() {}

        virtual void init(std::string imgPath, unsigned int threads, unsigned int se_width, unsigned int se_height, bool useThreadsAsDivisor) = 0;




        virtual void showOriginalImg() = 0;
        virtual void showProcessedImg() = 0;


        virtual const unsigned int getSeWidth()        const = 0;
        virtual const unsigned int getSeHeight()       const = 0;
        virtual const unsigned int getImgWidth()       const = 0;
        virtual const unsigned int getImgHeight()      const = 0;
        virtual const std::string getImgPath() const = 0;



        TimeProfiler start(unsigned int loops) {
            TimeProfiler Timer;
            onPreRun();
            Timer.start();
            for (int i = 0; i < loops; i++)
                run();
            Timer.stop();
            onPostRun();
            // TODO: make T immutable before return
            return Timer;
        }
        TimeProfiler start() {
            TimeProfiler Timer;
            onPreRun();
            Timer.start();
            run();
            Timer.stop();
            onPostRun();
            // TODO: make T immutable before return
            return Timer;
        }


        //        virtual unsigned int getImageRows() = 0;
//        virtual unsigned int getImageCols() = 0;
    };

}

#endif //STRUCTURED_BENCH_H
