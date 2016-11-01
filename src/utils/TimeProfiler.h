/*
 * TimeProfiler.h
 *
 *  Created on: Nov 1, 2016
 *      Author: Mr_Holmes
 */

#ifndef DISPLAYIMAGE_TIMEPROFILER_H
#define DISPLAYIMAGE_TIMEPROFILER_H

#include <iostream>
#include <sys/time.h>


#ifndef DST_NONE
    #define DST_NONE 0 // for linux ubuntu ..
#endif
class TimeProfiler
{
private:
    struct timeval t1, t2;
    bool running = false;
    bool resetted = true;
    long sec;
    long msec;
    long long usec;
public:
    TimeProfiler(bool start_now = false) {
        if(start_now) start();
        else reset();
    }

    inline void reset() {
        running = false;
        resetted = true;
        sec = -1;
        usec = -1;
    }

    inline void start() {
        reset();
        running = true;
        resetted = false;
        gettimeofday(&t1,DST_NONE);
    }

    inline void stop() {
        gettimeofday(&t2, DST_NONE);
        if(running && !resetted)
        {
            sec = t2.tv_sec - t1.tv_sec;
            usec = t2.tv_usec - t1.tv_usec;

            if(usec < 0)
            {
                usec += 1000000;
                sec  -=1;
            }

            msec = usec/1000;
            usec -= msec*1000;
            running = false;
        }
    }



    /**
     * @return: -1 if profiler has never been started
     */
    inline long getSec() const
    {
        if(resetted || running)
            return -1;
        else return sec;
    }

    /**
    * @return: -1 if profiler has never been started
    */
    inline long long getUsec() const
    {
        if(resetted || running)
            return -1;
        else return usec;
    }
    /**
    * @return: -1 if profiler has never been started
    */
    inline long long getMsec() const
    {
        if(resetted || running)
            return -1;
        else return msec;
    }

    inline double getDSeconds() {
        return sec + msec/1000.0 + usec/1000000.0;
    }

};

std::ostream &operator<<(std::ostream &os, TimeProfiler const &m);


#endif //DISPLAYIMAGE_TIMEPROFILER_H
