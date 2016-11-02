//
// Created by Nagash on 09/05/16.
//

#ifndef DISPLAYIMAGE_POINT_H
#define DISPLAYIMAGE_POINT_H

#include "../utils/utils.h"

namespace imProc
{

    template <class T>
    class Point
    {
    private:
        T _y;
        T _x;


    public:

        // y, x ---> row, col
        Point(const T& y, const T& x) :  _y(y), _x(x) {}


        __forceinline const int &y() const { return _y; }
        __forceinline const int &x() const { return _x; }

        __forceinline void y(const T& new_y) { _y = new_y; }
        __forceinline void x(const T& new_x) { _x = new_x; }

    };
}

#endif //DISPLAYIMAGE_POINT_H
