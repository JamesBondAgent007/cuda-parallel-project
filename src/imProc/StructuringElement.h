//
// Created by nagash on 22/10/16.
//

#ifndef STRUCTURED_STRUCTURINGELEMENT_H
#define STRUCTURED_STRUCTURINGELEMENT_H

#include "Matrix.h"
#include "Point.h"

namespace imProc
{

    class StructuringElement : public Matrix<bool>
    {
        Point<int> center;

    public:
        StructuringElement(int rows, int cols);

        const Point<int> getCenter() const { return center; }
    };

}

#endif //STRUCTURED_STRUCTURINGELEMENT_H
