//
// Created by nagash on 22/10/16.
//

#include "StructuringElement.h"

imProc::StructuringElement::StructuringElement(int rows, int cols)
        : Matrix(rows, cols), center((int)ceil(cols/2), (int)ceil(rows/2)) {}

