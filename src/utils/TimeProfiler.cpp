
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include "TimeProfiler.h"

std::ostream &operator<<(std::ostream &os, TimeProfiler const &m)  {
    //return os << m.getSec() << "s\t" << m.getUsec() << "uS";
    return os << m.getSec() << "s\t" << m.getMsec() << "mS\t" << m.getUsec() << "uS";

}
