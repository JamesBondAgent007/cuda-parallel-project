/*
 * utils.h
 *
 *  Created on: Nov 1, 2016
 *      Author: Mr_Holmes
 */

#ifndef DISPLAYIMAGE_IMPROC_H
#define DISPLAYIMAGE_IMPROC_H


 #define VERBOSE_INLINE_STATUS

// #define DISABLE_INLINE
// #define DISABLE_FORCE_INLINE



#ifdef DISABLE_INLINE
    #define __forceinline
    #ifdef VERBOSE_INLINE_STATUS
        #pragma message ("Inline status: Inline disabled.")
    #endif
#else
    #ifdef DISABLE_FORCE_INLINE
        #define __forceinline inline
        #ifdef VERBOSE_INLINE_STATUS
            #pragma message ("Inline status: Force Inline disabled.")
        #endif
    #else
        #define __forceinline __attribute__((always_inline)) inline
        #ifdef VERBOSE_INLINE_STATUS
            #pragma message ("Inline status: Force Inline enabled.")
        #endif
    #endif
#endif
typedef unsigned char uchar;
typedef unsigned int uint;

#endif //DISPLAYIMAGE_IMPROC_H
