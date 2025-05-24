//
//  Copyright (c) 2018-2025, Carnegie Mellon University
//  All rights reserved.
//
//  See LICENSE file for full information.
//

#include <complex>
FFTX_DEVICE_ERROR_T embed(
    std::complex<double> *dst,
    std::complex<double> *src,
    int a,
    int b
);

FFTX_DEVICE_ERROR_T embed(
    std::complex<double> *dst,
    std::complex<double> *src,
    int faster,
    int faster_padded,
    int slower
);

FFTX_DEVICE_ERROR_T embed(
    std::complex<double> *dst,
    std::complex<double> *src,
    size_t faster,
    size_t faster_padded,
    size_t slower,
    size_t copy_size
);
