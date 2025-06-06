//
//  Copyright (c) 2018-2025, Carnegie Mellon University
//  All rights reserved.
//
//  See LICENSE file for full information.
//

// #include <stdio.h>
#include <stdlib.h>
//  #include <sys/time.h>
#include <malloc.h>

#include <complex>

#include "fftxdevice_macros.h"

FFTX_DEVICE_ERROR_T pack(
	std::complex<double> *dst,
	std::complex<double> *src,
	size_t a_dim,
	size_t a_i_stride,
	size_t a_o_stride,
	size_t b_dim,
	size_t b_i_stride,
	size_t b_o_stride,
	size_t copy_size
);


// slowest to fastest
// [a, b, c] -> [b, a, 2c]
FFTX_DEVICE_ERROR_T pack_embedded(
	std::complex<double> *dst,
	std::complex<double> *src,
	size_t a,
	size_t b,
	size_t c
);


FFTX_DEVICE_ERROR_T unpack(
	std::complex<double> *dst,
	std::complex<double> *src,
	size_t a_dim,
	size_t a_i_stride,
	size_t a_o_stride,
	size_t b_dim,
	size_t b_i_stride,
	size_t b_o_stride,
	size_t copy_size
);


// slowest to fastest
// [a, b, c] -> [b, a, 2c]
FFTX_DEVICE_ERROR_T unpack_embedded(
	std::complex<double> *dst,
	std::complex<double> *src,
	size_t a,
	size_t b,
	size_t c
);



void execute_packing(size_t cp_size,
		     size_t a_dim, size_t b_dim,
		     std::complex<double> *src,
		     std::complex<double> *dst
		     );
