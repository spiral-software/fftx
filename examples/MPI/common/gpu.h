#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <malloc.h>

#include <complex>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"

cudaError_t pack(
	std::complex<double> *dst,
	std::complex<double> *src,
	size_t a_dim,
	size_t a_i_stride,
	size_t a_o_stride,
	size_t b_dim,
	size_t b_i_stride,
	size_t b_o_stride,
	size_t copy_size,
	size_t batch_size
);

void execute_packing(size_t cp_size,
		     size_t a_dim, size_t b_dim,
		     std::complex<double> *src,
		     std::complex<double> *dst
		     );
