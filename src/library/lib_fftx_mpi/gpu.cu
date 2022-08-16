#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <malloc.h>
#include <complex>
#include "gpu.h"

#define FFTX_CUDA 1
#include "device_macros.h"

using namespace std;

__global__ void __pack(
	std::complex<double> *dst,
	std::complex<double> *src,
	size_t a_dim,
	size_t a_i_stride,
	size_t a_o_stride,
	size_t b_dim,
	size_t b_i_stride,
	size_t b_o_stride,
	size_t copy_size, 
	size_t batch
) {
	size_t a_id = blockIdx.x; // index into dim_out
	size_t b_id = blockIdx.y; // index into dim_in
	size_t batch_id = blockIdx.z;
	size_t src_offset = a_id*a_i_stride + b_id*b_i_stride + batch_id*a_dim*b_dim*copy_size;
	src += src_offset;
	size_t dst_offset = a_id*a_o_stride + b_id*b_o_stride + batch_id*a_dim*b_dim*copy_size;
	dst += dst_offset;

	for (size_t e_id = threadIdx.x; e_id < copy_size; e_id += blockDim.x) {
		dst[e_id] = src[e_id];
	}
}

// slowest to fastest
// [a, b, c] -> [b, 2a, c]
__global__ void __pack_embed(
	double2 *dst,
	double2 *src,
	size_t a,
	size_t b,
	size_t c
) {
	size_t a_id = blockIdx.x; // index into dim_out
	size_t b_id = blockIdx.y; // index into dim_in
	size_t src_offset = a_id * 1*b*c + b_id * c;
	src += src_offset;
	size_t off = a_id < a/2 ? 0 : 3*a/2;
	double2 *dst_zero   = dst + b_id * 2*a*c + (a_id + off) * c;
	double2 *dst_offset = dst + b_id * 2*a*c + (a_id + a/2) * c;

	for (size_t c_id = threadIdx.x; c_id < c; c_id += blockDim.x) {
		double2 zero = {};
		dst_zero  [c_id] = zero;
		dst_offset[c_id] = src[c_id];
	}

}


DEVICE_ERROR_T pack(
	std::complex<double> *dst,
	std::complex<double> *src,
	size_t a_dim,
	size_t a_i_stride,
	size_t a_o_stride,
	size_t b_dim,
	size_t b_i_stride,
	size_t b_o_stride,
	size_t copy_size, 
	size_t batch
) {
	__pack<<<dim3(a_dim, b_dim, batch), dim3(min(copy_size, (size_t) 1024))>>>(dst, src, a_dim, a_i_stride, a_o_stride, b_dim, b_i_stride, b_o_stride, copy_size, batch);
	DEVICE_ERROR_T device_status = DEVICE_SYNCHRONIZE();
	if (device_status != DEVICE_SUCCESS) {
		fprintf(stderr, "DEVICE_SYNCHRONIZE returned error code %d after launching addKernel!\n", device_status);
		return device_status;
	}
	return DEVICE_SUCCESS;
}

DEVICE_ERROR_T pack_embedded(
	std::complex<double> *dst,
	std::complex<double> *src,
	size_t a,
	size_t b,
	size_t c
) {
	__pack_embed<<<dim3(a, b), dim3(min(c, (size_t) 1024))>>>((double2 *) dst, (double2 *) src, a, b, c);
	DEVICE_ERROR_T device_status = DEVICE_SYNCHRONIZE();
	if (device_status != DEVICE_SUCCESS) {
		fprintf(stderr, "DEVICE_SYNCHRONIZE returned error code %d after launching addKernel!\n", device_status);
		return device_status;
	}
	return DEVICE_SUCCESS;
}


void execute_packing(size_t cp_size,
		     size_t a_dim, size_t b_dim,
		     std::complex<double> *src,
		     std::complex<double> *dst     )
      {
      size_t a_i_stride =     1 * cp_size;
      size_t a_o_stride = b_dim * cp_size;

      size_t b_i_stride = a_dim * cp_size;
      size_t b_o_stride =     1 * cp_size;
		
      DEVICE_ERROR_T err = pack(
			     dst,
			     src,
        a_dim, a_i_stride, a_o_stride,
        b_dim, b_i_stride, b_o_stride,
        cp_size,
        1
      );
      if (err != DEVICE_SUCCESS) {
          fprintf(stderr, "pack failed!\n");
      }
    }
