#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <malloc.h>
#include <complex>
#include "gpu.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cufft.h"

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
	size_t batch
) {
	__pack<<<dim3(a_dim, b_dim, batch), dim3(min(copy_size, (size_t) 1024))>>>(dst, src, a_dim, a_i_stride, a_o_stride, b_dim, b_i_stride, b_o_stride, copy_size, batch);
	cudaError_t cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    return cudaStatus;
  }
	return cudaSuccess;
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
		
      cudaError_t err = pack(
			     dst,
			     src,
        a_dim, a_i_stride, a_o_stride,
        b_dim, b_i_stride, b_o_stride,
        cp_size,
        1
      );
      if (err != cudaSuccess) {
          fprintf(stderr, "pack failed!\n");
      }
    }
