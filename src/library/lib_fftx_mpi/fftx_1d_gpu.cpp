// implement embedded packing kernel
#include "fftxdevice_macros.h"
#include "fftx_1d_gpu.h"

//
//  Copyright (c) 2018-2025, Carnegie Mellon University
//  All rights reserved.
//
//  See LICENSE file for full information.

__global__
void
__embed(
    double2 *dst,
    double2 *src,
    int faster,
    int slower
) {
    int s = blockIdx.x;
    src += s * faster;
    dst += s * (2 * faster);

    int f = threadIdx.x;
    for (int i = f; i < 2*faster; i += blockDim.x) {
        if (i < faster/2) {
            dst[i] = {};
        } else if (i < 3 * faster/2) {
            dst[i] = src[i - faster/2];
        } else {
            dst[i] = {};
        }
    }
}

__global__
void
__embed(
    double2 *dst,
    double2 *src,
    int faster,
    int faster_padded,
    int slower
) {
    int s = blockIdx.x;
    src += s * faster_padded;
    dst += s * (2 * faster);

    int f = threadIdx.x;
    for (int i = f; i < 2*faster; i += blockDim.x) {
        if (i < faster/2) {
            dst[i] = {};
        } else if (i < 3 * faster/2) {
            dst[i] = src[i - faster/2];
        } else {
            dst[i] = {};
        }
    }
}

__global__
void
__embed(
    double2 *dst,
    double2 *src,
    size_t faster,
    size_t faster_padded,
    size_t slower,
    size_t copy_size
) {
    int f = blockIdx.x;
    int s = blockIdx.y;
    src += s * faster_padded*copy_size;
    dst += s * (2 * faster)*copy_size  + f * copy_size;

    if (f < faster/2) {
        for (int c = threadIdx.x; c < copy_size; c += blockDim.x) {
            dst[c] = {};
        }
    } else if (f < 3 * faster/2) {
        for (int c = threadIdx.x; c < copy_size; c += blockDim.x) {
            dst[c] = src[(f - faster/2)*copy_size + c];
        }
    } else {
        for (int c = threadIdx.x; c < copy_size; c += blockDim.x) {
            dst[c] = {};
        }
    }
}

FFTX_DEVICE_ERROR_T embed(
    std::complex<double> *dst,
    std::complex<double> *src,
    int faster,
    int slower
) {
    // pad fastest dim with zeros on either side in tensor of shape [b, 2a]
    __embed<<<dim3(slower), dim3(min(2*faster, 1024))>>>((double2 *) dst, (double2 *) src, faster, slower);
    FFTX_DEVICE_ERROR_T device_status = FFTX_DEVICE_SYNCHRONIZE();
	if (device_status != FFTX_DEVICE_SUCCESS) {
          // fprintf(stderr, "FFTX_DEVICE_SYNCHRONIZE returned error code %d after launching addKernel!\n", device_status);
          fftx::ErrStream() << "FFTX_DEVICE_SYNCHRONIZE returned error code " << device_status
                            << " after launching addKernel!" << std::endl;
		return device_status;
	}
	return FFTX_DEVICE_SUCCESS;

}

FFTX_DEVICE_ERROR_T embed(
    std::complex<double> *dst,
    std::complex<double> *src,
    int faster,
    int faster_padded,
    int slower
) {
    // faster_real <= faster_padded,
    // signal may be embedded in a padded region already,
    // get rid of padding as part of embedding.
    // pad fastest dim with zeros on either side in tensor of shape [b, 2a]
    __embed<<<dim3(slower), dim3(min(2*faster, 1024))>>>((double2 *) dst, (double2 *) src, faster, faster_padded, slower);
    FFTX_DEVICE_ERROR_T device_status = FFTX_DEVICE_SYNCHRONIZE();
	if (device_status != FFTX_DEVICE_SUCCESS) {
          // fprintf(stderr, "FFTX_DEVICE_SYNCHRONIZE returned error code %d after launching addKernel!\n", device_status);
          fftx::ErrStream() << "FFTX_DEVICE_SYNCHRONIZE returned error code " << device_status
                            << " after launching addKernel!" << std::endl;
		return device_status;
	}
	return FFTX_DEVICE_SUCCESS;

}

FFTX_DEVICE_ERROR_T embed(
    std::complex<double> *dst,
    std::complex<double> *src,
    size_t faster,
    size_t faster_padded,
    size_t slower,
    size_t copy_size
) {
    // faster_real <= faster_padded,
    // signal may be embedded in a padded region already,
    // get rid of padding as part of embedding.
    // pad fastest dim with zeros on either side in tensor of shape [b, 2a]
    __embed<<<dim3(2*faster, slower), dim3(min(copy_size, (size_t) 1024))>>>((double2 *) dst, (double2 *) src, faster, faster_padded, slower, copy_size);
    FFTX_DEVICE_ERROR_T device_status = FFTX_DEVICE_SYNCHRONIZE();
	if (device_status != FFTX_DEVICE_SUCCESS) {
          // fprintf(stderr, "FFTX_DEVICE_SYNCHRONIZE returned error code %d after launching addKernel!\n", device_status);
          fftx::ErrStream() << "FFTX_DEVICE_SYNCHRONIZE returned error code " << device_status
                            << " after launching addKernel!" << std::endl;
		return device_status;
	}
	return FFTX_DEVICE_SUCCESS;

}
