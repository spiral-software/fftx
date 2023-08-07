// implement embedded packing kernel
#include "device_macros.h"
#include "fftx_1d_gpu.h"

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

DEVICE_ERROR_T embed(
    std::complex<double> *dst,
    std::complex<double> *src,
    int faster,
    int slower
) {
    // pad fastest dim with zeros on either side in tensor of shape [b, 2a]
    __embed<<<dim3(slower), dim3(min(2*faster, 1024))>>>((double2 *) dst, (double2 *) src, faster, slower);
    DEVICE_ERROR_T device_status = DEVICE_SYNCHRONIZE();
	if (device_status != DEVICE_SUCCESS) {
		fprintf(stderr, "DEVICE_SYNCHRONIZE returned error code %d after launching addKernel!\n", device_status);
		return device_status;
	}
	return DEVICE_SUCCESS;

}