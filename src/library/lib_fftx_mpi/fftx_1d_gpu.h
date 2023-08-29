#include <complex>
DEVICE_ERROR_T embed(
    std::complex<double> *dst,
    std::complex<double> *src,
    int a,
    int b
);

DEVICE_ERROR_T embed(
    std::complex<double> *dst,
    std::complex<double> *src,
    int faster,
    int faster_padded,
    int slower
);

DEVICE_ERROR_T embed(
    std::complex<double> *dst,
    std::complex<double> *src,
    size_t faster,
    size_t faster_padded,
    size_t slower,
    size_t copy_size
);