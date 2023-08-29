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