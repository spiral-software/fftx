#include "fftx3.hpp"
#include "interface.hpp"
#include "mddftObj.hpp"
#include "imddftObj.hpp"
#include <string>
#include <fstream>
#include <fftw3.h>

#if defined FFTX_CUDA
#include "cudabackend.hpp"
#elif defined FFTX_HIP
#include "hipbackend.hpp"
#else  
#include "cpubackend.hpp"
#endif
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
#include "device_macros.h"
#endif
