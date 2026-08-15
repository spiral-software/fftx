## Release Notes for FFTX Version 1.3.1

## Introduction

This is a maintenance / support release for FFTX.

## Supported Platforms

FFTX is supported on Windows, Linux, and MacOS with CPU backend, and on AMD HIP, NVIDIA CUDA, and Intel SYCL with GPU backend.

FFTX is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for a specific platform.

## Get FFTX Version 1.3.1

You can download the latest release from [spiral-software/fftx](https://github.com/spiral-software/fftx.git).

## Change Summary

The changes for this release fall in several areas and are primarily intended to make the
build process more robust as well as to address issues uncovered with the use of more
modern compilers and tool-chains.

**1. Build System, CMake Modernization, and Dynamic CUDA Architecture Detection**

* Implemented an automated `nvidia-smi` probe to detect the native local GPU compute
  capability (e.g., `sm_80`, `sm_90`) and set `CMAKE_CUDA_ARCHITECTURES` accordingly. 
* Implemented a version-safe fallback for headless builds (WSL, HPC login nodes, CI) based
  on `CMAKE_CUDA_COMPILER_VERSION` (defaults to Ampere `80`, appends Hopper `90` on CUDA
  11.8+, and Blackwell `120` on CUDA 12.8+).
* Removed obsolete hardcoded architecture lists.
* Added early `enable_language(CUDA)` invocation so compiler version checks work properly.
* Migrated to CMake-provided imported targets for CUDA dependencies (e.g., `CUDA::cufft`) instead of
  relying on legacy manual search directories and include path variables.
* Updated CMake configuration to search for each CUDA library component independently.
* Removed unnecessary dependency on `culibos`.
* Resolved linker paths for the `mpi` library target when linking GPU-aware MPI components.

**2. Fortran and Cross-Toolchain Interoperability**

* Explicitly set Fortran linker language for Fortran targets, eliminating the need to toggle or swap linkers
  when compiling with Intel SYCL or oneAPI toolchains. 
* Honor user-specified Fortran compilers passed via the command line (e.g.,
  `-DCMAKE_Fortran_COMPILER=...`) instead of allowing default overrides.
* Prioritize `gfortran` over `mpifort` on non-Windows (`!WIN32`) platforms.
* Ensure Fortran examples requiring MPI are only compiled when **both** a valid Fortran
  compiler and MPI libraries are detected on the system. 

**3. Test Harness and Build Reliability**

* Introduced a lightweight `smoketest` runner for quick validation and CI sanity checks.
* Retained and restructured the comprehensive test suite for full transform validation.
* Updated `make install` to install the `test_scripts/` directory alongside binaries and libraries.
* Serialized Fortran and documentation generation targets to prevent race conditions
  during parallel builds (`make -j`). 

**4. General Cleanup**

* Removed dependency on `helper_cuda.h` (a sample header not in core CUDA releases).
* Updated error reporting macros to report source file and line number when a CUDA runtime or FFT failure occurred.
* Added `hiprtc` library when building for HIP and qualified `memset` calls as
  `std::memset` (resolves build errors on newer ROCm/HIP releases, i.e., ROCm 7.0+). 

## New Features

None.

## Examples

* No new examples.

## Bug Fixes

* None.

## Known Issues

On the CPU backend, the `batch1ddft` and `batch1dprdft` examples work only in the read sequential, write sequential case, which is the default case `-r 0x0`.

## License

FFTX is open source software licensed under the terms of the Simplified BSD
License (see the [**LICENSE**](./License.txt) file for the full text).

----------------------------------------------------------------------------------------------------

## Release Notes for FFTX Version 1.3.0

### Introduction

This is a release for FFTX introducing xSDK compliance and general cleanup.

### Supported Platforms

FFTX is supported on Windows, Linux, and MacOS with CPU backend, and on AMD HIP, NVIDIA CUDA, and Intel SYCL with GPU backend.

FFTX is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for a specific platform.

### Get FFTX Version 1.3.0

You can download the latest release from:

https://github.com/spiral-software/fftx.git

## Change Summary

* This version is the first FFTX version fully compliant with Extreme-scale Scientific Software Development Kit (xSDK) .
* Examples rationalized to use a common argument style, exit with appropriate exit status.
* A test_scripts folder was added to support various flavors of test scripts.

### New Features

* xSDK compliance

#### General Cleanup:

* Generation of documentation (using **doxygen** and **sphinx**) was cleaned up with
  documetation for basic types and classes now produced.
* Copyright notice updated (or added) in files.
* Cleaned up and made more robust for building on Windows.

#### Examples:

* No new examples.

### Bug Fixes

* None.

### Known Issues

On the CPU backend, the `batch1ddft` and `batch1dprdft` examples work only in the read sequential, write sequential case, which is the default case `-r 0x0`.

## License

FFTX is open source software licensed under the terms of the Simplified BSD
License (see the [**LICENSE**](./License.txt) file for the full text).

----------------------------------------------------------------------------------------------------

## Release Notes for FFTX Version 1.1

### Introduction

This is a release for FFTX with several new features.

### Supported Platforms

FFTX is supported on Windows, Linux, and MacOS with CPU backend, and on AMD HIP, NVIDIA CUDA, and Intel SYCL with GPU backend.

FFTX is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for a specific platform.

### Get FFTX Version 1.1

You can download the latest release from:

https://github.com/spiral-software/fftx.git

## Change Summary

* This version includes run-time compilation for sizes that are not in the library.
* FFTX now supports a SYCL backend, in addition to HIP and CUDA.
* A Fortran example has been added.
* Version number is accessible with the macro `FFTX_VERSION` and with the function `fftx::version()`.

### New Features

* Add support for SYCL. On the SYCL backend, the `mddft`, `mdprdft`, `batch1ddft` and `batch1dprdft`  examples compare outputs of FFTX transforms against those of MKLFFT. The `3DDFT_mpi` example and the distributed version of the `fortran` example do not support the SYCL backend yet.
* New output streams `fftx::OutStream()` replacing `stdout`, which is its default setting, and likewise `fftx::ErrStream()` replacing `stdin`, which is its default setting.
* On the CPU backend, the `testmdddft` and `testmdprdft` examples compare results with FFTW if that library is found by CMake.

#### General Cleanup:

* Macros now have names beginning with FFTX_.
* Files that are `#include`d anywhere now all have names that begin with `fftx`.
* Updated module lists in `supercomputer-README.md`.
* `fftx3.hpp` is now named `fftx.hpp`, and `fftx3utilities.h` is now `fftxutilities.hpp`.

#### Examples:

* New fortran example in `examples/fortran` directory.

### Bug Fixes

* The `rconv` example that was failing on Apple CPU has been corrected on that platform.

### Known Issues

On the CPU backend, the `batch1ddft` and `batch1dprdft` examples work only in the read sequential, write sequential case, which is the default case `-r 0x0`.

## License

FFTX is open source software licensed under the terms of the Simplified BSD
License (see the [**LICENSE**](./License.txt) file for the full text).

----------------------------------------------------------------------------------------------------
## Release Notes for FFTX Version 1.0.3

### Introduction

This is a patch release for FFTX.

### Supported Platforms

FFTX is supported on Windows, Linux, and MacOS.

FFTX is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for a specific platform.

### Get FFTX Version 1.0.3

You can download the latest release from:

https://github.com/spiral-software/fftx.git

## Change Summary

* Update README.md with libraries built, APIs, linking with FFTX, installed pieces.
* Add support for MAC M1 (arm architecture)

### New Features

N/A

#### General Cleanup:

N/A

#### Examples:

N/A

### Bug Fixes

N/A

### Known Issues

N/A

## License

FFTX is open source software licensed under the terms of the Simplified BSD
License (see the [**LICENSE**](./License.txt) file for the full text).

----------------------------------------------------------------------------------------------------
## Release Notes for FFTX Version 1.0.1

### Introduction

This is a patch release for FFTX.

### Supported Platforms

FFTX is supported on Windows, Linux, and MacOS.

FFTX is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for a specific platform.

### Get FFTX Version 1.0.1

You can download the latest release from:

https://github.com/spiral-software/fftx.git

## Change Summary

* Check for python/python3 when building library codes; get library extension on MAC.
* Define INSTALL_RPATH for libs.
* Fix memory leak in fftx3; Use pointers to loop through arrays in utilities.
* README.md updates.
* Don't run hipify-perl for HIP codegen; make spiral output source.
* file depend on the generator script.
* HIP example programs cleaned up.

### New Features

N/A

#### General Cleanup:

N/A

#### Examples:

Several examples (also doubling as tests) show FFTX performance for serial code
(i.e., CPU) and well as for GPU (both NVIDIA CUDA and AMD HIP).

### Bug Fixes

N/A

### Known Issues

N/A

## License

FFTX is open source software licensed under the terms of the Simplified BSD
License (see the [**LICENSE**](./LICENSE) file for the full text).

----------------------------------------------------------------------------------------------------
## Release Notes for FFTX Version 1.0.0

### Introduction

This is the first release of FFTX with libraries of transforms.

### Supported Platforms

FFTX is supported on Windows, Linux, and MacOS.

FFTX is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for a specific platform.

### Get FFTX Version 1.0.0

You can download the latest release from:

https://github.com/spiral-software/fftx.git

## Change Summary

N/A, First major release. 

### New Features

#### General Cleanup:

#### Examples:

Several examples (also doubling as tests) show FFTX performance for serial code
(i.e., CPU) and well as for GPU (both NVIDIA CUDA and AMD HIP).

### Bug Fixes

N/A

### Known Issues

N/A

## License

FFTX is open source software licensed under the terms of the Simplified BSD
License (see the [**LICENSE**](./LICENSE) file for the full text).

----------------------------------------------------------------------------------------------------

## Release Notes for FFTX Version 0.9.0

### Introduction

This is a placeholder for release notes to accompany FFTX when it is
formally released.
