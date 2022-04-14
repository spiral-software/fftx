## Release Notes for FFTX Version 1.0.3

### Introduction

This is a patch release for FFTX.

### Supported Platforms

FFTX is supported on Windows, Linux, and MacOS.

FFTX is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for a specific platform.

### Get FFTX Version 1.0.3

You can download the lastest release from:

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
License (see the [**LICENSE**](./LICENSE) file for the full text).

----------------------------------------------------------------------------------------------------
## Release Notes for FFTX Version 1.0.1

### Introduction

This is a patch release for FFTX.

### Supported Platforms

FFTX is supported on Windows, Linux, and MacOS.

FFTX is configured using **cmake** and is expected to run on most UNIX-like systems.

See the [**README**](./README.md) file for more information on how to build for a specific platform.

### Get FFTX Version 1.0.1

You can download the lastest release from:

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

You can download the lastest release from:

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
