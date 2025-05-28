# xSDK Community Policy Compatibility for FFTX

This document summarizes the efforts of current and future xSDK member packages to achieve compatibility with the xSDK community policies version 1.0.0. Below only short descriptions of each policy are provided. The full description is available [here](https://github.com/xsdk-project/xsdk-community-policies)
and should be considered when filling out this form.

*** A good example of how to complete this form can be found in the [hypre version](https://github.com/xsdk-project/xsdk-policy-compatibility/blob/master/hypre-policy-compatibility.md).

Please, provide information on your compability status for each mandatory policy, and if possible also for recommended policies.
If you are not compatible, state what is lacking and what are your plans on how to achieve compliance.

For current xSDK member packages: If you were not fully compatible at some point, please describe the steps you undertook to fulfill the policy. This information will be helpful for future xSDK member packages.

**Website:** https://spiral-software.github.io/fftx/

### Mandatory Policies

| Policy                 |Support| Notes                   |
|------------------------|-------|-------------------------|
|**M1.** Support portable installation through Spack. |Full| FFTX supports and provides a Spack package. |
|**M2.** Provide a comprehensive test suite for correctness of installation verification. |Full| In folder `test_scripts`, FFTX has a shell script, `test_suite.sh`, and batch scripts for NERSC's perlmutter, `test_batch_perlmutter.sh`; for OLCF's frontier, `test_batch_frontier.sh`; and for ALCF's aurora, `test_suite_aurora.pbs`. |
|**M3.** Employ user-provided MPI communicator (no MPI_COMM_WORLD). Don't assume a full MPI 3 implementation without checking. Provide an option to prevent any changes to MPI error-handling if it is changed by default. |Full| FFTX functions that use MPI can take a user-defined MPI communicator. |
|**M4.** Give best effort at portability to key architectures (standard Linux distributions, GNU, Clang, vendor compilers, and target machines at ALCF, NERSC, OLCF). |Full| FFTX is tested on Ubuntu Linux with gcc compiler, on macOS with clang compiler, on aurora at ALCF with icpx compiler for SYCL, on perlmutter at NERSC with nvcc for CUDA, and on frontier at OLCF with hipcc for HIP. |
|**M5.** Provide a documented, reliable way to contact the development team. |Full| The github website has an "Issues" tab: https://github.com/spiral-software/fftx/issues |
|**M6.** Respect system resources and settings made by other previously called packages (e.g. signal handling). |Full| FFTX does not change these. | 
|**M7.** Come with an open source (BSD style) license. |Full| BSD license at https://github.com/spiral-software/fftx/blob/main/License.txt |
|**M8.** Provide a runtime API to return the current version number of the software. |Full| The main FFTX header `src/include/fftx.hpp` defines macro `FFTX_VERSION`, and there is also a function `fftx::version()` that returns a string containing the version number. |
|**M9.** Use a limited and well-defined symbol, macro, library, and include file name space. |Full| All C++ symbols are inside `namespace fftx`. All generated library (i.e., any `lib_fftx_*`) code, defines, or header files use "`fftx`" as a prefix. All preprocessor macros defined in FFTX library source begin with "`FFTX_`". |
|**M10.** Provide a publicly available repository. |Full| Public repository is available at https://github.com/spiral-software/fftx |
|**M11.** Have no hardwired print or IO statements that cannot be turned off. |Full| FFTX I/O can be turned off at runtime. |
|**M12.** For external dependencies, allow installing, building, and linking against an outside copy of external software. |Full| FFTX allows this. |
|**M13.** Install headers and libraries under \<prefix\>/include and \<prefix\>/lib. |Full| FFTX is installed in exactly this way. When building with `cmake` we specify: `-DCMAKE_INSTALL_PREFIX=$FFTX_HOME`. At install time, the header files, libraries, CMake includes, and any built example executables are installed to folders `include`, `lib`, `CMakeIncludes`, and `bin` under the location referenced as `$FFTX_HOME`.  The `CMakeIncludes` are to allow other packages to easily use FFTX with CMake. |
|**M14.** Be buildable using 64 bit pointers. 32 bit is optional. |Full| FFTX all builds with 64-bit pointers by default. |
|**M15.** All xSDK compatibility changes should be sustainable. |Full| All changes or updates will be applied to development and release versions by policy. |
|**M16.** Have a debug build option. |Full| When building FFTX with CMake, the flag `-DCMAKE_BUILD_TYPE=Debug` enables debug mode. |
|**M17.** Each xSDK member package should have sufficient documentation to support use and further development.  |Full| Full documentation for usage, example codes, and instructions for developers are at https://spiral-software.github.io/fftx/. |

### Recommended Policies

| Policy                 |Support| Notes                   |
|------------------------|-------|-------------------------|
|**R1.** At least one validation (smoke) test that can be invoked through the Spack package. |None| None. |
|**R2.** Possible to run test suite under valgrind in order to test for memory corruption issues. |None| Haven't tried this. |
|**R3.** Adopt and document consistent system for error conditions/exceptions. |None| None. |
|**R4.** Free all system resources acquired as soon as they are no longer needed. |Full| FFTX does this. |
|**R5.** Provide a mechanism to export ordered list of library dependencies. |None| None. |
|**R6.** Document versions of packages that it works with or depends upon, preferably in machine-readable form.  |None| None. |
|**R7.** Have README, SUPPORT, LICENSE, and CHANGELOG files in top directory.  |Partial| Repository has `README.md`, `License.txt`, and `ReleaseNotes.md`. |
|**R8.** Provide version comparison preprocessor macros.  |Full| In `src/include/fftx.hpp` there is `#define FFTX_VERSION`. |
