FFTX Project
============

This is the public repository for the **FFTX** API source, examples, and documentation.

## Building FFTX

### Prerequisites

There are several pre-requisites that must be installed prior to building and
using **FFTX**.  You will need:

* **CMake**, minimum version 3.14; however, a more-modern version is preferred
as some deprecated features must be used with older versions.
* if on Linux/Unix, **gcc** and **make**.
* if on Windows, **Visual Studio**, and an ability to run **bash** shell
scripts.  You can use the Git Bash shell available with **git**, but other
shells such as Cygwin or msys will also work.
* if on macOS, version 10.14 (Mojave) or later of macOS, with a compatible
version of **Xcode** and **Xcode Command Line Tools**.
* **python**, version 3.6 or higher.  On some systems, both **python** (usually
version 2.7) and **python3** exist.  The scripts used to create the **FFTX** library
source code check the version of **python**, and if it is version 2.X it will
try to run **python3** instead.  A user, therefore, should not have to worry
whether **python** or **python3** comes first in the user's path.

To build and use **FFTX**, follow these steps:
1. [Install **SPIRAL** and associated packages.](#1-install-spiral-and-associated-packages)
2. [Clone the **FFTX** repository.](#2-clone-the-fftx-repository)
3. [Generate library source code.](#3-generate-library-source-code)
4. [Compile library source code and examples.](#4-compile-library-source-code-and-examples)

### 1. Install SPIRAL and associated packages

If you already have SPIRAL installed (and have the **SPIRAL_HOME** environment variable
set) FFTX will use that installation and you can skip to step 2, "Clone the FFTX repository".

If you want to manually install SPIRAL follow these steps; alternatively, skip to step 2,
"Clone the FFTX repository" and have FFTX pull down the necessary SPIRAL repositories and
perform the build steps.

Clone **spiral-software** (available [**here**](https://www.github.com/spiral-software/spiral-software))
to a location on your computer.  E.g.,:
```
cd ~/work
git clone https://www.github.com/spiral-software/spiral-software
```
This location is known as *SPIRAL HOME*, and you must set an environment variable
**SPIRAL_HOME** (here, **`~/work/spiral-software`**) to point to this location later.

**FFTX** requires the **SPIRAL** packages
[**fftx**](https://www.github.com/spiral-software/spiral-package-fftx),
[**simt**](https://www.github.com/spiral-software/spiral-package-simt),
[**mpi**](https://www.github.com/spiral-software/spiral-package-mpi), and
[**jit**](https://www.github.com/spiral-software/spiral-package-jit).

You need to download these separately, as follows:
```
cd $SPIRAL_HOME/namespaces/packages
git clone https://www.github.com/spiral-software/spiral-package-fftx fftx
git clone https://www.github.com/spiral-software/spiral-package-simt simt
git clone https://www.github.com/spiral-software/spiral-package-mpi mpi
git clone https://www.github.com/spiral-software/spiral-package-jit jit
```
**NOTES:**
* The **SPIRAL** packages must be installed under directory **$SPIRAL_HOME/namespaces/packages**
and must be placed in folders with the prefix *spiral-package-* removed. 
* If you already have **spiral-software** installed, please refresh the
installation to ensure you're up-to-date, especially for the **SPIRAL** packages.
* It is preferable to download the **SPIRAL** packages *before* performing
the **SPIRAL** build steps.

Follow the build instructions for **spiral-software** (see the **README**
[**here**](https://github.com/spiral-software/spiral-software/blob/master/README.md) ).

### 2. Clone the FFTX repository.

Clone **FFTX** to a location on your computer.  E.g.,
```
cd ~/work
git clone https://www.github.com/spiral-software/fftx
cd fftx
```
Set the environment variable **FFTX_HOME** to point to the directory where
you want to install **FFTX** (which is not necessarily the same directory
where you have cloned **FFTX**; you may want to have separate installation
directories for different backends).

If you have not already installed SPIRAL you can have it downloaded and built from the
repositories by sourcing the **get_spiral.sh** shell script now.  This script checks the
definition of the **SPIRAL_HOME** environment variable, and if undefined it will get the
SPIRAL code, build it and export a definition for **SPIRAL_HOME**.  Make sure you source
(vs run) the script:
```
. get_spiral.sh
or
source get_spiral.sh
```

### 3. Generate library source code.

**FFTX** builds libraries of transforms for a set of different sizes.  The library
source code is generated from **SPIRAL** script specifications, and may be
created before building **FFTX** itself.  If the sizes are pre-built the code will be
added to libraries of pre-defined fixed size.  Alternatively, sizes not defined can have
the code generated and compiled at run-time (RTC).

Before creating the library source code consider if you will be
running on CPU only, or also utilizing a GPU.  If you create all the source code
(and related **cmake** scripts and library APIs) for GPU and then try building
for CPU only you may encounter compiler errors or unexpected results.

The shell script **config-fftx-libs.sh** is a utility script in the **FFTX** home
directory that allows you to configure which libraries (and by extension, which
examples) are to be built.  Each library has a flag (true/false) stating whether
it will be built or not.  There is also a flag allowing the building of the
example programs to be skipped (useful when you only need to build the libraries
for an external application).  The script is self-documenting; edit the script
to set the flags for the libraries either **true** or **false**.  Once you have
made the appropriate choices simply run the script:
```
./config-fftx-libs.sh
```
No arguments are required for this script.  This script runs the
**build-lib-code.sh** script in the **src/library** directory and will marshall
the resources and options needed for the set of libraries selected.  This step
can take quite a long time depending on the number of transforms and set of
sizes to create.  The code is targeted to run on a CPU, a GPU (either CUDA or
HIP) depending on the selections made in the configure script.  Depending on the
number of sizes being built for each transform this process can take a
considerable amount of time.  By default, only a very small number of fixed sizes will be
created for the fixed-size libraries. 

The text file **`build-lib-code-failures.txt`** will contain a list of all library
transforms that failed to generate in this step.

Running **config-fftx-libs.sh** also creates a file called **options.cmake**.
The options defined in this file indicate the library choices made in the shell
script and are used by **CMake** to determine which examples are to be built; as
a rule, one should not attempt to build an example whose dependent libraries are
not built.

### 4. Compile library source code and examples.

From your **FFTX** home directory, set up a **build** folder (which can be given
any name, and you may want to have separate ones for different backends).  When
you configure using **CMake** you must specify the install prefix that **CMake**
should use (the default location for **CMake** may be a directory for which you
do not have write privilidges).  Do that by setting the environment variable
**FFTX_HOME** and specifying either the directory path or the environment
variable on the **CMake** command line.

**NOTES:**
* You will need **FFTX_HOME** set in order to use or reference **FFTX** artifacts externally.<br>
* **FFTX_HOME** may be, but does not have to be, the same as your **FFTX** home directory.
* Some tips for building on a supercomputing platform at NERSC or OLCF are available
[**here**.](https//github.com/spiral-software/spiral-software/blob/master/supercomputer-README.md)

```
export FFTX_HOME=~/work/fftx
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=~/work/fftx -D_codegen=CPU ..      # build for CPU, *or*
cmake -DCMAKE_INSTALL_PREFIX=$FFTX_HOME  -D_codegen=CUDA ..     # build for CUDA, *or*
cmake -DCMAKE_INSTALL_PREFIX=~/work/fftx -DCMAKE_CXX_COMPILER=hipcc -D_codegen=HIP ..      # build for HIP
make install
```

#### Building on Windows

**FFTX** can be built on Windows, however, you need to be able to run a [bash]
shell script as mentioned above to build the library source code.  To build
**FFTX**, edit **config-fftx-libs.sh** as described above, then open a shell and
do the following:
```
cd fftx
./config-fftx-libs.sh
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=~/work/fftx -D_codegen=CUDA ..
cmake --build . --target install --config Release
```
This shows an example building for CUDA on Windows, you can also build for CPU
or AMD HIP as shown above (under Building for Linux).

When **FFTX** is built, the final step (of *make install*) creates a tree structure
(at the location specified by **CMAKE_INSTALL_PREFIX**).  The following
directories will be created/populated:

|Directory Name|Description|
|:-----|:-----|
|**./CMakeIncludes**|CMake include files and functions to ease integration with **FFTX**|
|**./bin**|Executables for example programs built as part of the **FFTX** distribution|
|**./lib**|**FFTX** libraries, that can be called by external applications|
|**./include**|Include files for using **FFTX** libraries|
|**./cache_jit_files**|Folder containing the RTC code generated for any transform not <br>found in a fixed-size library|

## Running FFTX Example Programs

After building **FFTX**, to run the programs that are in the **examples** subtree, simply do:
```
cd $FFTX_HOME/bin
./testcompare_device
./testverify
```
etc.  Since we set **RPATH** to point to where the libraries are installed, you
likely will not need to adjust the library path variable, typically
**LD_LIBRARY_PATH**.

The README in the **examples** folder
[**here**](https://github.com/spiral-software/fftx/blob/master/examples/README.md)
contains a list of examples, how to run them, and how to add a new example.

## Libraries

The libraries built and copied to the **$FFTX_HOME/lib** folder can be used by
external applications to leverage **FFTX** transforms.  To access the necessary
include files and libraries, an external application's **CMakeLists.txt** should
include **CMakeInclude/FFTXCmakeFunctions.cmake**.  A full example of an
external application linking with **FFTX** libraries is available in the
[**fftx-demo-extern-app**](https://www.github.com/spiral-software/fftx-demo-extern-app).

### FFTX Libraries Built

**FFTX** builds libraries for 1D and 3D FFTs for a single device.  FFTs are
built for a set of specific sizes, thus not all possible sizes will be found in
the libraries.  There are default files specifying the sizes to build for each
of: 1D FFTs, 3D FFTS, and distributed FFTs (defaults file names below).  You can
customize the set of sizes to build by either editing these files **or** provide
your own files -- just override the default file name in the
**config-fftx-libs.sh** script.  If you provide your own file(s) just follow the
format shown in the default file(s).  The default files specifying the sizes to
build for each group of transforms are (all files are in the src/library
folder):

|Type|File Name|Description|
|:-----:|:-----|:-----|
|1D FFT|dftbatch-sizes.txt|Batch of 1D FFTs|
|3D FFT|cube-sizes-cpu.txt|3D FFTs for CPU| 
|3D FFT|cube-sizes-gpu.txt|3D FFTs for GPU| 
|Distributed FFT|distdft_sizes.txt|Distributed FFTs|

The following is a list of libraries potentially built:

|Type|Name|Description|
|:-----:|:-----|:-----|
|3D FFT|fftx_mddft|Forward 3D FFT complex to complex|
|3D FFT|fftx_imddft|Inverse 3D FFT complex to complex|
|3D FFT|fftx_mdprdft|Forward 3D FFT real to complex|
|3D FFT|fftx_imdprdft|Inverse 3D FFT complex to real|
|3D Convolution|fftx_rconv|3D real convolution|
|1D FFT|fftx_dftbat|Forward batch of 1D FFT complex to complex (in development)|
|1D FFT|fftx_idftbat|Inverse batch of 1D FFT complex to complex (in development)|
|1D FFT|fftx_prdftbat|Forward batch of 1D FFT real to complex (in development)|
|1D FFT|fftx_iprdftbat|Inverse batch of 1D FFT complex to real (in development)|
|Distributed FFT|fftx_distdft|Distributed 3D FFT complex to complex (in development)|
|Distributed embedded FFT|fftx_distdft|Distributed embedded 3D FFT complex to complex (in development)|

### Library API

The following example shows usage of the 3D FFT complex-to-complex transform
(others are similar; just use the appropriate names and header file(s) from the
table above).  The user (calling application) is responsible for setting up
memory buffers and allocations as required (i.e., host memory for serial code,
device memory for GPU code).  The following abstract assumes device memory
buffers are already allocated/initialized as needed; error checking is omitted
for clarity and brevity:

```
#include "fftx3.hpp"
#include "transformlib.hpp"

    std::vector<int> sizes{ mm, nn, kk };         // 'cube' dimensions
    std::vector<void *> args{ dY, dX, dsym };     // pointers to the Output, Input, and symbol arrays

    MDDFTProblem mdp ( args, sizes, "mddft" );    // Define transform

    mdp.transform();                              // Run the transform
```

If the size specified with the transform definition is found in a library then that code
is executed; however, if it is not found in  a library then RTC is invoked to generate and
compile the necessary code (this is also cached for future use).

### Linking Against FFTX Libraries

**FFTX** provides a **cmake** include file, **FFTXCmakeFunctions.cmake**, that
provides functions to facilitate compiling and linking external applications
with the **FFTX** libraries.  An external application should include this file
(**$FFTX_HOME/CMakeIncludes/FFTXCmakeFunctions.cmake**) in order to access the
following helper functions to compile/link with the **FFTX** libraries.  Two
functions are available:

1.  **FFTX_find_libraries**() -- This function finds the **FFTX** libraries, linker
library path, and include file paths and exposes the following variables:
|CMake Variable Name|Description|
|:-----|:-----|
|**FFTX_LIB_INCLUDE_PATHS**|Include paths for **FFTX** include & library headers|
|**FFTX_LIB_NAMES**|List of **FFTX** libraries|
|**FFTX_LIB_LIBRARY_PATH**|Path to libraries (for linker)|
2.  **FFTX_add_includes_libs_to_target** ( target ) -- This function adds the
include file paths, the linker library path, and the library names to the
specified target.

An application will typically need only call
**FFTX_add_includes_libs_to_target**(), and let **FFTX** handle the assignment
of paths, etc. to the target.  Only if an application specifically needs to
access the named variables above is it necessary to call
**FFTX_find_libraries**().

### External Application Linking With FFTX

A complete example of an external application that builds test programs
utilizing the **FFTX** libraries is available at 
[**fftx-demo-extern-app**](https://www.github.com/spiral-software/fftx-demo-extern-app).
If you're interested in how to link an external application with **FFTX** please
download this example and review the **`CMakeLists.txt`** therein for specific
details.
