FFTX Project
============

This is the public repository for the FFTX API source, examples, and documentation.

## Building FFTX

To use and build FFTX you must install the following pre-requisites:<br><br>
    1.  **spiral-software**, available [**here**.](https://www.github.com/spiral-software/spiral-software)<br>
    2.  **spiral-package-fftx**, available [**here**.](https://www.github.com/spiral-software/spiral-package-fftx)<br>
    3.  **spiral-package-simt**, available [**here**.](https://www.github.com/spiral-software/spiral-package-simt)<br>

### C Compiler and Build Tools

SPIRAL builds on Linux/Unix with **gcc** and **make**, on Windows it builds with **Visual Studio**.

For macOS SPIRAL requires version 10.14 (Mojave) or later of macOS, with a compatible version of **Xcode** and
and **Xcode Command Line Tools**. 

### Installing Pre-requisites

#### spiral-software
Tools required on the target machine in order to build SPIRAL and FFTX, include:
<br>
**cmake**, version 3.14 or higher
<br>
**python**, version 3.6 or higher

Clone **spiral-software** to a location on you computer.  E.g., do:
```
cd ~/work
git clone https://www.github.com/spiral-software/spiral-software
```
This location is known as *SPIRAL HOME* and you must set an environment variable
**SPIRAL_HOME** to point to this location later.

To install the spiral packages do the following:
```
cd ~/work/spiral-software/namespaces/packages
git clone https://www.github.com/spiral-software/spiral-package-fftx fftx
git clone https://www.github.com/spiral-software/spiral-package-simt simt
git clone https://www.github.com/spiral-software/spiral-package-mpi mpi
```
**NOTE:** The spiral packages must be installed under directory
**$SPIRAL_HOME/namespaces/packages** and must be placed in folders with the
prefix *spiral-package* removed. 

Follow the build instructions for **spiral-software** (see the **README**
[**here**](https://github.com/spiral-software/spiral-software/blob/master/README.md) ).

#### Python 3

FFTX, like SPIRAL, requires **Python 3**.

On some systems both **python** (usually version 2.7) and **python3** exist.
The scripts used to create the FFTX library source code check the version of
**python**, and if it is version 2.X it will try to run **python3** instead.  A
user therefore, should not have to worry whether **python** or **python3** come
first in the user's path.

### Installing FFTX

Clone **FFTX** to a location on your computer.  E.g., do:
```
cd ~/work
git clone https://www.github.com/spiral-software/fftx
```
**NOTE:** Before attempting to build ensure you have set environment variable
**SPIRAL_HOME** to point to your **spiral-software** instance.

Also set the environment variable **FFTX_HOME** to point to the directory where
you have cloned **FFTX** (so `~/work/fftx` if following the exact commands above).

#### Building on Linux or Linux-like Systems

FFTX builds libraries of transforms for different sizes.  The library source
code is generated from Spiral script specifications, and must be created before
building FFTX.  Before creating the library source code consider if you will be
running on CPU only or also utilizing a GPU.  If you create all the source code
(and related **cmake** scripts and library APIs) for GPU and then try building
for CPU only you may encounter compiler errors or unexpected results.

The shell script **build-lib-code.sh** builds the library code.  The script
takes one optional argument to specify what code to build.  Serial code (CPU) is
always built, GPU code is built when the argument passed is either **CUDA** or
**HIP**.  Serial code is built if no argument is given or if the argument is
**CPU**.

To create the library source code do the following:
```
cd fftx				## your FFTX install directory
cd src/library
./build-lib-code.sh CUDA	## build CUDA code
cd ../..
```
This step can take quite a long time depending on the number of transforms and
set of sizes to create.  The code is targeted to run on the CPU, and code is
created targeted to run on a GPU (CUDA or HIP) depending on the argument given
to the build script.  Depending on the number of sizes being built for each
transform this process can take a considerable amount of time.

Next, run **cmake** and build the software:
```
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=~/work/fftx -D_codegen=CPU ..      # build for CPU, *or*
cmake -DCMAKE_INSTALL_PREFIX=~/work/fftx -D_codegen=CUDA ..     # build for CUDA, *or*
cmake -DCMAKE_INSTALL_PREFIX=~/work/fftx -DCMAKE_CXX_COMPILER=hipcc -D_codegen=HIP ..      # build for HIP
make install
```

#### Building on Windows

FFTX can be built on Windows, however, you need to be able to run a [bash] shell
script to build the library source code.  The easiest option to accomplish this
may be using the Git Bash shell available with **git** (other shells such as
Cygwin or msys will also work).  To build FFTX, open a shell and do the
following:
```
cd fftx
cd src/library
./build-lib-code.sh CUDA
cd ../..
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=~/work/fftx -D_codegen=CUDA ..
cmake --build . --target install --config Release
```
This shows an example building for CUDA on Windows, you can also build for CPU
or AMD HIP as shown above (under Building for Linux).

#### Running FFTX Example Programs

Currently, **FFTX** builds a number of example programs; the programs will be
installed in the location specified by **CMAKE_INSTALL_PREFIX**.  This often
defaults to a system location, such as /usr/local, to which you may not have
write priviliges; thus it is best to specify **CMAKE_INSTALL_PREFIX** explicitly
on the **cmake** command line (as shown above).  A reasonable option is the root
of your FFTX tree (e.g., ~/work/fftx).  The example programs are written to a
**bin** folder and the libraries created are written to a **lib** folder.  To
run the programs simply do (we set RPATH to point to where the libraries are
installed so you likelly will not need to adjust the library path variable,
typically **LD_LIBRARY_PATH**):
```
cd bin
./testcompare_device
./testverify
```

The libraries built and copied to the **lib** folder can be used by external
applications to leverage FFTX transforms.  To access the necessary include files
and libraries an external application's **cmake** should include
**CMakeInclude/FFTXCmakeFunctions.cmake**.  A full example of an external
application linking with FFTX libraries is available in the
[**fftx-demo-extern-app**](https://www.github.com/spiral-software/fftx-demo-extern-app).

### FFTX Libraries Built

FFTX builds libraries for 1D and 3D FFTs for a single device.  FFTs are built
for specific sizes, thus not all possible sizes will be found in the libraries.
The sizes built for 3D FFTs are defined in the file **cube-sizes.txt** (in the
examples/library folder).  The sizes built for 1D FFTs are defined in the file
**dftbatch-sizes.txt** (in the examples/library folder).  The following
libraries are built:

|Type|Name|Description|Include Header File|
|:-----:|:-----|:-----|:-----:|
|3D FFT|fftx_mddft|Forward 3D FFT complex to complex |fftx_mddft_public.h|
|3D FFT|fftx_imddft|Inverse 3D FFT complex to complex |fftx_imddft_public.h|
|3D FFT|fftx_mdprdft|Forward 3D FFT real to complex |fftx_mdprdft_public.h|
|3D FFT|fftx_imdprdft|Inverse 3D FFT complex to real |fftx_imdprdft_public.h|
|3D Convolution|fftx_rconv|3D real convolution (in development) |fftx_rconv_public.h|
|1D FFT|fftx_dftbat|Forward batch of 1D FFT complex to complex|fftx_dftbat_public.h|
|1D FFT|fftx_idftbat|Inverse batch of 1D FFT complex to complex|fftx_idftbat_public.h|
|1D FFT|fftx_prdftbat|Forward batch of 1D FFT real to complex|fftx_prdftbat_public.h|
|1D FFT|fftx_iprdftbat|Inverse batch of 1D FFT complex to real|fftx_iprdftbat_public.h|

### Library API

Each library has serial code (CPU) and optionally GPU code (assuming either
CUDA or HIP code was built when the libraries were generated).  There are API
calls to do the following:
* Determine (get) the mode for the library (serial or GPU)
* Specify (set) whether the library should operate in serial or GPU mode
* Get the list of sizes built in the library
* Get a tuple containing pointers to the init, destroy, and run functions for a particular size
* Run a specific size transform once

The following example shows usage of the 3D FFT complex to complex transform
(others are similar, just use the appropriate names and header file(s) from the
table above).  The user (calling application) is responsible for setting up
memory buffers and allocations as required (i.e., host memory for serial code
and device memory for GPU code).

```
#include "fftx3.hpp"
#include "fftx_mddft_public.h"

    int libmode = fftx_mddft_GetLibraryMode ();   // get the library mode
    fftx_mddft_SetLibraryMode ( LIB_MODE_CUDA );  // specify CUDA mode (default)
    
    fftx::point_t<3> *wcube, curr;
    wcube = fftx_mddft_QuerySizes ();             // Get a list of sizes in library

    transformTuple_t *tupl;
    for ( int iloop = 0; ; iloop++ ) {
        if ( wcube[iloop].x[0] == 0 ) break;      // last entry in list is zero
        tupl = fftx_mddft_Tuple ( wcube[iloop] );

        ( * tupl->initfp )();                    // init function for transform

        ( * tupl->runfp )( outbuf, input, symbol );  // run the transform (may call multiple times)

        ( * tupl->destroyfp )();
    }
```

### Linking Against FFTX Libraries

FFTX provides a **cmake** include file, **FFTXCmakeFunctions.cmake**, that
provides functions to facilitate compiling and linking external applications
with the FFTX libraries.  An external application should include this file
(**$FFTX_HOME/CMakeIncludes/FFTXCmakeFunctions.cmake**) in order to access the
following helper functions to compile/link with the FFTX libraries.  Two
functions are available:

1.  **FFTX_find_libraries**() -- this function finds the FFTX libraries, linker
library path, and include file paths and exposes the following variables:
|CMake Variable Name|Description|
|:-----|:-----|
|**FFTX_LIB_INCLUDE_PATHS**|Include paths for FFTX include & library headers|
|**FFTX_LIB_NAMES**|List of FFTX libraries|
|**FFTX_LIB_LIBRARY_PATH**|Path to libraries (for linker)|
2.  **FFTX_add_includes_libs_to_target** ( target ) -- this function adds the
include file paths, the linker library path, and the library names to the
specified target.

An application typically need only call the second function and let FFTX handle
the assignment of paths, etc. to the target.  Only if an application
specifically needs to access the named variables above is it necessary to call
the first function.

### External Application Linking With FFTX

A complete example of an external application that builds test programs
utilizing the FFTX libraries is available at 
[**fftx-demo-extern-app**](https://www.github.com/spiral-software/fftx-demo-extern-app).
If you're interested in how to link an external application with FFTX please
download this example and review the **CMakeLists.txt** therein for specific
details.

When FFTX is built, the final step (of *make install*) creates a tree structure
(at the location specified by **CMAKE_INSTALL_PREFIX**).  The following
directories will be created/populated:

|Directory Name|Description|
|:-----|:-----|
|**./CMakeIncludes**|CMake include files and functions to ease integration with FFTX|
|**./bin**|Example programs built as part of the FFTX distribution|
|**./lib**|FFTX libraries|
|**./include**|Include files for using FFTX libraries|

## Examples Structure

The **examples** tree holds several examples.  Each example follows a structure
and naming conventions (described below).  The process to follow when adding a
new example is also described.

### Structure Of An Example Program

Each example is expected to reside in its own directory; the directory should be
named for the transform or problem it illustrates or tests.   At its most basic
an example consists of a driver program that defines the transform, a test
harness to exercise the transform and a **cmake** file to build the example.

### Naming Conventions For Examples

The folder containing the example should be named for the transform or problem
being illustrated, e.g., **mddft**.  This name will be used as the *project*
name in the **cmake** file (details below).

Within each folder there should be one (or possibly several) file(s) defining
transforms to be tested.  These *driver* programs are named as
*prefix*.*stem*.**cpp**, where *prefix* is the transform name, e.g., **mddft**;
*stem* is the root or stem, currently always **fftx**.

There should also be one test harness program used to exercise the transform(s)
defined.  The naming convention for the test harness is
**test**_project_.**cpp**.  There may be two flavours of the test harness: one
named with a **.cpp** suffix used to exercise a CPU version of the transform(s)
and one named with a **.cu** suffix used to exercise a GPU version of the
transform(s).

The **cmake** file is named in the usual standard way as: **CMakeLists.txt**.

### How To Add A New Example

To add a new example to **FFTX**, simply create a folder in **fftx/examples**,
called *project*.  Then in the newly created folder add your transform
definition(s), named *prefix*.**fftx.cpp**.  Add a test harness named
**test**_project_; with either, or both, suffixes: **.cpp**, or **.cu**.

Add (or copy and edit) a **cmake** file (instructions for editing below).

### Setting Up The CMakeLists.txt File

The **CMakeLists.txt** file has a section in the beginning to specifiy a few names;
most of the rules and targets are defined automatically and few, if any, changes
should be required.  The **cmake** file uses the varaible **\_codegen** to determine
whether to build for CPU or GPU (either CUDA or HIP).  This variable is defined
on the **cmake** command line (or defaults to CPU); do **not** override it in the
**cmake** file.

&nbsp;&nbsp;**1.**&nbsp;&nbsp;
Set the project name.  The preferred name is the same name as the example folder, e.g., **mddft**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
project ( mddft ${\_lang\_add} ${\_lang\_base} )

&nbsp;&nbsp;**2.**&nbsp;&nbsp;
As noted above, the file naming convention for the *driver* programs is *prefix.stem*.**cpp**.
Specify the *stem* and *prefix(es)* used; e.g., from the **mddft** example:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
set ( \_stem fftx )<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
set ( \_prefixes mddft imddft )

&nbsp;&nbsp;**3.**&nbsp;&nbsp;
Check the test harness program name: you won't need to modify this if you've
followed the recommended conventions:  The test harness program name is expected
to be **test**_project<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    set ( BUILD\_PROGRAM test${PROJECT\_NAME} )
<br>

Finally add an entry to the **CMakeLists.txt** file in the **examples** folder.  We use a **cmake**
function, **manage_add_subdir,** to control this.  Call the function with
parameters: example directory name and True/False flags for building for CPU and GPU, for
example:
```
##                  subdir name   CPU       GPU
manage_add_subdir ( hockney       TRUE      FALSE )
manage_add_subdir ( mddft         TRUE      TRUE  )
```
