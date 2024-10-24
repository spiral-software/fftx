FFTX Project
============

This is the public repository for the **FFTX** API source, examples, and
documentation.

License
~~~~~~~

This project is licensed under the BSD License (available
`here <https://www.github.com/spiral-software/fftx/blob/main/License.txt>`__)

Copyright (c) 2018-2023, Carnegie Mellon University. All rights
reserved.

Building FFTX
-------------

Prerequisites
~~~~~~~~~~~~~

There are several pre-requisites that must be installed prior to
building and using **FFTX**. You will need:

-  **CMake**, minimum version 3.14; however, a more-modern version is
   preferred as some deprecated features must be used with older
   versions.
-  if on Linux/Unix, **gcc** and **make**.
-  if on Windows, **Visual Studio**, and an ability to run **bash**
   shell scripts. You can use the Git Bash shell available with **git**,
   but other shells such as Cygwin or msys will also work.
-  if on macOS, version 10.14 (Mojave) or later of macOS, with a
   compatible version of **Xcode** and **Xcode Command Line Tools**.
-  **python**, version 3.6 or higher. On some systems, both **python**
   (usually version 2.7) and **python3** exist. The scripts used to
   create the **FFTX** library source code check the version of
   **python**, and if it is version 2.X it will try to run **python3**
   instead. A user, therefore, should not have to worry whether
   **python** or **python3** comes first in the user's path.

Summary
~~~~~~~

A brief summary of the steps necessary to install FFTX follows; these
steps outline the most straight forward case (building for CPU only),
for more complex installs and to utilize GPUs please follow the more
detailed instructions below.

::

   1.  Make a directory to hold the download code and cd to it (e.g., mkdir ~/work ; cd ~/work)
   2.  git clone https://github.com/spiral-software/fftx
   3.  cd fftx
   4.  export FFTX_HOME=`pwd`
   5.  unset SPIRAL_HOME
   6.  source ./get_spiral.sh
   7.  ./config-fftx-libs.sh 
   8.  mkdir build
   9.  cd build/
   10. cmake -DCMAKE_INSTALL_PREFIX=$FFTX_HOME -D_codegen=CPU ..
   11. make install -j

To build and use **FFTX**, including how to utilize GPUs, please follow
these steps:

1. `Install SPIRAL and associated
   packages. <#1-install-spiral-and-associated-packages>`__
2. `Clone the FFTX repository. <#2-clone-the-fftx-repository>`__
3. `Generate library source code. <#3-generate-library-source-code>`__
4. `Compile library source code and
   examples. <#4-compile-library-source-code-and-examples>`__

.. _1-install-spiral-and-associated-packages:

1. Install SPIRAL and associated packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have SPIRAL installed (and have the **SPIRAL_HOME**
environment variable set) FFTX will use that installation and you can
skip to `step 2, "Clone the FFTX
repository" <#2-clone-the-fftx-repository>`__.

If you want to manually install SPIRAL follow the steps below;
alternatively, skip to `step 2, "Clone the FFTX
repository" <#2-clone-the-fftx-repository>`__ and have FFTX pull down
the necessary SPIRAL repositories and perform the build steps.

Clone **spiral-software** (available
`here <https://www.github.com/spiral-software/spiral-software>`__) to a
location on your computer. E.g.,:

::

   cd ~/work
   git clone https://www.github.com/spiral-software/spiral-software

This location is known as *SPIRAL HOME*, and you must set an environment
variable **SPIRAL_HOME** (here, **``~/work/spiral-software``**) to point
to this location later.

**FFTX** requires the **SPIRAL** packages
`fftx <https://www.github.com/spiral-software/spiral-package-fftx>`__,
`simt <https://www.github.com/spiral-software/spiral-package-simt>`__,
`mpi <https://www.github.com/spiral-software/spiral-package-mpi>`__, and
`jit <https://www.github.com/spiral-software/spiral-package-jit>`__.

You need to download these separately, as follows:

::

   cd $SPIRAL_HOME/namespaces/packages
   git clone https://www.github.com/spiral-software/spiral-package-fftx fftx
   git clone https://www.github.com/spiral-software/spiral-package-simt simt
   git clone https://www.github.com/spiral-software/spiral-package-mpi mpi
   git clone https://www.github.com/spiral-software/spiral-package-jit jit

**NOTES:**

-  The **SPIRAL** packages must be installed under directory
   **$SPIRAL_HOME/namespaces/packages** and must be placed in folders
   with the prefix *spiral-package-* removed.
-  If you already have **spiral-software** installed, please refresh the
   installation to ensure you're up-to-date, especially for the
   **SPIRAL** packages.
-  It is preferable to download the **SPIRAL** packages *before*
   performing the **SPIRAL** build steps.

Follow the build instructions for **spiral-software** (see the
**README**
`here <https://github.com/spiral-software/spiral-software/blob/master/README.md>`__
).

.. _2-clone-the-fftx-repository:

2. Clone the FFTX repository.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone **FFTX** to a location on your computer. E.g.,

::

   cd ~/work
   git clone https://www.github.com/spiral-software/fftx
   cd fftx

Set the environment variable **FFTX_HOME** to point to the directory
where you want to install **FFTX** (which is not necessarily the same
directory where you have cloned **FFTX**; you may want to have separate
installation directories for different backends).

If you have not already installed SPIRAL and the SPIRAL packages
required for FFTX, you can have them downloaded and built from the
repositories by sourcing the **get_spiral.sh** shell script now. This
script checks the definition of the **SPIRAL_HOME** environment
variable, and if undefined it will get the SPIRAL code, build it and
export a definition for **SPIRAL_HOME**. Make sure you source (vs. run)
the script:

::

   . ./get_spiral.sh

or

::

   source get_spiral.sh

.. _3-generate-library-source-code:

3. Generate library source code.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**FFTX** builds libraries of transforms for a set of different sizes.
The library source code is generated from **SPIRAL** script
specifications, and must be created before building **FFTX** itself. For
the sizes that are pre-built the code will be added to libraries (the
pre-defined fixed-size libraries). Alternatively, sizes not defined can
have the code generated and compiled at run-time (RTC).

Before creating the library source code consider if you will be running
on CPU only, or also utilizing a GPU. If you create all the source code
(and related **cmake** scripts and library APIs) for GPU and then try
building for CPU only you may encounter compiler errors or unexpected
results. It is recommended that you have separate installs for CPU and
GPU.

The shell script **config-fftx-libs.sh** is a utility script in the
**FFTX** home directory that marshals resources for building the
libraries and examples. There is a flag that enables or disables the
building of examples (enabled by default). NOTE: All libraries **should
be created**; creating the library is a pre-requisite to creating its
API files (files which are required later when building the RTC code).
Building the examples may be turned off if you only need to build the
libraries for an external application). The script is self-documenting;
typically it should not be edited (unless you need to disable building
the examples).

NOTE: It is required that you build all libraries (even if you add few
or no fixed size entries to the library). This is because the API header
files are created automatically at build time. At run time the API can
check if a specific transform of the requested size exists in the
library and use it if it exists. If it doesn't exist then the desired
transform can be generated and compiled on-the-fly (RTC).

Run the script as follows:

::

   ./config-fftx-libs.sh <platform>
   ##  where <platform> is one of { CPU [default] | CUDA | HIP | SYCL }

If no argument is provided, then the platform defaults to CPU. This
script runs the **build-lib-code.sh** script in the **src/library**
directory and will marshal the resources and options needed for the set
of libraries. This step can take quite some time depending on the number
of transforms and set of sizes to create. The code is targeted to run on
a CPU or a GPU (either CUDA or HIP or SYCL) depending on the platform
specified with the script. By default, only a small number of fixed
sizes will be created for the fixed-size libraries.

The text file **``build-lib-code-failures.txt``** will contain a list of
any library transforms that failed to generate in this step.

Running **config-fftx-libs.sh** also creates a file called
**options.cmake**. The options defined in this file are used by
**CMake** to determine what is actually compiled at build time.

.. _4-compile-library-source-code-and-examples:

4. Compile library source code and examples.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From your **FFTX** home directory, set up a **build** folder (which can
be given any name, and you may want to have separate ones for different
backends). When you configure using **CMake** you must specify the
install prefix that **CMake** should use (the system default location
for **CMake** may be a directory for which you do not have write
privileges). Do that by setting the environment variable **FFTX_HOME**
and specifying either the directory path or the environment variable on
the **CMake** command line.

**NOTES:**

-  You will need **FFTX_HOME** set in order to use or reference **FFTX**
   artifacts externally.
-  **FFTX_HOME** may be, but does not have to be, the same as your
   **FFTX** home directory.
-  Some tips for building on a supercomputing platform at NERSC or OLCF
   are available
   `here. <https//github.com/spiral-software/spiral-software/blob/master/supercomputer-README.md>`__

::

   export FFTX_HOME=~/work/fftx
   mkdir build
   cd build
   cmake -DCMAKE_INSTALL_PREFIX=$FFTX_HOME -D_codegen=CPU ..      # build for CPU, *or*
   cmake -DCMAKE_INSTALL_PREFIX=$FFTX_HOME -D_codegen=CUDA ..     # build for CUDA, *or*
   cmake -DCMAKE_INSTALL_PREFIX=$FFTX_HOME -DCMAKE_CXX_COMPILER=hipcc -D_codegen=HIP ..      # build for HIP
   cmake -DCMAKE_INSTALL_PREFIX=$FFTX_HOME -DCMAKE_CXX_COMPILER=icpx -D_codegen=SYCL ..      # build for SYCL

   make install

When **FFTX** is built, the final step (of *make install*) creates a
tree structure (at the location specified by **CMAKE_INSTALL_PREFIX**).
The following directories will be created/populated:

+-----------------------+---------------------------------------------+
| Directory Name        | Description                                 |
+=======================+=============================================+
| **./CMakeIncludes**   | CMake include files and functions to ease   |
|                       | integration with **FFTX**                   |
+-----------------------+---------------------------------------------+
| **./bin**             | Executables for example programs built as   |
|                       | part of the **FFTX** distribution           |
+-----------------------+---------------------------------------------+
| **./lib**             | **FFTX** libraries, that can be called by   |
|                       | external applications                       |
+-----------------------+---------------------------------------------+
| **./include**         | Include files for using **FFTX** libraries  |
+-----------------------+---------------------------------------------+
| **./cache_jit_files** | Folder containing the RTC code generated    |
|                       | for any transform not found in a fixed-size |
|                       | library                                     |
+-----------------------+---------------------------------------------+

Building on Windows
~~~~~~~~~~~~~~~~~~~

**FFTX** can be built on Windows; however, you need to be able to run a
[bash] shell script as mentioned above to build the library source code.
To build **FFTX**, open a shell and do the following:

::

   cd fftx
   ./config-fftx-libs.sh <platform> where <platform> is one of { CPU [default] | CUDA | HIP }
   export FFTX_HOME=~/work/fftx
   mkdir build
   cd build
   cmake -DCMAKE_INSTALL_PREFIX=$FFTX_HOME -D_codegen=CUDA ..
   cmake --build . --target install --config Release

This shows an example building for CUDA on Windows. You can also build
for CPU or AMD HIP or Intel SYCL as shown above (under Building for
Linux).

Running FFTX Example Programs
-----------------------------

After building **FFTX**, to run the programs that are in the
**examples** subtree, simply do:

::

   cd $FFTX_HOME/bin
   ./testrconv
   ./testverify

etc. Since we set **RPATH** to point to where the libraries are
installed, you likely will not need to adjust the library path variable,
typically **LD_LIBRARY_PATH**.

NOTE: On Windows **RPATH** does not work correctly. If you encounter
"missing library" errors when trying to run add the directory containing
the installed libraries to your **PATH**.

The README in the **examples** folder
`here <https://github.com/spiral-software/fftx/blob/master/examples/README.md>`__
contains a list of examples, how to run them, and how to add a new
example. More details about individual examples may be found in the
individual example folder's README file.

Libraries
---------

The libraries built and copied to the **$FFTX_HOME/lib** folder can be
used by external applications to leverage **FFTX** transforms. To access
the necessary include files and libraries, an external application's
**CMakeLists.txt** should include
**CMakeInclude/FFTXCmakeFunctions.cmake**. A full example of an external
application linking with **FFTX** libraries is available in the
`fftx-demo-extern-app <https://www.github.com/spiral-software/fftx-demo-extern-app>`__.

FFTX Libraries Built
~~~~~~~~~~~~~~~~~~~~

**FFTX** builds libraries for 1D and 3D FFTs for a single device. FFTs
are built for a set of specific sizes, thus not all possible sizes will
be found in the libraries. There are default files specifying the sizes
to build for each of: 1D FFTs and 3D FFTs (default file names below).
You can customize the set of sizes to build by either editing these
files **or** provide your own files -- just override the default file
name in the **config-fftx-libs.sh** script. If you provide your own
file(s) just follow the format shown in the default file(s). The default
files specifying the sizes to build for each group of transforms are
(all files are in the src/library folder):

====== ================== ================
Type   File Name          Description
====== ================== ================
1D FFT dftbatch-sizes.txt Batch of 1D FFTs
3D FFT cube-sizes-cpu.txt 3D FFTs for CPU
3D FFT cube-sizes-gpu.txt 3D FFTs for GPU
====== ================== ================

The following is a list of the libraries built:

+----------------+----------------+----------------------------------+
| Type           | Name           | Description                      |
+================+================+==================================+
| 3D FFT         | fftx_mddft     | Forward 3D FFT complex to        |
|                |                | complex                          |
+----------------+----------------+----------------------------------+
| 3D FFT         | fftx_imddft    | Inverse 3D FFT complex to        |
|                |                | complex                          |
+----------------+----------------+----------------------------------+
| 3D FFT         | fftx_mdprdft   | Forward 3D FFT real to complex   |
+----------------+----------------+----------------------------------+
| 3D FFT         | fftx_imdprdft  | Inverse 3D FFT complex to real   |
+----------------+----------------+----------------------------------+
| 3D Convolution | fftx_rconv     | 3D real convolution              |
+----------------+----------------+----------------------------------+
| 1D FFT         | fftx_dftbat    | Forward batch of 1D FFT complex  |
|                |                | to complex                       |
+----------------+----------------+----------------------------------+
| 1D FFT         | fftx_idftbat   | Inverse batch of 1D FFT complex  |
|                |                | to complex                       |
+----------------+----------------+----------------------------------+
| 1D FFT         | fftx_prdftbat  | Forward batch of 1D FFT real to  |
|                |                | complex (in development)         |
+----------------+----------------+----------------------------------+
| 1D FFT         | fftx_iprdftbat | Inverse batch of 1D FFT complex  |
|                |                | to real (in development)         |
+----------------+----------------+----------------------------------+

Library API
~~~~~~~~~~~

The following example shows usage of the 3D FFT complex-to-complex
transform (others are similar; just use the appropriate names and header
file(s) from the table above). The user (calling application) is
responsible for setting up memory buffers and allocations as required
(i.e., host memory for serial code, device memory for GPU code). The
following abstract assumes device memory buffers are already
allocated/initialized as needed; error checking is omitted for clarity
and brevity:

::

   #include "fftx3.hpp"
   #include "transformlib.hpp"

       std::vector<int> sizes{ mm, nn, kk };         // 'cube' dimensions
       std::vector<void *> args{ dY, dX, dsym };     // pointers to the Output, Input, and symbol arrays

       MDDFTProblem mdp ( args, sizes, "mddft" );    // Define transform

       mdp.transform();                              // Run the transform

If the size specified with the transform definition is found in a
library then that code is executed; however, if it is not found in a
library then RTC is invoked to generate and compile the necessary code
(this is also cached for future use).

Linking Against FFTX Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**FFTX** provides a **cmake** include file,
**FFTXCmakeFunctions.cmake**, that provides functions to facilitate
compiling and linking external applications with the **FFTX** libraries.
An external application should include this file
(**$FFTX_HOME/CMakeIncludes/FFTXCmakeFunctions.cmake**) in order to
access the following helper functions to compile/link with the **FFTX**
libraries. Two functions are available:

1. **FFTX_add_includes_libs_to_target** ( target ) -- This function adds
   the include file paths, the linker library path, and the library
   names to the specified target.
2. **FFTX_find_libraries**\ () -- This function finds the **FFTX**
   libraries, linker library path, and include file paths and exposes
   the following variables:

+----------------------------+----------------------------------------+
| CMake Variable Name        | Description                            |
+============================+========================================+
| **FFTX_LIB_INCLUDE_PATHS** | Include paths for **FFTX** include &   |
|                            | library headers                        |
+----------------------------+----------------------------------------+
| **FFTX_LIB_NAMES**         | List of **FFTX** libraries             |
+----------------------------+----------------------------------------+
| **FFTX_LIB_LIBRARY_PATH**  | Path to libraries (for linker)         |
+----------------------------+----------------------------------------+

An application will typically need only call
**FFTX_add_includes_libs_to_target**\ (), and let **FFTX** handle the
assignment of paths, etc. to the target. Only if an application
specifically needs to access the named variables above is it necessary
to call **FFTX_find_libraries**\ ().

External Application Linking With FFTX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A complete example of an external application that builds test programs
utilizing the **FFTX** libraries is available at
`fftx-demo-extern-app <https://www.github.com/spiral-software/fftx-demo-extern-app>`__.
If you're interested in how to link an external application with
**FFTX** please download this example and review the
**``CMakeLists.txt``** therein for specific details. As an example, in
brief, the required steps to add the Poisson test program (using
**cmake**) are:

::

   set ( POISSON1_TEST poissonTest )

   ##  FFTX_HOME must be defined in the environment or on the command line
   if ( DEFINED ENV{FFTX_HOME} )
       message ( STATUS "FFTX_HOME = $ENV{FFTX_HOME}" )
       set ( FFTX_SOURCE_DIR $ENV{FFTX_HOME} )
   else ()
       if ( "x${FFTX_HOME}" STREQUAL "x" )
           message ( FATAL_ERROR "FFTX_HOME environment variable undefined and not specified on command line" )
       endif ()
       set ( FFTX_SOURCE_DIR ${FFTX_HOME} )
   endif ()

   ##  Include FFTX CMake functions
   include ( "${FFTX_SOURCE_DIR}/CMakeIncludes/FFTXCmakeFunctions.cmake" )

   add_executable          ( ${POISSON1_TEST} ${POISSON1_TEST}.cpp )
   target_link_libraries   ( ${POISSON1_TEST} PRIVATE dl )

   FFTX_add_includes_libs_to_target ( ${POISSON1_TEST} )

The Poisson test program is a CPU only sample program using the FFTX
library and RTC interfaces. When the program is run it accepts a single
argument, **nx**, that specifies the dimension of a cube to test. If no
argument is provided, a default value of **128** is used. The program
may be run for different size cubes as follows:

::

   poissonTest             ## defaults to 128^8; runs codegen
   poissonTest -nx 80      ## Size 80^3, is present in the FFTX libraries
   poissonTest -nx 64      ## Size 64^3, is present in the FFTX libraries
   poissonTest -nx 224     ## Size 224^3; runs codegen

For sizes in the FFTX libraries the program calls into the library and
runs immediately. For sizes not in the libraries, **Spiral** is run to
generate the required source code, which is then compiled into a
temporary library and executed. The source code is cached (meaning that
if the specific size is run again, **Spiral** is not required as the
source code is reused).
