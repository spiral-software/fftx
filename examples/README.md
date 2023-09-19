FFTX examples
=============

This **examples** tree holds several examples.
Each example follows a structure and naming conventions, as described below.
The process to follow when adding a new example is also described.

### Structure of an example program

Each example is expected to reside in its own directory
under **examples**; the directory
should be named for the transform or problem it illustrates or tests.
At its most basic, an example consists of a test harness to exercise the transform,
and a **cmake** file to build the example.

Most examples are structured this way.  A header file (typically named
*transform*Obj.hpp provides the **Spiral** specification such that **Spiral**
can generate the code for the transform.  The header files for defined
transforms are maintained inthe directory: **$FFTX_HOME/src/include**

### Naming conventions for examples

The folder containing the example should be named for the transform or problem
being illustrated, e.g., **mddft**.  This name will be used as the *project*
name in the **cmake** file (details below).

Within each folder there may be one (or more) sample / test programs
allowing different transforms to be tested.

There will be at least one test harness program used to exercise the
transform(s) defined.  The naming convention for the test harness is
**test**_project_.**cpp**.  The suffix for programs is **.cpp**, in general the
codes were developed using macros (see **device_macros.h**) that define the
appropiate code or API depending on whether one is building for CPU, CUDA, or
HIP.  The CMakeLists.txt in each folder set appropriate properties on source
files for the intended compiler (e.g., hipcc for HIP or nvcc for CUDA).

The **cmake** file is named in the usual standard way as: `CMakeLists.txt`.

### How to add a new example

To add a new example to **FFTX**:
1. Create a folder in **fftx/examples**, called *project*.
2. Add your transform definition(s) in header files to **src/include**.
3. In the *project* folder, add a test harness named **test**_project_**.cpp**.
4. In the *project* folder, add (or copy and edit)
a **CMakeLists.txt** file (instructions for editing below).

### Setting up the Cmake file

The **cmake** file, **CMakeLists.txt**, has a section in the beginning to
specify a few names;
most of the rules and targets are defined automatically,
and few, if any, changes should be required.

The **cmake** file uses the variable **\_codegen**
setting of either CPU or CUDA or HIP
to determine whether to build, respectively, for CPU, or CUDA on GPU,
or HIP on GPU.
The variable **\_codegen** is defined
on the **cmake** command line (or defaults to CPU);
do *not* override it in the **cmake** file.

**1.** Set the project name.  The preferred name is
the same name as the example folder, e.g., **mddft**
```
project ( mddft ${_lang_add} ${_lang_base} )
```

**2.** All examples should simply define *stem* and *prefix* as follows:
```
set ( _stem fftx )
set ( _prefixes )
```

**3.** Check the test harness program name.
You won't need to modify this if you've followed the recommended conventions.
The test harness program name is expected to be **test**_*project*:
```
set ( BUILD_PROGRAM test${PROJECT_NAME} )
```

Finally, add an entry to the **CMakeLists.txt** file in
the **examples** folder.
We use a **cmake** function, **manage_add_subdir,** to control this.
The **manage_add_subdir** function should be called with parameters:
example directory name and TRUE/FALSE flags for building for
CPU and GPU, as in:
```
manage_add_subdir ( mddft         TRUE      TRUE )
manage_add_subdir ( mdprdft       TRUE      TRUE )
```

## Running FFTX example programs

If you follow the **FFTX** build instructions, then executables for
the examples will be placed in the **$FFTX_HOME/bin** directory.
So to run the programs (with default input parameters), you can simply do:
```
cd $FFTX_HOME/bin
./testrconv_lib
./testverify
```
etc. Since we set RPATH to point to where the libraries are installed,
you likely will not need to adjust the library path variable,
typically **LD_LIBRARY_PATH**.

NOTE: On Windows **RPATH** does not work correctly.  If you encounter "missing
library" errors when trying to run an example, add the directory containing the
installed libraries to your **PATH**.

### Existing examples in this repo

* **mddft**
```
./testmddft: [ -i iterations ] [ -s MMxNNxKK ] [ -h (print help message) ]
```
Runs the forward and inverse complex-to-complex 3D FFTs
the number of times specified by `iterations` (default 2)
on the fixed size `[24, 32, 40]`,
and displays the
amount of time taken for each iteration by the CPU and by the GPU
(if running on GPU).
When a size is specified (e.g., `-s 64x64x64`), it will run the given size.  In
all cases if a predefined transform of the appropraite size is found in a
library it will be used; otherwise, RTC generates the required size.

* **mdprdft**
```
./testmdprdft: [ -i iterations ] [ -s MMxNNxKK ] [ -h (print help message) ]
```
Runs the forward (real-to-complex) and inverse (complex-to-real) 3D FFTs
the number of times specified by `iterations` (default 2)
on the fixed size `[24, 32, 40]`,
and displays the
amount of time taken for each iteration by the CPU and by the GPU
(if running on GPU).
When a size is specified (e.g., `-s 64x64x64`), it will run the given size.  In
all cases if a predefined transform of the appropraite size is found in a
library it will be used; otherwise, RTC generates the required size.

* **rconv**   
These examples run tests of **FFTX** real 3D convolution transforms:
tests with random input and a constant-valued symbol,
a test on a delta function,
and a test of a Poisson equation solver.   
On the tests with random input and a constant-valued symbol,
the number of times is specified by `[iterations]` (default 20).  
There is also a `[verbosity]` setting that defaults to 0.   
If `verbosity` is at least 1, then also writes out
the maximum error for each of the three test categories.  
If `verbosity` is at least 3, then also writes out
the maximum error for every iteration with random input.
```
./testrconv [verbosity] [iterations]
```
Runs tests of **FFTX** real 3D convolution transform
on the fixed size `[fftx_nx, fftx_ny, fftx_nz]` where
`fftx_nx`, `fftx_ny`, and `fftx_nz` are defined in the file `rconv.h`.
```
./testrconv_lib [verbosity] [iterations]
```
Runs tests of **FFTX** real 3D convolution transforms
for all 3D sizes in the **FFTX** library.

* **verify**  
These examples all run a series of verification tests on
forward and inverse complex-to-complex, real-to-complex, and
complex-to-real 3D FFTs,
where on tests using random data,
the number of times is specified by `[iterations]` (default 20).  
There is also a `[verbosity]` setting that defaults to 0.   
If `verbosity` is at least 1, then also writes out
the maximum error for each category of linearity, impulses and shifts.   
If `verbosity` is at least 2, then also writes out
the maximum error for each type of test within each category.  
If `verbosity` is at least 3, then also writes out
the maximum error for every iteration of every test.
```
./testverify [verbosity] [iterations]
```
Runs tests of **FFTX** library transforms
on the fixed size `[fftx_nx, fftx_ny, fftx_nz]` where
`fftx_nx`, `fftx_ny`, and `fftx_nz` are defined in the file `verify.h`.
```
./testverify_device [nx] [ny] [nz] [verbosity] [iterations]
```
Runs tests of either cuFFT transforms (on CUDA)
or rocFFT transforms (on HIP)
on the fixed size `[nx, ny, nz]`.
```
./testverify_lib [verbosity] [iterations]
```
Runs tests of **FFTX**
transforms for all 3D sizes in the **FFTX** library.

* **warpx**   
```
./warpx
```
Runs the **FFTX** transform for
the [**Pseudo-Spectral Analytical Time Domain solver**](https://warpx.readthedocs.io/en/latest/theory/picsar_theory.html#pseudo-spectral-analytical-time-domain-psatd)
used in WarpX.
This program only takes a zero-valued input and writes the output
to a file named `fftxout`.
