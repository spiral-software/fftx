FFTX examples
=============

This **examples** tree holds several examples.
Each example follows a structure and naming conventions, as described below.
The process to follow when adding a new example is also described.

### Structure of an example program

Each example is expected to reside in its own directory
under **examples**; the directory
should be named for the transform or problem it illustrates or tests.
At its most basic, an example consists of a driver program that
defines the transform, a test harness to exercise the transform,
and a **cmake** file to build the example.

### Naming conventions for examples

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

The **cmake** file is named in the usual standard way as: `CMakeLists.txt`.

### How to add a new example

To add a new example to **FFTX**:
1. Create a folder in **fftx/examples**, called *project*.
2. In the newly created *project* folder, add your transform
definition(s), named *prefix*.**fftx.cpp**.
3. In the *project* folder, add a test harness named **test**_project_,
with either, or both, suffixes: **.cpp**, or **.cu**.
4. In the *project* folder, add (or copy and edit)
a **CMakeLists.txt** file (instructions for editing below).

### Setting up the cmake file

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

**2.** As noted above, the file naming convention for the
*driver* programs is *prefix.stem*.**cpp**.
Specify the *stem* and *prefix(es)* used; e.g., from the **mddft** example:
```
set ( _stem fftx )
set ( _prefixes mddft imddft )
```

**3.** Check the test harness program name.
You won't need to modify this if you've followed the recommended conventions.
The test harness program name is expected to be **test**_*project*:
```
    set ( BUILD_PROGRAM test${PROJECT_NAME} )
```

Finally, add an entry to the **CMakeLists.txt** file in
the **examples** folder.
We use a **cmake** function, **manage_add_subdir,** to control this.  If the
example you add depends on libraries built in **FFTX**, then the call to
**manage_add_subdir** should be conditional based on a CMake dependent option
derived from the options (see the file **options.cmake** in the **FFTX** root
directory) indicating if the required libraries were built.  As a rule, one
should not attempt to build an example whose dependent libraries have not been
built.  The **manage_add_subdir** function should be called with parameters:
example directory name and TRUE/FALSE flags for building for CPU and GPU, as in:
```
if ( BLD_COMPARE_EX )
    manage_add_subdir ( compare       FALSE     TRUE )
endif ()

manage_add_subdir ( mddft         TRUE      TRUE )
manage_add_subdir ( hockney       TRUE      FALSE )
```
**BLD_COMPARE_EX** is a CMake dependent option derived from the build status for
the **mddft** and **mdprdft** libraries; the compare examples depend on thoise
libraries and we don't build it unless those libraries are also built.

## Running FFTX example programs

If you follow the **FFTX** build instructions, then executables for
the examples will be placed in the **$FFTX_HOME/bin** directory.
So to run the programs (with default input parameters), you can simply do:
```
cd $FFTX_HOME/bin
./testcompare_device
./testverify
```
etc. Since we set RPATH to point to where the libraries are installed,
you likely will not need to adjust the library path variable,
typically **LD_LIBRARY_PATH**.

### Existing examples in this repo

* **mddft**
```
./testmddft [iterations]
```
Runs the forward and inverse complex-to-complex 3D FFTs
the number of times specified by `iterations` (default 20)
on the fixed size `[fftx_nx, fftx_ny, fftx_nz]` where
`fftx_nx`, `fftx_ny`, and `fftx_nz` are defined in the file `test_plan.h`,
and displays the
amount of time taken for each iteration by the CPU and by the GPU
(if running on GPU).

* **compare_cufft**
```
./testcompare_cufft [verbosity] [iterations]
```
Runs forward and inverse complex-to-complex 3D FFTs,
both **FFTX**-generated and either cuFFT (for CUDA) or rocFFT (for HIP),
for `iterations` iterations (default 20)
on the fixed size `[fftx_nx, fftx_ny, fftx_nz]` where
`fftx_nx`, `fftx_ny`, and `fftx_nz` are defined in the file `test_comp.h`.   
Writes whether or not the results of **FFTX** and cuFFT/rocFFT match,
and the average time on CPU and GPU for **FFTX** and cuFFT/rocFFT,
both including and not including the first iteration.  
The `[verbosity]` setting defaults to 0.  
If `verbosity` is at least 1, then also writes out minimum and maximum times.  
If `verbosity` is at least 2, then also writes out the time for every
iteration.  
If `verbosity` is at least 3, then also writes out every point
where **FFTX** and cuFFT/rocFFT fail to match.

* **compare**   
All of these tests run four different 3D FFTs:
forward and inverse
complex to complex, real to complex, and complex to real.
They may run **FFTX**-generated transforms,
the device library transforms (cuFFT on CUDA, rocFFT on HIP),
or both.   
```
./testcompare_device [verbosity] [iterations]
```
Runs the four transforms,
both **FFTX**-generated and cuFFT/rocFFT,
the number of times specified by `iterations` (default 20),
for all 3D sizes in the **FFTX** library.   
Writes whether or not the results of **FFTX** and cuFFT/rocFFT match,
and the average time on CPU and GPU for **FFTX** and cuFFT/rocFFT,
both including and not including the first iteration.  
The `[verbosity]` setting defaults to 0.  
If `verbosity` is at least 1, then also writes out minimum and maximum times.  
If `verbosity` is at least 2, then also writes out the time for every
iteration.  
If `verbosity` is at least 3, then also writes out every point
where **FFTX** and cuFFT/rocFFT fail to match.
```
./testconstant [nx] [ny] [nz] [which] [verbosity]
```
Runs the four transforms
with a constant-valued array input on size `[nx, ny, nz]`,
and writes out the maximum error in the results.  
If `which` is set to 0, then only cuFFT or rocFFT is run.  
If `which` is set to 1, then only the **FFTX** library routine is run.  
If `which` is set to 2, then both **FFTX** and cuFFT/rocFFT are run.  
The `[verbosity]` setting defaults to 0.  
If `verbosity` is at least 2, then diagnostic messages are written out
to indicate progress through the different stages, as can be useful
for debugging purposes.  
If `verbosity` is at least 3, then all nonzero entries in the output
are written out.
```
./testimpulse [nx] [ny] [nz] [which] [verbosity]
```
Runs forward and inverse complex-to-complex, real-to-complex, and
complex-to-real 3D FFTs,
with a unit-impulse array input on size `[nx, ny, nz]`,
and writes out the maximum error in the results.  
If `which` is set to 0, then only cuFFT or rocFFT is run.  
If `which` is set to 1, then only the **FFTX** library routine is run.  
If `which` is set to 2, then both **FFTX** and cuFFT/rocFFT are run.  
The `[verbosity]` setting defaults to 0.  
If `verbosity` is at least 2, then diagnostic messages are written out
to indicate progress through the different stages, as can be useful
for debugging purposes.  
If `verbosity` is at least 3, then all entries in the output
that are not equal to 1 are written out.

* **hockney**
```
./testhockney [verbosity]
```
Runs **FFTX** functions in Hockney's algorithm on a 3D domain of length `n`
on input of length `ns` and output of length `nd`, where
`n`, `ns`, and `nd` are defined in the file `hockney.h`.
The formula is
```
output[id, jd, kd] =
sum_{is, js, ks} ( input([is, js, ks]) * G([id-is, jd-js, kd-ks]) )
```
where `n`-1-`nd` &le; `id`, `jd`, `kd` &le; `n`-1, and the sum is
over the range 0 &le; `is`, `js`, `ks` &le; `ns`-1.  
The `[verbosity]` setting defaults to 0.  
If `verbosity` is at least 1, then also checks that the output matches
that from a direct computation, which can be slow.  
If `verbosity` is at least 2, then also writes out the output arrays
for both the **FFTX** transformation and the direct computation.

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
