FFTX Project
============

This is the public repository for the FFTX API source, examples, and documentation.

## Building FFTX

To use and build FFTX you must install the following pre-requisites:<br><br>
    1.  **spiral-software**, available [here.](https://www.github.com/spiral-software/spiral-software)<br>
    2.  **spiral-package-fftx**, available [here.](https://www.github.com/spiral-software/spiral-package-fftx)<br>
    3.  **spiral-package-simt**, available [here.](https://www.github.com/spiral-software/spiral-package-simt)<br>

### C Compiler and Build Tools

SPIRAL builds on Linux/Unix with **gcc** and **make**, on Windows it builds with **Visual Studio**.

For macOS SPIRAL requires version 10.14 (Mojave) or later of macOS, with a compatible version of **Xcode** and
and **Xcode Command Line Tools**. 

### Installing Pre-requisites

Clone **spiral-software** to a location on you computer.  E.g., do:
```
cd ~/work
git clone https://www.github.com/spiral-software/spiral-software
```
This location is known as *SPIRAL HOME* and you must set an environment variable
**SPIRAL_HOME** to point to this location later.

To install the two spiral packages do the following:
```
cd ~/work/spiral-software/namespaces/packages
git clone https://www.github.com/spiral-software/spiral-package-fftx fftx
git clone https://www.github.com/spiral-software/spiral-package-simt simt
```
**NOTE:** The spiral packages must be installed under directory
**$SPIRAL_HOME/namespaces/packages** and must be placed in folders with the
prefix *spiral-package* removed. 

Follow the build instructions for **spiral-software** (see the **README**
[**here**](https://github.com/spiral-software/spiral-software/blob/master/README.md) ).

### Installing FFTX

Clone **FFTX** to a location on your computer.  E.g., do:
```
cd ~/work
git clone https://www.github.com/spiral-software/fftx
```
**NOTE:** Before attempting to build ensure you have set environment variable
**SPIRAL_HOME** to point to your **spiral-software** instance.

You can build the examples for CPU or GPU (for those examples that support GPU).
By default they are built for CPU (this is your only option if your machine
doesn't have a GPU or the NVIDIA nvcc compiler).  To build the software do the
following:
```
mkdir build
cd build
cmake ..				# build for CPU, *OR*
cmake -D_codegen=CPU ..			# build for CPU, *OR*
cmake -D_codegen=GPU ..			# build for GPU
```
If you are building on Linux or Linux like systems then do:
```
make install
```
or, if you are building on Windows (for Release configuration), do:
```
cmake --build . --config Release --target install
```

Currently, **FFTX** builds a number of example programs; the programs will be
installed in the *build/bin* folder and can be run from there; e.g.,
```
cd build/bin
./testhockney
./testrconv
./testmddft
```

## Examples Structure

The **examples** tree holds several examples.  Each example follows a structure
and naming conventions (described below).  The process to follow when adding a
new example is also described.

### Structure Of An Example Program

Each example is expected to reside in its own directory; the directory should be
named for the transform or problem it illustrates or tests.   At its most basic
an example consists of a driver program that defines the transform, a test
harness to exercise the transform and a *cmake* file to build the example.

### Naming Conventions For Examples

The folder containing the example should be named for the transform or problem
being illustrated, e.g., **rconv**.  This name will be used as the *project*
name in the *cmake* file (details below).

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

The *cmake* file is named in the usual standard way as: **CMakeLists.txt**.

### How To Add A New Example

To add a new example to **FFTX**, simply create a folder in **fftx/examples**,
called *project*.  Then in the newly created folder add your transform
definition(s), named *prefix*.**fftx.cpp**.  Add a test harness named
**test**_project_; with either, or both, suffixes: **.cpp**, or **.cu**.

Add (or copy and edit) a *cmake* file (instructions for editing below).

### Setting Up The CMakeLists.txt File

The CMakeLists.txt file has a section in the beginning to specifiy a few names;
most of the rules and targets are defined automatically and few, if any, changes
should be required.  The cmake file uses the varaible **\_codegen** to determine
whether to build for CPU or GPU.  This variable is defined on the cmake command
line (or defaults to CPU); do **not** override it in the cmake file.

```
##  _codegen specifies CPU or GPU (serial or CUDA) code generation.  Will be set
##  in parent so don't change here.
```

&nbsp;&nbsp;**1.**&nbsp;&nbsp;
Set the project name.  The preferred name is the same name as the example folder, e.g., **verify**<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
project ( verify ${\_lang\_add} )

&nbsp;&nbsp;**2.**&nbsp;&nbsp;
As noted above, the file namimg convention for the *driver* programs is *prefix.stem*.**cpp**.
Specify the *stem* and *prefix(es)* used; e.g., from the **verify** example:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
set ( \_stem fftx )<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
set ( \_prefixes mddft2 mddft3 imddft2 imddft3 )

&nbsp;&nbsp;**3.**&nbsp;&nbsp;
Check the test harness program name: you won't need to modify this if you've
followed the recommended conventions:  The test harness program name is expected
to be **test**_project_.{**cpp|cu**}<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    set ( BUILD\_PROGRAM test${PROJECT\_NAME} )
<br>

Finally add an entry to the cmake file in the **examples** folder.  We use a cmake
function **manage_add_subdir** to control this.  Call the function with
parameters directory name and True/False flags for building for CPU and GPU, for
example:
```
##                  subdir name   CPU       GPU
manage_add_subdir ( hockney       TRUE      FALSE )
manage_add_subdir ( test_plan_dft TRUE      TRUE  )
```