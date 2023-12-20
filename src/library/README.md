Managing FFTX Libraries
=======================

This README documents the scripts and tools used to create the **FFTX** libraries and
basic tools to validate the output of the generated code against NumPy.

The overall process for generating libraries and building **FFTX** is discussed in the main
README found in the project root directory, [**here**](https://www.github.com/spiral-software/fftx).

The library building process is controlled by the shell script **build-lib-code.sh**,
which accepts one argument -- the architecture for which to build; if omitted the default
is for CPU.  The other choices available are: CUDA, HIP, and SYCL.

The libraries are built using python scripts to iterate over a set of sizes for each
library (the set may be empty, in which case just the library API is created). It is
required to create all the libraries (otherwise applications won't build due to missing
header files).

Libraries are built for the following transforms:

|Type|Name|Description|
|:-----:|:-----|:-----|
|3D FFT|fftx_mddft|Forward 3D FFT complex to complex|
|3D FFT|fftx_imddft|Inverse 3D FFT complex to complex|
|3D FFT|fftx_mdprdft|Forward 3D FFT real to complex|
|3D FFT|fftx_imdprdft|Inverse 3D FFT complex to real|
|3D Convolution|fftx_rconv|3D real convolution|
|1D FFT|fftx_dftbat|Forward batch of 1D FFT complex to complex|
|1D FFT|fftx_idftbat|Inverse batch of 1D FFT complex to complex|
|1D FFT|fftx_prdftbat|Forward batch of 1D FFT real to complex (in development)
|1D FFT|fftx_iprdftbat|Inverse batch of 1D FFT complex to real (in development)

The 3D FFT libraries are built using **gen_files.py**, and the 1D batch libraries are
built using **gen_dftbat.py**, (the *gen* scripts).  Both of these Python
scripts accept similar arguments that direct what to build.  The parameters for
these scripts are:

```
usage: gen_files.py -t TRANSFORM -s SIZES_FILE -p {CPU,CUDA,HIP,SYCL} [-i] [-m SIZES_MASTER]

Build FFTX library code with Spiral and transform specifications

optional arguments:
  -h, --help            show this help message and exit
  -i, --inverse         False [default], run forward transform; when specified run Inverse transform
  -m SIZES_MASTER, --sizes_master SIZES_MASTER
                        Master sizes filename; Regenerate headers & API files [uses existing code files] for the library

required arguments:
  -t TRANSFORM, --transform TRANSFORM
                        transform to use use for building the library
  -s SIZES_FILE, --sizes_file SIZES_FILE
                        filename containing the sizes to build
  -p [{CPU,CUDA,HIP,SYCL}], --platform [{CPU,CUDA,HIP,SYCL}]
                        Platform: one of {CUDA | HIP | SYCL | CPU}
```
The parameters are essentially the same for **gen_dftbat.py**
```
usage: gen_dftbat.py -t TRANSFORM -s SIZES_FILE -p {CPU,CUDA,HIP,SYCL} [-i] [-m SIZES_MASTER]

Build FFTX library code with Spiral and transform specifications

optional arguments:
  -h, --help            show this help message and exit
  -i, --inverse         False [default], run forward transform; when specified run Inverse transform
  -m SIZES_MASTER, --sizes_master SIZES_MASTER
                        Master sizes filename; Regenerate headers & API files [uses existing code files] for the library

required arguments:
  -t TRANSFORM, --transform TRANSFORM
                        transform to use use for building the library
  -s SIZES_FILE, --sizes_file SIZES_FILE
                        filename containing the sizes to build
  -p [{CPU,CUDA,HIP,SYCL}], --platform [{CPU,CUDA,HIP,SYCL}]
                        Platform: one of {CUDA | HIP | SYCL | CPU}
```

### Creating Empty Libraries

It is not required to have any defined sizes in a library; however, the basic
library must be created so that the API exists for applications to call/link
against.  An empty library is created by using a sizes file with no entries
(e.g., **empty-sizes.txt**) and running the *gen* script as follows:
```
python gen_files.py -t fftx_mddft -s empty-sizes.txt -p HIP
```

### Adding Size(s) To A Library

The shell script **build-lib-code.sh** calls the python *gen* scripts to build the
libraries.  The transform definitions are contained in a set of files prefixed
by: **fftx_**. Normally, an entire library's code is generated in a single run
of one of the *gen* scripts -- this can be quite time-consuming if a large
number of sizes are to be built.  It is possible to add additional sizes to a
library without having to regenerate all the source code.  The way to achieve
this is to add the new size(s) specs to the master sizes file *and additionally*
put them is a separate temporary file.  Then the *gen* script can be run with
the option to regenerate the libarary API files; e.g.,
```
python gen_files.py -t fftx_mddft -s tempf -p HIP -m cube-sizes-gpu.txt
```
In this example, the temporary file **tempf** should hold the new size(s) to add
to the MDDFT library -- they must also be added to **cube-sizes-gpu.txt**, we're
building for the HIP platform.  Once Spiral has completed building the new sizes
the API files are regenerated with the necessary tables for what is defined in
the library.

If one needed to regenerate the API file for a library (without adding new
sizes), specify **empty-sizes.txt** (in place of **tempf** in the above example)
and new API files will be generated based on the sizes in the master sizes file
(**cube-sizes-gpu.txt**), assuming all the source code files have been
previously generated.

Whenever a *gen* script runs, any size that cannot be created (e.g., Spiral
fails due to bad input) is noted by placing an entry into the file
**build-lib-code-failures.txt**.

Once file generation is complete the newly generated code must be recompiled (go
to the project's build directory and run cmake/make as required).  This is true
whether sizes are added or even just when the API files are regenerated.

### Validating The Generated Code

The are four python validation scripts that are designed to validate the
generated Spiral code against NumPy to verify correctness.  The four scripts
are:

|Name|Library Tested|Description|
|:-----:|:-----|:-----|
|val_mddft.py|MDDFT & IMDDFT|Test the forward and inverse 3D complex to complex transforms|
|val_mdprdft.py|MDPRDFT & IMDPRDFT|Test the forward (R2C) and inverse (C2R) 3D real transforms|
|val_dftbat.py|DFTBAT & IDFTBAT|Test the forward and inverse batch 1D complex to complex transforms|
|val_prdftbat.py|PRDFTBAT & IPRDFTBAT|Test the forward and inverse batch 1D real transforms|

All the validation scripts are used in the same manner and all accept the same
arguments (they differ in libraries tested based on the name of the script).
The usage and arguments are as follows:
```
python val_mddft.py -h
usage: val_mddft.py [-h] [-e] [-s SIZE | -f FILE] libdir

Validate FFTX built libraries against NumPy computed versions of the transforms

positional arguments:
  libdir                directory containing the library

optional arguments:
  -h, --help            show this help message and exit
  -e, --emit            emit warnings when True, default is False
  -s SIZE, --size SIZE  3D size specification of the transform (e.g., 80x80x120)
  -f FILE, --file FILE  file containing sizes to loop over
```
If neither file nor size is specified then the script attempts to validate all
sizes listed in the transform's default master sizes file.

NOTE: The default files associated to each transform type are specified in the
**config-fftx-libs.sh** script (found in the FFTX_HOME directory).