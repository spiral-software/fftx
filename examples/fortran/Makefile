#
# FFTX
#
FFTX_INCLUDE_DIR=$(FFTX_HOME)/include
FFTX_MPI_INCLUDE_DIR=$(FFTX_HOME)/src/library/lib_fftx_mpi
FFTX_INCLUDE=-I$(FFTX_INCLUDE_DIR) -I$(FFTX_MPI_INCLUDE_DIR)

FFTX_LIB_DIR=$(FFTX_HOME)/lib
# The FFTX libraries (for either GPU or CPU) go after $(FFTX_LINK).
# Apple needs a space after rpath, while other platforms need an equal sign.
ifeq ($(VENDOR),apple)
  FFTX_LINK=-Wl,-rpath $(FFTX_LIB_DIR) -L$(FFTX_LIB_DIR)
else
  ifndef ROCM_PATH
    # This flag gets rid of annoying "DSO missing from command line" error.
    LDFLAGS=-Wl,--copy-dt-needed-entries
  endif
  FFTX_LINK=-Wl,-rpath=$(FFTX_LIB_DIR) -L$(FFTX_LIB_DIR)
endif

# Need to link to ALL CPU/GPU libraries, even though we don't call them all.
FFTX_CPU_LIBRARIES=-lfftx_mddft_cpu -lfftx_imddft_cpu -lfftx_mdprdft_cpu -lfftx_imdprdft_cpu -lfftx_dftbat_cpu -lfftx_idftbat_cpu -lfftx_prdftbat_cpu -lfftx_iprdftbat_cpu -lfftx_rconv_cpu
FFTX_MPI_LIBRARY=-lfftx_mpi
FFTX_GPU_LIBRARIES=-lfftx_mddft_gpu -lfftx_imddft_gpu -lfftx_mdprdft_gpu -lfftx_imdprdft_gpu -lfftx_dftbat_gpu -lfftx_idftbat_gpu -lfftx_prdftbat_gpu -lfftx_iprdftbat_gpu -lfftx_rconv_gpu

ifdef CUDATOOLKIT_HOME
  default: CUDA
else ifdef ROCM_PATH
  default: HIP
else ifdef ONEAPI_DEVICE_SELECTOR
  default: SYCL
else
  default: CPU
endif

#
## make CUDA: needs CUDATOOLKIT_HOME
#
CUDA: CC=nvcc
CUDA: CCFLAGS=-x cu
CUDA: PRESETS=-DFFTX_CUDA
# To get helper_cuda.h
CUDA: CC_INCLUDE=-I$(CUDATOOLKIT_HOME)/../../examples/OpenMP/SDK/include
CUDA: CC_LINK=-L$(CUDATOOLKIT_HOME)/lib64 -lcudart
CUDA: FFTX_LIBRARIES=$(FFTX_MPI_LIBRARY) $(FFTX_GPU_LIBRARIES)
# Targets to build.
CUDA: fortran_main

#
## make HIP: needs ROCM_PATH and CRAY_MPICH_PREFIX
#
HIP: CC=hipcc
HIP: PRESETS=-DFFTX_HIP
# To get mpi.h
HIP: CC_INCLUDE=-I$(CRAY_MPICH_PREFIX)/include
HIP: CC_LINK=-L$(ROCM_PATH)/lib -lamdhip64 -lhipfft -lrocfft -lstdc++
HIP: FFTX_LIBRARIES=$(FFTX_MPI_LIBRARY) $(FFTX_GPU_LIBRARIES)
# Targets to build.
HIP: fortran_main

#
## make SYCL: needs AURORA_PE_ONEAPI_ROOT
#
SYCL: CC=icpx
SYCL: CCFLAGS=-std=c++17 -fsycl
SYCL: PRESETS=-DFFTX_SYCL
SYCL: CC_LINK=-L$(AURORA_PE_ONEAPI_ROOT) -lOpenCL -lmkl_core -lmkl_cdft_core -lmkl_sequential -lmkl_rt -lmkl_intel_lp64 -lmkl_sycl
SYCL: FFTX_LIBRARIES=$(FFTX_GPU_LIBRARIES)
SYCL: fortran_main

#
## make CPU: default
#
CPU: CC=mpicxx
CPU: CCFLAGS=-std=c++11
CPU: FFTX_LIBRARIES=$(FFTX_CPU_LIBRARIES)
# Targets to build.
CPU: fortran_main

#
# Fortran
#
FC = mpif90

RM = rm -f

# Find all source files in all directories and subdorectories and resolve program dependencies.
# Actually compile.
fortran_main: test.o
		$(FC) $(PRESETS) \
			math_constants_mod.F90 \
			mpi_utils_mod.F90 \
			problem_dimensions_mod.F90 \
			fft_mod.F90 \
			data_functions_mod.F90 \
			transform_tests_mod.F90 \
			convolution_mod.F90 \
			fortran_main.F90 \
			$(LDFLAGS) test.o \
			$(CC_LINK) \
			$(FFTX_LINK) $(FFTX_LIBRARIES) \
			-o  fortran_main

  # delete modules
		@$(RM) *.mod
		@$(RM) *.smod
		@$(RM) *.pyc

test.o: test.cpp
	$(CC) $(CCFLAGS) $(PRESETS) $(CC_INCLUDE) $(FFTX_INCLUDE) test.cpp \
	-c -o test.o

## make clean: remove executables and other files
clean:
		@rm -f fortran_main
		@rm -f test.o
		@rm -rf *.mod
		@rm -rf *.smod
		@rm -rf *.pyc
		@rm -rf *.dSym
		@rm -rf *.dSYM

## make help: prints the help
.PHONY: help
help:
		@awk '/^##/ {$$1=""; print $$0}' Makefile
