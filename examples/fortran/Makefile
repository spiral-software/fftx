#
# FFTX
#
FFTX_INCLUDE_DIR=$(FFTX_HOME)/include
FFTX_MPI_INCLUDE_DIR=$(FFTX_HOME)/src/library/lib_fftx_mpi
FFTX_INCLUDE=-I$(FFTX_INCLUDE_DIR) -I$(FFTX_MPI_INCLUDE_DIR)

FFTX_LIB_DIR=$(FFTX_HOME)/lib
FFTX_LINK=-Wl,-rpath=$(FFTX_LIB_DIR) -L$(FFTX_LIB_DIR)
# The FFTX libraries (for either GPU or CPU) go after $(FFTX_LINK).

ifdef CUDATOOLKIT_HOME
  default: CUDA
else ifdef ROCM_PATH
  default: HIP
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
# Get rid of annoying "DSO missing from command line" error.
CUDA: LDFLAGS=-Wl,--copy-dt-needed-entries
# Need to link to ALL the libraries, even though we don't call them all.
CUDA: FFTX_LIBRARIES=-lfftx_mpi -lfftx_mddft_gpu -lfftx_imddft_gpu -lfftx_mdprdft_gpu -lfftx_imdprdft_gpu -lfftx_dftbat_gpu -lfftx_idftbat_gpu -lfftx_prdftbat_gpu -lfftx_iprdftbat_gpu -lfftx_rconv_gpu
# Targets to build.
CUDA: main

#
## make HIP: needs ROCM_PATH and CRAY_MPICH_PREFIX
#
HIP: CC=hipcc
HIP: PRESETS=-DFFTX_HIP
# To get mpi.h
HIP: CC_INCLUDE=-I$(CRAY_MPICH_PREFIX)/include
HIP: CC_LINK=-L$(ROCM_PATH)/lib -lamdhip64 -lhipfft -lrocfft -lstdc++
# Need to link to ALL the libraries, even though we don't call them all.
HIP: FFTX_LIBRARIES=-lfftx_mpi -lfftx_mddft_gpu -lfftx_imddft_gpu -lfftx_mdprdft_gpu -lfftx_imdprdft_gpu -lfftx_dftbat_gpu -lfftx_idftbat_gpu -lfftx_prdftbat_gpu -lfftx_iprdftbat_gpu -lfftx_rconv_gpu
# Targets to build.
HIP: main

#
## make CPU: default
#
CPU: CC=mpicxx
# Get rid of annoying "DSO missing from command line" error.
CPU: LDFLAGS=-Wl,--copy-dt-needed-entries
# Need to link to ALL the libraries, even though we don't call them all.
CPU: FFTX_LIBRARIES=-lfftx_mddft_cpu -lfftx_imddft_cpu -lfftx_mdprdft_cpu -lfftx_imdprdft_cpu -lfftx_dftbat_cpu -lfftx_idftbat_cpu -lfftx_prdftbat_cpu -lfftx_iprdftbat_cpu -lfftx_rconv_cpu
# Targets to build.
CPU: main

#
# Fortran
#
FC = mpif90

RM = rm -f

# Find all source files in all directories and subdorectories and resolve program dependencies.
# Actually compile.
main: test.o
		$(FC) $(PRESETS) \
			math_constants_mod.F90 \
			mpi_utils_mod.F90 \
			problem_dimensions_mod.F90 \
			fft_mod.F90 \
			data_functions_mod.F90 \
			transform_tests_mod.F90 \
			convolution_mod.F90 \
			main.F90 \
			$(LDFLAGS) test.o \
			$(CC_LINK) \
			$(FFTX_LINK) $(FFTX_LIBRARIES) \
			-o  main

  # delete modules
		@$(RM) *.mod
		@$(RM) *.smod
		@$(RM) *.pyc

test.o: test.cpp
	$(CC) $(CCFLAGS) $(PRESETS) $(CC_INCLUDE) $(FFTX_INCLUDE) test.cpp \
	-c -o test.o

## make clean: remove executables and other files
clean:
		@rm -f main-opt
		@rm -f main-dbg
		@rm -rf main-dbg.*
		@rm -rf *.mod
		@rm -rf *.smod
		@rm -rf *.pyc
		@rm -rf *.dSym
		@rm -rf *.dSYM

## make help: prints the help
.PHONY: help
help:
		@awk '/^##/ {$$1=""; print $$0}' Makefile
