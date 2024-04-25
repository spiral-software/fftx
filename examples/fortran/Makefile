# to add a
# Compiler/Linker settings

#
# FFTX
#
FFTX_INCLUDE=$(FFTX_HOME)/include/
FFTX_MPI_INCLUDE=$(FFTX_HOME)/src/library/lib_fftx_mpi
FFTX_EXAMPLE_INCLUDE=$(FFTX_HOME)/examples/3DDFT_mpi
FFTX_LIB_DIR=$(FFTX_HOME)/lib
FFTX_LINK=-Wl,-rpath=$(FFTX_LIB_DIR) -L$(FFTX_LIB_DIR)
# libraries after $(FFTX_LINK)

#
# either CUDA or HIP
#
ifndef_any_of = $(filter undefined,$(foreach v,$(1),$(origin $(v))))
ifdef_any_of = $(filter-out undefined,$(foreach v,$(1),$(origin $(v))))
ifneq ($(call ifdef_any_of,CUDATOOLKIT_HOME ROCM_PATH), )
  # Need to link to ALL the libraries, even though we don't call them all.
  FFTX_LIBRARIES=-lfftx_mpi -lfftx_mddft_gpu -lfftx_imddft_gpu -lfftx_mdprdft_gpu -lfftx_imdprdft_gpu -lfftx_dftbat_gpu -lfftx_idftbat_gpu -lfftx_prdftbat_gpu -lfftx_iprdftbat_gpu
#
# CPU only
#
else
  # Need to link to ALL the libraries, even though we don't call them all.
  FFTX_LIBRARIES=-lfftx_mddft_cpu -lfftx_imddft_cpu -lfftx_mdprdft_cpu -lfftx_imdprdft_cpu -lfftx_dftbat_cpu -lfftx_idftbat_cpu -lfftx_prdftbat_cpu -lfftx_iprdftbat_cpu -lfftx_rconv_cpu  
endif

#
# CUDA only
#
ifdef CUDATOOLKIT_HOME
  # FFTX_GPU_INCLUDE=/usr/local/cuda/include
  ### To get helper_cuda.h
  CUDA_INCLUDE=$(CUDATOOLKIT_HOME)/../../examples/OpenMP/SDK/include
  # CUDA_LIB=/usr/local/cuda/lib64/
  CUDA_LIB=$(CUDATOOLKIT_HOME)/lib64
  ### To get libcudart.so.11.0 on perlmutter.
  # CUDA_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/lib64
  CUDA_LINK=-DFFTX_CUDA -L$(CUDA_LIB) -lcudart
endif

#
# HIP only
#
ifdef ROCM_PATH
  ### To get mpi.h
  HIP_INCLUDE=$(CRAY_MPICH_PREFIX)/include
  # FFTX_GPU_INCLUDE=/opt/rocm-5.3.0/include
  HIP_LIB=$(ROCM_PATH)/lib
  # HIP_LIB=/opt/rocm-5.3.0/lib/
  HIP_LINK=-DFFTX_HIP -L$(HIP_LIB) -lamdhip64 -lhipfft -lstdc++
endif

#
# Fortran
#
FC = mpif90
FSTD = -std=f2018
FFLAGS = -m64 -xf95-cpp-input -fall-intrinsics -fno-align-commons -fimplicit-none  -ffree-line-length-none -fconvert=big-endian -fbacktrace

BASE_PROGRAM_NAME = fftx_convolution

RM = rm -f

## OPTIONS:
## OPT: the level of optimization: 1, 2, or 3, default 2. Only useful for METHOD=opt
ifndef OPT
  OPT=2
endif

## METHOD: the type of compilation. Valid options are: opt and dbg. opt is default
ifndef METHOD
  METHOD=opt
endif

# define debug flags
ifeq ($(METHOD),dbg)
  DEBUGFLAGS = -D__DEBUG__ -gdwarf-2 -fbounds-check -pedantic-errors -ggdb -O -Wall  -g -fbacktrace -ffpe-trap=invalid,zero,overflow -fcheck=all
  OPT=0
endif

# Get rid of annoying "DSO missing from command line" error.
LDFLAGS=-Wl,--copy-dt-needed-entries

##
## EXAMPLE: make METHOD=dbg


##
## TARGETS
## make: build with the provided options, see above
 # Find all source files in all directories and subdorectories and resolve program dependencies
 # actually compile
default: test.o
		$(FC) \
			math_constants_mod.F90 \
			mpi_utils_mod.F90 \
			problem_dimensions_mod.F90 \
			fft_mod.F90 \
			data_functions_mod.F90 \
			transform_tests_mod.F90 \
			convolution_mod.F90 \
			main.F90 \
			$(LDFLAGS) test.o \
			$(CUDA_LINK) \
			$(HIP_LINK) \
			$(FFTX_LINK) $(FFTX_LIBRARIES) \
			-o  main-$(METHOD)

  # delete modules
		@$(RM) *.mod
		@$(RM) *.smod
		@$(RM) *.pyc

test.o: test.cpp
ifdef CUDATOOLKIT_HOME
	nvcc -x cu -DFFTX_CUDA -I$(CUDA_INCLUDE) test.cpp \
	-I$(FFTX_MPI_INCLUDE) -I$(FFTX_INCLUDE) -I$(FFTX_EXAMPLE_INCLUDE) \
	-c -o test.o
else ifdef ROCM_PATH
	hipcc      -DFFTX_HIP  -I$(HIP_INCLUDE)  test.cpp \
	-I$(FFTX_MPI_INCLUDE) -I$(FFTX_INCLUDE) -I$(FFTX_EXAMPLE_INCLUDE) \
	-c -o test.o
else
	mpicc                                    test.cpp \
	-I$(FFTX_MPI_INCLUDE) -I$(FFTX_INCLUDE) -I$(FFTX_EXAMPLE_INCLUDE) \
	-c -o test.o
endif

# -lcudart -lcufft

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
