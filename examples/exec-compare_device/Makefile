CC=nvcc

FFTX_LIB_DIR = ../library

SUFFIX = cu

INCLUDE_FLAGS = -I$(FFTX_HOME)/include \
	-I$(FFTX_LIB_DIR)/lib_fftx_mddft_srcs \
	-I$(FFTX_LIB_DIR)/lib_fftx_imddft_srcs \
	-I$(FFTX_LIB_DIR)/lib_fftx_mdprdft_srcs \
	-I$(FFTX_LIB_DIR)/lib_fftx_imdprdft_srcs \
	-I$(FFTX_LIB_DIR) \
	-I$(CUDA_HOME)/samples/common/inc

LIB_FLAGS = -lcufft -L$(FFTX_LIB_DIR)/build/bin -lfftx_mddft_precomp -lfftx_imddft_precomp -lfftx_mdprdft_precomp -lfftx_imdprdft_precomp

COMPILE_FLAGS = -O3 -DNDEBUG -std=c++11

CUDA_FLAGS = --generate-code=arch=compute_70,code=[compute_70,sm_70] -m64 -rdc=false -Xptxas -v -maxrregcount=64 -x cu

testrconv: testcompare_device.$(SUFFIX)
	$(CC) $(INCLUDE_FLAGS) $(COMPILE_FLAGS) $(CUDA_FLAGS) $(LIB_FLAGS) \
	-o testcompare_device testcompare_device.$(SUFFIX)
