#Volta config
FFTX_DIR= ../../..
GPU_DIR=/usr/local/cuda-10.1
GPUCC=/usr/local/cuda/bin/nvcc
CXX=g++
MPICXX=mpicxx
LIBRARIES=-fopenmp $(GPU_LIBS) -lmpi  


INCLUDES = -I../common -I$(GPU_DIR)/include -I$(FFTX_DIR)/include
GPU_LIBS =  -L$(GPU_DIR)/lib64 -lcufft -lcudart

gen_all:
	$(CXX)    -std=c++11 -o gen_script.x dist_3dfft.cpp $(INCLUDES) $(LIBRARIES)

	$(GPUCC)  -std=c++11 -c ../common/gpu.cu -o gpu.o -lcufft

run:
	mpirun -np 4 ./main.x

clean:
	rm -f *~ *.o *.x \#*\#
