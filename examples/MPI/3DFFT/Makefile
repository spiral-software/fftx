FFTX_DIR=
GPU_DIR=
GPUCC=nvcc
CXX=g++
MPICXX=mpicxx
LIBRARIES=-fopenmp $(GPU_LIBS) -lmpi
SPIRAL_DIR = 


INCLUDES = -I../common -I$(GPU_DIR)/include -I$(FFTX_DIR)/include
GPU_LIBS =  -L$(GPU_DIR)/lib64 -lcufft -lcudart

FILEROOT = mddft3d_32x32x32_mpi

gen_all:
	$(CXX)    -std=c++11 -o gen_script.x dist_3dfft.cpp $(INCLUDES) $(LIBRARIES)
	./gen_script.x > dist_3dfft.g
	$(SPIRAL_DIR)/spiral < dist_3dfft.g
	$(GPUCC) -std=c++11 -c ../common/gpu.cu -o gpu.o -lcufft
	$(CXX)   -std=c++11 -c ../common/mddft3d.cpp -o mddft3d.o $(INCLUDES) 
	$(GPUCC) -std=c++11 -c -D__FILEROOT__=$(FILEROOT) $(FILEROOT).cu -o $(FILEROOT).o $(INCLUDES)
	$(MPICXX) -std=c++11 -D__FILEROOT__=$(FILEROOT)  -o main.x main_example.cpp $(FILEROOT).o mddft3d.o gpu.o $(INCLUDES) $(LIBRARIES)

run:
	mpirun -np 4 ./main.x

clean:
	rm -f *~ *.o *.x \#*\# *.g
