test1d:
	nvcc -DFFTX_CUDA ../../src/library/lib_fftx_mpi/fftx_gpu.cu       -I../../src/library -I../../src/library/lib_fftx_mpi -I../3DDFT_mpi -c
	nvcc -DFFTX_CUDA ../../src/library/lib_fftx_mpi/fftx_1d_gpu.cu    -I../../src/library -I../../src/library/lib_fftx_mpi -I../3DDFT_mpi -c
	mpicc -DFFTX_CUDA ../../src/library/lib_fftx_mpi/fftx_mpi.cpp      -I../../src/library -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart \
		-c
	mpicc -DFFTX_CUDA ../../src/library/lib_fftx_mpi/fftx_1d_mpi.cpp   -I../../src/library -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart \
		-c
	mpicc -DFFTX_CUDA embed1d.cpp -I../../src/library/lib_fftx_mpi/    -I../../src/library -I../../src/include -I../3DDFFT_mpi -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart \
		-c
	mpicc fftx_gpu.o fftx_1d_gpu.o fftx_mpi.o fftx_1d_mpi.o embed1d.o \
		-L/usr/local/cuda/lib64/ -lcudart -lcufft -lstdc++ \
		-o embed1d.x


# test1d:
# 	hipcc -DFFTX_HIP ../../src/library/lib_fftx_mpi/fftx_gpu.cu -I../../src/library -I../../src/library/lib_fftx_mpi -I../3DDFT_mpi -c
# 	hipcc -DFFTX_HIP ../../src/library/lib_fftx_mpi/fftx_1d_gpu.cu -I../../src/library -I../../src/library/lib_fftx_mpi -I../3DDFT_mpi -c
# 	hipcc -DFFTX_HIP ../../src/library/lib_fftx_mpi/fftx_mpi.cpp \
# 		-I${MPICH_DIR}/include \
# 		-I../../src/library \
# 		-c
# 	hipcc -DFFTX_HIP ../../src/library/lib_fftx_mpi/fftx_1d_mpi.cpp \
# 		-I${MPICH_DIR}/include \
# 		-I../../src/library \
# 		-c
# 	hipcc -DFFTX_HIP embed1d.cpp -I../../src/library/lib_fftx_mpi/ -I../../src/library -I../../src/include -I../3DDFFT_mpi \
# 		-I${MPICH_DIR}/include \
# 		-c
# 	hipcc fftx_gpu.o fftx_1d_gpu.o fftx_mpi.o fftx_1d_mpi.o embed1d.o \
# 		-I${MPICH_DIR}/include \
# 		-L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa \
# 		-lhipfft \
# 		-o embed1d.x



test:
	nvcc -DFFTX_CUDA ../../src/library/lib_fftx_mpi/fftx_gpu.cu -I../../src/library -I../../src/library/lib_fftx_mpi -I../3DDFT_mpi -c
	mpiCC -DFFTX_CUDA ../../src/library/lib_fftx_mpi/fftx_mpi.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart -I../../src/library -c
	mpiCC -DFFTX_CUDA ../../src/library/lib_fftx_mpi/fftx_1d_mpi.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart -I../../src/library -c
	nvcc -DFFTX_CUDA test1d.cpp -I../../src/library/lib_fftx_mpi/ -I../../src/library -I../../src/include -c -I../../src/library/lib_fftx_distdft_gpu_srcs -I../3DDFFT_mpi
	mpiCC fftx_gpu.o fftx_mpi.o fftx_1d_mpi.o test1d.o -L/usr/local/cuda/lib64/ -lcudart -lcufft -o test1d.x

pencils:
	mpiCC -DFFTX_CUDA pencils.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart -I../../src/library -c
	mpiCC pencils.o -L/usr/local/cuda/lib64/ -lcudart -lcufft -o pencils.x

batch:
	nvcc -DFFTX_CUDA ../../src/library/lib_fftx_mpi/fftx_gpu.cu -I../../src/library -I../../src/library/lib_fftx_mpi -I../3DDFT_mpi -c
	mpiCC -DFFTX_CUDA ../../src/library/lib_fftx_mpi/fftx_mpi.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart -I../../src/library -c
	nvcc -DFFTX_CUDA test3DDFT_mpi_demo.cpp -I../../src/library/lib_fftx_mpi/ -I../../src/library -I../../src/include -c -I../../src/library/lib_fftx_distdft_gpu_srcs -I../3DDFT_mpi 
	mpiCC fftx_gpu.o fftx_mpi.o test3DDFT_mpi_demo.o -L/usr/local/cuda/lib64/ -lcudart -lcufft -o test3DDFT_mpi_demo.x
	mpirun --oversubscribe -np 4 ./test3DDFT_mpi_demo.x 8 4 16 1 2 0 1 0

all:
	nvcc -DFFTX_CUDA ../../src/library/lib_fftx_mpi/gpu.cu -I../../src/library -I../../src/library/lib_fftx_mpi -I../3DDFT_mpi -c
	mpiCC -DFFTX_CUDA ../../src/library/lib_fftx_mpi/fftx_mpi.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart -I../../src/library -c
	nvcc new_embedded.cu -I ../../src/library/lib_fftx_mpi/ -I../../src/library -I../../src/include -c -I ../../src/library/lib_fftx_distdft_gpu_srcs -I../3DDFT_mpi -c
	nvcc -DFFTX_CUDA test3DDFT_mpi_embedded.cpp -I ../../src/library/lib_fftx_mpi/ -I../../src/library -I../../src/include -c -I ../../src/library/lib_fftx_distdft_gpu_srcs -I../3DDFT_mpi 
	mpiCC new_embedded.o gpu.o fftx_mpi.o test3DDFT_mpi_embedded.o -L/usr/local/cuda/lib64/ -lcudart -lcufft -o test3DDFT_mpi_embedded.x



full:
	nvcc -DFFTX_CUDA ../../src/library/lib_fftx_mpi/gpu.cu -I../../src/library -I../../src/library/lib_fftx_mpi -I../3DDFT_mpi -c
	mpiCC -DFFTX_CUDA ../../src/library/lib_fftx_mpi/fftx_mpi.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart -I../../src/library -c
	nvcc -DFFTX_CUDA test3DDFT_mpi.cu -I../../src/library/lib_fftx_mpi/ -I../../src/library -I../../src/include -c -I../../src/library/lib_fftx_distdft_gpu_srcs -I../3DDFT_mpi 
	mpiCC gpu.o fftx_mpi.o test3DDFT_mpi.o -L/usr/local/cuda/lib64/ -lcudart -lcufft -o test3DDFT_mpi.x


clean:
	rm -f *.o *~ *.x

