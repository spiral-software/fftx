Test 1D Distributed 3DDFT
================================

Build
--------------------------------
CUDA device::

	nvcc -DFFTX_CUDA -x cu ../../src/library/lib_fftx_mpi/fftx_gpu.cpp -I../../src/library -I../../src/library/lib_fftx_mpi -I../3DDFT_mpi -c
	nvcc -DFFTX_CUDA -x cu ../../src/library/lib_fftx_mpi/fftx_1d_gpu.cpp -I../../src/library -I../../src/library/lib_fftx_mpi -I../3DDFT_mpi -c
	mpicc -DFFTX_CUDA ../../src/library/lib_fftx_mpi/fftx_mpi.cpp -I../../src/library -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart -c
	mpicc -DFFTX_CUDA ../../src/library/lib_fftx_mpi/fftx_1d_mpi.cpp -I../../src/library -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart -c
	mpicc -DFFTX_CUDA test3DDFT_mpi_1D.cpp -I../../src/library/lib_fftx_mpi/ -I../../src/library -I../../src/include -I../3DDFFT_mpi -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart -c
	mpicc fftx_gpu.o fftx_1d_gpu.o fftx_mpi.o fftx_1d_mpi.o test3DDFT_mpi_1D.o -L/usr/local/cuda/lib64/ -lcudart -lcufft -lstdc++ -o test3DDFT_mpi_1D.x

HIP device::

	hipcc -DFFTX_HIP ../../src/library/lib_fftx_mpi/fftx_gpu.cu -I../../src/library -I../../src/library/lib_fftx_mpi -I../3DDFT_mpi -c
	hipcc -DFFTX_HIP ../../src/library/lib_fftx_mpi/fftx_1d_gpu.cu -I../../src/library -I../../src/library/lib_fftx_mpi -I../3DDFT_mpi -c
	hipcc -DFFTX_HIP ../../src/library/lib_fftx_mpi/fftx_mpi.cpp -I${MPICH_DIR}/include -I../../src/library -c
	hipcc -DFFTX_HIP ../../src/library/lib_fftx_mpi/fftx_1d_mpi.cpp -I${MPICH_DIR}/include -I../../src/library -c
	hipcc -DFFTX_HIP test3DDFT_mpi_1D.cpp -I../../src/library/lib_fftx_mpi/ -I../../src/library -I../../src/include -I../3DDFFT_mpi -I${MPICH_DIR}/include -c
	hipcc fftx_gpu.o fftx_1d_gpu.o fftx_mpi.o fftx_1d_mpi.o test3DDFT_mpi_1D.o -I${MPICH_DIR}/include -L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa -lhipfft -o test3DDFT_mpi_1D.x


Run
--------------------------------

To run::

    mpirun -n <ranks> ./test3DDFT_mpi_1D.x <M> <N> <K> <batch> <embedded> <forward> <complex> <check>

where:
    ``ranks`` - number of MPI ranks to distributed the 3D DFT.

    ``M``, ``N``, ``K`` - describe the size of the dimensions of the 3D DFT for X, Y, and Z, respectively.

    ``batch`` - describes the number of 3D DFTs that are computed at a time. This has been tested with 1.

    ``embedded`` - determines whether the input tensor is embedded in the center of a tensor twice the size in each dimension [2K, 2N, 2N]. This has been tested with 0.

    ``forward`` - 1 for a forward transform and 0 for an inverse transform.

    ``complex`` - 1 for a complex-to-complex transform, or 0 if the input or output is real (e.g. R2C or C2R). This has been tested with 1.

    ``check`` - 1 to check the distributed computation with an equivalent 3D transform using vendor libraries only on rank 0. This should be 0 for problem sizes that would be too large to fit in device memory.

For a forward transform, the input data is laid out as ``[(Z0), Z1, Y, X]``, where ``Z0`` and ``Z1`` are derived from tiling the ``Z`` dimension and distributing a block to each of the ``p`` MPI ranks.
Therefore, ``Z0`` is distributed over the ranks, ``Z1`` is the slowest local dimension and ``X`` is the fastest dimension.
The local tensor is of size ``[K/p, N, M]``.
The output data is laid out as ``[(X0), Y, X1, Z]``, where ``X0`` and ``X1`` are the slower and faster dimensions, respectively, of the ``X`` dimension that is tiled by ``M/p``.

For an inverse transform, the input data is laid out as ``[(X0), Y, X1, Z]`` and the output data is laid out as ``[(Z0), Z1, Y, X]``.