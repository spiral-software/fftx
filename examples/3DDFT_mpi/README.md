Test 1D Distributed 3D DFT (test3DDFT_mpi_1D.x)
============================================
This example routines
allows the running of various different distributed 3DDFTs over
MPI, including the Plane Wave (PW) version for the embedded sphere to cube
distributed FFT used in Materials and Chemistry codes based on a plane
wave basis for the electron wave functions. The MPI ranks are assumed to be organized in a linear array, and
the 3D DFT is partitioned along the Z dimension. The local computation
is performed on the GPU assigned to the local rank..

To run with MPI::

    mpirun -n <ranks> ./test3DDFT_mpi_1D.x <M> <N> <K> <batch> <embedded> <forward> <complex> <check>

To run with Slurm::

     srun <Slurm/Network options> -n <ranks> ./test3DDFT_mpi_1D.x <M> <N> <K> <batch> <embedded> <forward> <complex> <check>
     
where:
    ``ranks`` - number of MPI ranks to distributed the 3D DFT.

    ``M``, ``N``, ``K`` - describe the input size of the dimensions of the 3D DFT for X, Y, and Z, respectively.

    ``batch`` - describes the number of 3D DFTs that are computed at a time. 

    ``embedded`` - determines whether the input tensor is embedded in the center of a tensor twice the size in each dimension [2K, 2N, 2M]. To perform the PW FFT the half sized Fourier space sphere/ellipsoid should be embedded in an array of size [K,N,M] padded with zeros outside the sphere and passed into this routine. The output in real space will be the double sized grid. 

    ``forward`` - 1 for a forward transform and 0 for an inverse transform.

    ``complex`` - 1 for a complex-to-complex transform, or 0 if the input or output is real (e.g. R2C or C2R). 

    ``check`` - 1 to check the distributed computation with an equivalent 3D transform using vendor libraries only on rank 0. This should be 0 for problem sizes that would be too large to fit in device memory.

For a forward transform, the input data is laid out as ``[(Z0), Z1, Y, X]``, where ``Z0`` and ``Z1`` are derived from tiling the ``Z`` dimension and distributing a block to each of the ``p`` MPI ranks.
Therefore, ``Z0`` is distributed over the ranks, ``Z1`` is the slowest local dimension and ``X`` is the fastest dimension.
The local tensor is of size ``[K/p, N, M]``.
The output data is laid out as ``[(X0), Y, X1, Z]``, where ``X0`` and ``X1`` are the slower and faster dimensions, respectively, of the ``X`` dimension that is tiled by ``M/p``.

For an inverse transform, the input data is laid out as ``[(X0), Y, X1, Z]`` and the output data is laid out as ``[(Z0), Z1, Y, X]``.

* **Example configurations**

The following examples configurations have been verified on Frontier on 4 ranks with 1 GPU per rank.

| M   | N | K  | Batch | Embedded | Forward | Complex | Check |
|-----|---|----|-------|----------|---------|---------|-------|
| 32  |32 | 32 |  1    |    1     |    0    |    1    |    1  |
| 32  |32 | 32 |  1    |    1     |    1    |    1    |    1  |
| 32  |32 | 32 |  1    |    1     |    0    |    0    |    1  |
| 32  |32 | 32 |  1    |    1     |    1    |    0    |    1  |
| 40  |40 | 40 |  1    |    1     |    0    |    1    |    1  |
| 40  |40 | 40 |  1    |    1     |    1    |    1    |    1  |
| 40  |40 | 40 |  1    |    1     |    0    |    0    |    1  |
| 40  |40 | 40 |  1    |    1     |    1    |    0    |    1  |
| 48  |48 | 48 |  1    |    1     |    0    |    1    |    1  |
| 48  |48 | 48 |  1    |    1     |    1    |    1    |    1  |
| 48  |48 | 48 |  1    |    1     |    0    |    0    |    1  |
| 48  |48 | 48 |  1    |    1     |    1    |    0    |    1  |
| 64  |64 | 64 |  1    |    1     |    0    |    1    |    1  |
| 64  |64 | 64 |  1    |    1     |    1    |    1    |    1  |
| 64  |64 | 64 |  1    |    1     |    0    |    0    |    1  |
| 64  |64 | 64 |  1    |    1     |    1    |    0    |    1  |
| 80  |80 | 80 |  1    |    1     |    0    |    1    |    1  |
| 80  |80 | 80 |  1    |    1     |    1    |    1    |    1  |
| 80  |80 | 80 |  1    |    1     |    0    |    0    |    1  |
| 80  |80 | 80 |  1    |    1     |    1    |    0    |    1  |
| 96  |96 | 96 |  1    |    1     |    0    |    1    |    1  |
| 96  |96 | 96 |  1    |    1     |    1    |    1    |    1  |
| 96  |96 | 96 |  1    |    1     |    0    |    0    |    1  |
| 96  |96 | 96 |  1    |    1     |    1    |    0    |    1  |
| 128  |128 | 128 |  1    |    1     |    0    |    1    |    1  |
| 128  |128 | 128 |  1    |    1     |    1    |    1    |    1  |
| 128  |128 | 128 |  1    |    1     |    0    |    0    |    1  |
| 128  |128 | 128 |  1    |    1     |    1    |    0    |    1  |
| 160  |160 | 160 |  1    |    1     |    0    |    1    |    1  |
| 160  |160 | 160 |  1    |    1     |    1    |    1    |    1  |
| 160  |160 | 160 |  1    |    1     |    0    |    0    |    1  |
| 160  |160 | 160 |  1    |    1     |    1    |    0    |    1  |
| 192  |192 | 192 |  1    |    1     |    0    |    1    |    1  |
| 192  |192 | 192 |  1    |    1     |    1    |    1    |    1  |
| 192  |192 | 192 |  1    |    1     |    0    |    0    |    1  |
| 192  |192 | 192 |  1    |    1     |    1    |    0    |    1  |

* **Limitations**

Limited testing on Frontier for batch > 1 has been
verified to be correct. Due to cuFFT's unique alignment requirements,
batch > 1 configurations is currently not supported for NVIDIA devices.
