FFTX Distributed 3D FFT Library
===============================

FFTX provides a distributed interface for describing and computing 3D FFTs in a distributed setting using MPI.  This guide documents instructions for setting up, and computing with the distributed interface. Furthermore, it also documents the assumptions behind the different versions of distributed FFTs.

Scope of FFTX Distributed Library
---------------------------------

The scope of the distributed library is described by 7 different factors, namely:

1. Types of FFTs.  (Real vs Complex)
2. Directions.     (Forward vs Backwards)
3. Batch size.     (Unit vs Non-unit)
4. Processor Grid. (1D vs 2D decomposition)
5. Embedded.       (Full FFT vs Embeded in zero-padded cube)
6. MPI Type.       (Device-aware vs Host-based)
7. Packing.        (Packing performed on host vs device)

Currently, only double-precision (complex or real) data is supported. Additional details are describe in the appropriate sections.

API
---

The design of the FFTX distributed API is similar to many other libraries, i.e. there is a plan, execute and destroy routines. The APIs are as follows:

.. code-block:: none

    //1D processor distribution. The number of processors is given by p, and comm is an MPI communicator, typically MPI_COMM_WORLD.
    fftx_plan  plan = fftx_plan_distributed(comm, p, X, Y, Z, batch, embedded, complex);

    //2D processor distribution. The p processors are organized into a r x c grid, and comm is an MPI communicator, typically MPI_COMM_WORLD.
    fftx_plan  plan = fftx_plan_distributed(comm, r, c, X, Y, Z, batch, embedded, complex);

    //device_out_buffers, and device_in_buffers are assumed to be GPU pointers.
    fftx_execute(plan, device_out_buffer, device_in_buffer, direction);
 
    //release all resources allocated by the planning functions
    fftx_plan_destroy(plan);

Only one of the planning function is required. The choice of the planning function depends on how the distributed data is mapped onto the processor grid.


Running the first example
---------------------------------

After compiling FFTX, a distributed example, test3DDFT_mpi_1D.cpp, will be compiled (if enabled). The file will execute a distributed 3DFFT using a 1D processor grid. The compiled file takes 7 command line arguments as follows: ``<M> <N> <K> <batch> <embedded> <forward> <complex>``

``M``, ``N``, ``K`` represents the 3 dimensions of the 3D FFT. 

The ``batch`` parameter indicates how many 3D-FFTs are performed simultaneously. A minimum value of ``1`` needs to be passed as the ``batch`` parameter. 

The ``embedded`` parameter takes values either ``1`` or ``0``. A value of ``0`` means that a full 3DFFT is implemented. A value of ``1`` indicates that the data cube is embedded in a larger data cube that that is twice the size in all dimension. The padded values are default to 0. 

The ``forward`` parameter indicates if a foward (``1``) or inverse (``0``) FFT is performed. 

The ``complex`` parameter, when set to ``1`` indicates that the FFT is a complex 3DFFT. However, when the parameter is set to ``0``, the 3DFFT is either a real-to-complex FFT (if the ``forward`` parameter is set to ``1``), or a complex-to-real FFT (otherwise). 

Additional details for running the example can be found in the `example directory <../examples/3DDFT_mpi/README.md>`_.

Types of FFTs
-------------
**Options:** Real,  Complex

Currently, the FFTX distributed library supports only real and complex FFTs. Complex FFT means that both inputs and outputs are complex numbers. Real FFTs mean that either the inputs or the outputs are real numbers. The following data layout are assumed for the different types of FFTs.

1. Complex data are assumed to be in interleaved data format. This means that the real and imaginary components of the complex number are stored in consecutive memory location.
2. Real data are assumed to be in natural order.

It should be noted that further factors may impose additional constraints on the input/output data. These additional constraints are described in their respective sub-sections.

Directions
----------
**Options:** FFTX_FORWARD, FFTX_BACKWARD

The Forward FFT computes the discrete Fourier transform (DFT) and takes the origin input from the time/space domain into the frequency domain.
The Backwards FFT, also commonly known as the Inverse FFT, is computation that goes from frequency domain back to time/space domain.

Batch size
----------
**Options:** Integer values

The batch size describes how many distributed FFT is computed. A minimum value of 1 is required. Values higher than 1 means that multiple distributed FFTs are computed at the same time. These FFTs are assumed to be interleaved. This means that the 1^{st} element of the 1^{st} FFT is followed by the 1^{st} element of the 2^{nd} FFT, and the 2^{nd} element of the 1^{st} FFT is preceded by the 1^{th} element of the last FFT, so forth. 

Processor Grid
--------------
**Options:** 1D grid, 2D grid

1D grid distribution assumes that all p processors are logically organized into a linear array. The entire 3D FFT is distributed along the Z dimension of the FFT. Using this processor grid, the X dimension of the FFT is assumed to be laid out consecutively in local memory.

2D grid distribtion assumes that all p processors are organized into a square grid of r \times c. The entire 3D FFT is distributed along the X and Y dimensions of the FFT, and the Z dimensions are stored consecutively.

Embedded
--------
**Options:** Embedded, Not Embedded

The current version of FFTX allows one to embed a data cube into a larger data cube that has been padded with zeros. Each dimension of the padded cube is twice that of the original dimensions. The 3D FFT is performed on the padded data cube. The data is embedded in the center, with equal number of zeros padded on both sides of the data cube. When a dimension of the original data cube is an odd size, the computation is undefined. 

MPI Type
--------
**Options:** Device-aware MPI (default), Host-based MPI

Two MPI versions are supported. At compile time, one can choose to compile for device-aware MPI or host-based MPI. The library does not check if the appropriate MPI is installed, and the behavior is undefined if the distributed library is compiled for an inappropriate MPI type. 

Packing
-------
**Options:** Host-based packing, Device-based Packing (default)

Packing routines are used to pack/unpack data from the MPI send/receive buffers into data buffers that are used for computation. Within FFTX, there are multiple variants of these packing routines. In general, these packing routines can be divided into host-based packing (i.e. packing on the CPU) or device-based packing (i.e. packing on the GPU). The choice of packing routines is set at compile time. Typically host-based packing are used for debugging/error checking purposes, while device-based packing are designed for performance.
