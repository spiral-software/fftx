Collaborations with other projects
==================================

This page lists collaborations with other projects
to use FFTX for algorithm kernels.

Spinifel single-particle imaging
--------------------------------

`Spinifel <https://gitlab.osti.gov/mtip/spinifel>`_
is an application that recovers the density distribution of
molecules from X-ray diffraction images.
It is being developed at Lawrence Berkeley National Laboratory, SLAC
National Accelerator Laboratory, and Los Alamos National Laboratory.

Spinifel is written in Python and depends on the NumPy and CuPy libraries.
Two kernels within Spinifel make heavy use of FFTs:
*free-space convolution* and *phasing*.

The :ref:`Python Package for FFTX <python_for_fftx>`
includes a module, ``convo``, with interfaces to functions
performing these two operations.

- The **free-space convolution** kernel
  has two 3D input arrays:
  
  - ``ugrid``, of dimensions ``M x M x M``;
  - ``F_ugrid_conv_``, of dimensions ``2M x 2M x 2M``.

  The original Python code for this kernel is:

  ::

    ugrid_ups = cupy.zeros((2*M,) * 3, dtype=uvect.dtype)
    ugrid_ups[:M, :M, :M] = ugrid
    F_ugrid_ups = cupy.fft.fftn(np.fft.ifftshift(ugrid_ups))
    F_ugrid_conv_out_ups = F_ugrid_ups * F_ugrid_conv_
    ugrid_conv_out_ups = cupy.fft.fftshift(cupy.fft.ifftn(F_ugrid_conv_out_ups))
    ugrid_conv_out = ugrid_conv_out_ups[:M, :M, :M]

  The output array ``ugrid_conv_out`` has dimensions ``M x M x M``.
  
  With the Python Package for FFTX,
  we can replace this kernel with the line:

  ::
  
    ugrid_conv_out = fftx.convo.mdrfsconv(ugrid, F_ugrid_conv_)

- The **phasing** kernel
  has two 3D input arrays:

  - ``rho_``, of dimensions ``M x M x M``;
  - ``amplitudes``, also of dimensions ``M x M x M``.

  The original Python code for this kernel is:

  ::

    rho_hat_ = cupy.fft.fftn(rho_)
    phases_ = cupy.angle(rho_hat_)
    amp_mask_ = cupy.ones((M, M, M), dtype=cupy.bool_)
    amp_mask_[0, 0, 0] = 0
    rho_hat_mod_ = cupy.where(amp_mask_, amplitudes_ * cupy.exp(1j*phases_), rho_hat_)
    rho_mod_ = cupy.fft.ifftn(rho_hat_mod_).real

  The output array ``rho_mod_`` has dimensions ``M x M x M``.
  
  With the Python Package for FFTX,
  we can replace this kernel with the line:

  ::

    rho_mod_ = fftx.convo.stepphase(rho_, amplitudes_)

Both free-space convolution (``mdrfsconv``)
and phasing (``stepphase``) are implemented in the
``convo`` module of the
:ref:`Python Package for FFTX <python_for_fftx>`
as *integrated algorithms*.  That is to say, each instance on its particular
size is run through the SPIRAL code-generation system, optimizing
computation and communication within the whole kernel.
This approach speeds up both kernels by **4 times** over the
original Python with CuPy on arrays of the sizes used by Spinifel
(``M`` set to 81, 105, 125, 165, 189, or 225).
