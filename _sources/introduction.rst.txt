
============
Introduction
============

FFTX is the exascale follow-on to the `FFTW <https://fftw.org>`_
open-source package for executing the discrete Fast Fourier
Transform, as well as higher-level operations composed of FFTs
and linear operators.
It is based on `SPIRAL <https://www.spiral.net>`_,
a build-time code generator that produces
high-performance kernels targeted to specific uses and platform
environments.

FFTX is being developed as a cooperative effort between Lawrence
Berkeley National Laboratory, Carnegie Mellon University, and
SpiralGen, Inc.

Motivation
----------

The workhorse FFT, used heavily across a wide spectrum of large
scientific applications, has typically been provided by vendor
libraries and a few open source projects, primarily FFTW.
As computing
platforms have quickly evolved into ever larger and more complex
heterogeneous composites of new types of processors with new
instruction sets and deep layers of parallelism, it has been
increasingly difficult to keep the libraries in sync with the rapid
progress, and FFTW is no longer under active development.  That leaves
a long list of still-relevant legacy code unable to take advantage of
the scale and speed of the new machines using their existing methods,
but FFTX provides a path for moving this code into exascale and
beyond.

FFTs are not especially useful by themselves,
but they show their real value in
the context of larger algorithms that have a typical sequence of:

1) gather a bunch of data
2) transform it with FFTs
3) do something to the transformed data
4) transform it back with inverse FFTs
5) pass on the interesting parts of the results

Using library routines for the FFTs still leaves a lot of complicated
platform-sensitive hand-written code to glue it all together.
Like the FFT libraries, reworking this non-trivial algorithm-specific
code for a new high-performance system's architecture is a task
approaching the edge of possible.  A key goal of FFTX's design is to
provide an API for specifying entire FFT-using kernels as single
hardware-independent entities, thus opening an avenue for tremendous
performance enhancements while freeing the scientific application
writer from the grungy hardware details.

Concept
-------

From a casual look at its :ref:`API <fftx_apis>`, FFTX appears to be a
pre-built library, but the heart of FFTX is a build-time code
generator, `SPIRAL <http://spiral.net>`_, that produces very high
performance kernels targeted to their specific uses and platform
environments.
Coupled to the platform-aware code generator is a sophisticated front
end that interprets the details of the algorithms from the FFTX API,
which it treats as a DSL for algorithm specification.

As part of the build process, the application is compiled and run in
*observe* mode, when FFTX follows the execution path through calls
into the FFTX API while implementing the calls with simple unoptimized
library code for correctness checks.  Once the use details are
collected, they are passed on to SPIRAL, which generates the
high-performance kernels for linking into the production version of
the code.  These two steps can be seamless parts of build requiring no
extra interaction.

License
-------

FFTX is a fully open-source project publicly available under a
BSD-style license.  Its main tool chain component SPIRAL, along with
its companion project :ref:`SpectralPACK <fftxpfft18>`, are both fully
open source with similar "no-strings" licenses.  This means you are
free to use FFTX for whatever you like, be it academic, commercial,
creating forks or derivatives, as long as you copy the license
statement if you redistribute it.

Although not required by the FFTX license, if you can, please cite
FFTX when using it in your work and also consider
:ref:`contributing <contribute>`
your changes back to the main FFTX project.


Citation
--------

To cite FFTX in publications use:

	Franz Franchetti, Daniele G. Spampinato, Anuva Kulkarni,
        Doru Thom Popovici, Tze Meng Low,
	Michael Franusich, Andrew Canning, Peter McCorquodale,
        Brian Van Straalen, and Phillip Colella,
	"**FFTX and SpectralPack: A First Look**",
	*IEEE International Conference on High Performance Computing,
        Data, and Analytics*,
        Bengaluru, India (2018), pp. 18--27,
        doi: 10.1109/HiPCW.2018.8634111.


Acknowledgments
---------------

This project was supported by the
`Exascale Computing Project <https://www.exascaleproject.org/>`_
(17-SC-20-SC), a joint project of the U.S. Department of Energy’s
Office of Science and National Nuclear Security Administration,
responsible for delivering a capable exascale ecosystem, including
software, applications, and hardware technology, to support the
nation’s exascale computing imperative.
