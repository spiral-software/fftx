
============
Introduction
============

FFTX is the exascale follow-on to the `FFTW <http://fftw.org>`_ open source discrete FFT package for executing the Fast Fourier Transform as well as 
higher-level operations composed of linear operations combined with DFT transforms.  Though backwards compatible with FFTW, 
this is an entirely new work developed as a cooperative effort between Lawrence Berkeley National Laboratory, Carnegie Mellon University,
and SpiralGen, Inc.

Motivation
----------

The workhorse FFT, used heavily across a wide spectrum of large scientific applications, has typically been provided by vendor libraries and a few 
open source projects, primarily FFTW.  The FFTW API is a de facto standard for FFTs across platforms with many vendor libraries 
providing an FFTW-compatible implementations.  As computing platforms have quickly evolved into ever larger and more complex heterogeneous 
composites of new types of processors with new instruction sets and deep layers of parallelism, 
it has been increasingly difficult to keep the libraries in sync with the rapid progress,
and FFTW is no longer under active development.  That leaves a long list of still-relevant legacy code unable to take advantage of the 
scale and speed of the new machines using their existing methods, but FFTX provides a path for moving this code into exascale and beyond.

By themselves, FFTs aren't that useful.  They show their real value in the context of larger algorithms along the lines of:

1) gather a bunch of data
2) transform it with FFTs
3) do something to the transformed data
4) transform it back with inverse FFTs
5) pass on the interesting parts of the results

Using library routines for the FFTs still leaves a lot of complicated platform-sensitive hand-written code to glue it all together.  
Like the FFT libraries, reworking this non-trivial algorithm-specific code for a new high 
performance system's architecture is a task approaching the edge of possible.  A key goal of FFTX's design is to 
provide an API for specifying entire FFT-using kernels as single hardware-independent entities, thus opening an avenue 
for tremendous performance enhancements while freeing the scientific application writer from the grungy hardware details.

Concept
-------

From a casual look at its :ref:`API <fftx_apis>`, FFTX appears to be a pre-built library, but the heart of FFTX is a
build-time code generator, `SPIRAL <http://spiral.net>`_, that produces very high performance kernels targeted to their specific uses and 
platform environments.  Coupled to the platform-aware code generator is a sophisticated front end that interprets the details of the algorithms 
from the FFTX API, which it treats as a DSL for algorithm specification.

A key feature of the FFTX API is the concept of nested plans, or "plans of plans".  The individual steps of the algorithm are 
described by FFTW-like plans, then collected together as part of a compound plan that presents the entire algorithm as 
a single kernel.

As part of the build process, the application is compiled and run in *observe* mode, when FFTX follows the execution path through calls into
the FFTX API while implementing the calls with simple unoptimized library code for correctness checks.  Once the use details are collected, they
are passed on to SPIRAL, which generates the high-performance kernels for linking into the production version of the code.  These two steps can
be seamless parts of build requiring no extra interaction.

For maximum effectiveness in producing high-performance kernels, FFTX provides a finer-grained API based
on `FFTW's Guru Interface <http://www.fftw.org/fftw3_doc/Guru-Interface.html>`_.  FFTX adds a powerful construct,
the "plan of plans", along with enhancements to the ``fftw_iodim``, which allows an arbitrarily complex algorithm to be 
expressed as a single entity.  Application programmers can detail data layout, link together sequential operations, and specify 
various aspects of parallelism through one API.


License
-------

FFTX is a fully open source project publicly available under a BSD-style license.  Its main tool chain component SPIRAL, along 
with its companion project :ref:`SpectralPACK <fftxpfft18>`, are both fully open source with similar "no-strings" licenses.  
This means you are free to use FFTX for whatever you like, be it academic, commercial, creating forks or derivatives, 
as long as you copy the license statement if you redistribute it.

Although not required by the FFTX license, if you can, please cite FFTX when using it in your work and also 
consider :ref:`contributing <contribute>` your changes back to the main FFTX project.


Citation
--------

To cite FFTX in publications use:

	| Franz Franchetti, Daniele G. Spampinato, Anuva Kulkarni, Doru Thom Popovici, Tze Meng Low,
	| Michael Franusich, Andrew Canning, Peter McCorquodale, Brian Van Straalen, Phillip Colella
	| **FFTX and SpectralPack: A First Look**.
	| *IEEE International Conference on High Performance Computing, Data, and Analytics*, 2018


Acknowledgments
---------------

This project was supported by the `Exascale Computing Project <https://www.exascaleproject.org/>`_
(17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative. 




