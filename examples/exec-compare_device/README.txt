Build an example with CUDA.

Assumes that the libraries have been built in the ../library/build directory,
say with cmake .. -DPROJECT=dummy

Set FFTX_HOME to root of fftx repo.

Then all you do is:  make

When running, make sure ../library/build/bin is in LD_LIBRARY_PATH.
