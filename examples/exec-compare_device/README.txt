Build an example with CUDA.

Assumes that the libraries have been built in the ../library directory.

Set FFTX_HOME to root of fftx repo.

Then do:
cmake -S . -B build
cmake --build build --target install

When running, make sure ../library/build/bin is in LD_LIBRARY_PATH.
