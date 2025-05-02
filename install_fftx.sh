#!/bin/sh
# Assumes FFTX is already cloned into directory fftx, and we are in it.
export FFTX_HOME=$PWD
./config-fftx-libs.sh CPU > config-fftx-libs.out
mkdir -p build
pushd build
  cmake -DCMAKE_INSTALL_PREFIX=$FFTX_HOME -D_codegen=CPU .. > ../cmake.out
  make install > ../make.out
popd
