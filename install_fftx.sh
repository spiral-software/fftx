#!/bin/sh
# Assumes FFTX is already cloned into directory fftx.
export FFTX_HOME=$PWD/fftx
pushd $FFTX_HOME
  ./config-fftx-libs.sh CPU
  mkdir -p build
  pushd build
    cmake -DCMAKE_INSTALL_PREFIX=$FFTX_HOME -D_codegen=CPU ..
  popd
popd
