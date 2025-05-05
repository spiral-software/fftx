#!/bin/sh
# Called from full_regression.sh.
# Assumes FFTX is already cloned into directory fftx, and we are in it.
# Need _codegen set to one of: CPU CUDA HIP SYCL.
# Optional _compilerspec set to -DCMAKE_CXX_COMPILER=(compiler)
export FFTX_HOME=$PWD
./config-fftx-libs.sh $_codegen > config-fftx-libs.out
mkdir -p build
pushd build
  cmake -DCMAKE_INSTALL_PREFIX=$FFTX_HOME $_compilerspec -D_codegen=$_codegen .. > ../cmake.out
  make install > ../make.out
popd
