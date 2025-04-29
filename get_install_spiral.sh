#!/bin/bash
git clone -b develop https://github.com/spiral-software/spiral-software.git
export SPIRAL_HOME=$PWD/spiral-software
pushd $SPIRAL_HOME/namespaces/packages
  git clone -b develop https://www.github.com/spiral-software/spiral-package-fftx fftx
  git clone -b develop https://www.github.com/spiral-software/spiral-package-simt simt
  git clone -b develop https://www.github.com/spiral-software/spiral-package-mpi mpi
  git clone -b develop https://www.github.com/spiral-software/spiral-package-jit jit
popd
pushd $SPIRAL_HOME
  mkdir -p build
  pushd build
    cmake ..
    make install
  popd
popd
