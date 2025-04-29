#!/bin/bash
git clone -b develop https://github.com/spiral-software/spiral-software.git ../spiral-software
pushd ../spiral-software
  export SPIRAL_HOME=$PWD
  pushd namespaces/packages
    git clone -b develop https://www.github.com/spiral-software/spiral-package-fftx fftx
    git clone -b develop https://www.github.com/spiral-software/spiral-package-simt simt
    git clone -b develop https://www.github.com/spiral-software/spiral-package-mpi mpi
    git clone -b develop https://www.github.com/spiral-software/spiral-package-jit jit
  popd
  mkdir -p build
  pushd build
    which cmake > /tmp/cmakewhich.txt
    cmakeversion > /tmp/cmakeversion.txt
    ls .. > /tmp/dir.txt
    cmake .. > /tmp/cmakeout.txt
    make install
  popd
popd
