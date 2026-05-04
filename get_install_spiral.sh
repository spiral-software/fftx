#!/bin/bash
if [ -n "$SPIRAL_HOME" ]; then
    ##  SPIRAL_HOME is set; don't pull down another version of spiral
    echo "Using the spiral version installed at: $SPIRAL_HOME"
    $SPIRAL_HOME/bin/spiral -B
else
    echo "SPIRAL_HOME is not set; install spiral"
    git clone -b develop https://github.com/petermcLBL/spiral-software.git ../spiral-software
    pushd ../spiral-software
      sed -i '1s/^/#define _GNU_SOURCE\n/' gap/src/system.c
      export SPIRAL_HOME=$PWD
      pushd namespaces/packages
        git clone -b develop https://www.github.com/spiral-software/spiral-package-fftx fftx
        git clone -b develop https://www.github.com/spiral-software/spiral-package-simt simt
        git clone -b develop https://www.github.com/spiral-software/spiral-package-mpi mpi
        git clone -b develop https://www.github.com/spiral-software/spiral-package-jit jit
      popd
    mkdir -p build
    pushd build
      cmake .. &> ../cmake.out
      make install &> ../make.out
    popd
  popd
fi
# back to fftx directory
