#! /bin/bash

##  Source this script (don't just run it) so that it sets the environment variable for the parent shell

if [ -v SPIRAL_HOME ]; then
    ##  SPIRAL_HOME is set; don't pull down another version of spiral
    echo "Using the spiral version installed at: $SPIRAL_HOME"
    $SPIRAL_HOME/bin/spiral -B
else
    echo "SPIRAL_HOME is not set, install spiral"

    ##  We are in the root directory of the FFTX source tree, go up one level and install spiral
    pushd ..
    git clone https://github.com/spiral-software/spiral-software
    pushd spiral-software
    export SPIRAL_HOME=`pwd`
    pushd namespaces/packages
    git clone https://github.com/spiral-software/spiral-package-fftx fftx
    git clone https://github.com/spiral-software/spiral-package-simt simt
    git clone https://github.com/spiral-software/spiral-package-mpi mpi
    git clone https://github.com/spiral-software/spiral-package-jit jit
    popd
    mkdir build
    cd build
    cmake ..
    make install -j
    popd
    popd

    ##  New version of spiral installed - report info to user
    echo "New version of spiral installed at: $SPIRAL_HOME"
    $SPIRAL_HOME/bin/spiral -B
fi

##  Don't exit -- would exit the parent shell if source'd
