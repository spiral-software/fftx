FFTX Project
============

This is the public repository for the FFTX API source, examples, and documentation.


## To build:

Clone __Spiral__ from https://github.com/spiral-software/spiral-software
and check out the __develop__ branch.

Set __SPIRAL_HOME__ to the directory where you have cloned __Spiral__.

Now from the home directory in the current repo:

    mkdir build
    pushd build
    cmake ..
    make

## To run:

The build process generates three examples, with these executables:

- build/examples/hockney/testhockney
- build/examples/rconv/testrconv
- build/examples/test_plan_dft/testmddft
