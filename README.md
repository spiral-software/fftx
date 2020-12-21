FFTX Project
============

This is the public repository for the FFTX API source, examples, and documentation.


## To build:

Clone __Spiral__ from https://github.com/spiral-software/spiral-software
and check out the __develop__ branch.

Set __SPIRAL_HOME__ to the directory where you have cloned __Spiral__.

In addition, you will need two __Spiral__ packages:
- Clone __spiral-package-fftx__ from https://github.com/spiral-software/spiral-package-fftx
into the directory
$SPIRAL_HOME/namespaces/packages/fftx
- Clone __spiral-package-simt__ from https://github.com/spiral-software/spiral-package-simt
into the directory
$SPIRAL_HOME/namespaces/packages/simt

Now from the home directory for the current repository:

    mkdir build
    pushd build
    cmake ..
    make
    popd

## To run:

The build process generates examples with these executables:

- build/examples/hockney/testhockney
- build/examples/rconv/testrconv
- build/examples/test_plan_dft/testmddft
- build/examples/verify/testverify
