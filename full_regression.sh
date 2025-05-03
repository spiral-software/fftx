#!/bin/bash
echo "Starting `date`"
source get_install_spiral.sh
if [ -f "$SPIRAL_HOME/bin/spiral" ]; then
    echo "SUCCESS: Spiral installed at $SPIRAL_HOME/bin/spiral"
    # Install FFTX, and set FFTX_HOME to its location.
    source install_fftx.sh
    if [ -d "bin" ]; then
        echo "SUCCESS: FFTX installed at $FFTX_HOME"
        source test_suite.sh > test_suite.out
    else
        echo "FAILED: FFTX not installed at $FFTX_HOME"
    fi
else
    echo "FAILED: Spiral not installed at $SPIRAL_HOME"
fi
