#!/bin/bash
timestamp=`date +%G%m%d_%H%M`
statusfile=$HOME/regression_status_$timestamp.txt
outputfile=$HOME/regression_output_$timestamp.txt
# Get and install SPIRAL, and set SPIRAL_HOME to its location.
# Should have spiral-software and fftx at same directory level.
pushd ..
  source get_install_spiral.sh
popd
if [ -f "$SPIRAL_HOME/bin/spiral" ]; then
    pushd fftx
      # Install FFTX, and set FFTX_HOME to its location.
      source install_fftx.sh
      if [ -d "bin" ]; then
          source bash_test_suite.sh $statusfile > $outputfile
      else
          echo "FAILED: FFTX not installed" > $statusfile
      fi
    popd
else
    echo "FAILED: Spiral not installed" > $statusfile
fi
