## `examples/rconv`

These examples run tests of **FFTX** real 3D convolution transforms:
tests with random input and a constant-valued symbol,
a test on a delta function,
and a test of a Poisson equation solver.

The output gives the maximum error found
by all the tests performed on the convolution transform.
Maximum error on a test
is the maximum over all points of the absolute value of the
difference between the calculated answer and the correct answer.

On the tests with random input and a constant-valued symbol,
the number of rounds is specified with the flag `-i` (default 2).  

There is also a `verbosity` setting with the flag `-v` that defaults to 0.   
If `verbosity` is at least 1, then also writes out the
maximum error for each of the three test categories.   
If `verbosity` is at least 3, then also writes out the
maximum error for every round with random input.

* **testrconv**
```
./testrconv [-i rounds] [-v verbosity] [-s MMxNNxKK] [-h {print help message}]
```
Runs tests of **FFTX** real 3D convolution transform
on the fixed size `MMxNNxKK`, with default 24x32x40.

* **testrconv_lib**
```
./testrconv_lib [-i rounds] [-v verbosity] [-h {print help message}]
```
Runs tests of **FFTX** real 3D convolution transforms
for all 3D sizes in the **FFTX** library.
