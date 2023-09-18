## `examples/verify`

These examples all run a series of verification tests on
forward and inverse complex-to-complex,
real-to-complex and complex-to-real 3D FFTs, on random data.   
The tests belong to three categories:
1. Linearity on random data.
2. Impulses:  a unit impulse in one corner,
a unit impulse in one corner plus random data,
a constant (which should transform to an impulse in one corner),
a constant plus random data,
and a unit impulse at a random position.
3. Shifts:  a time shift in each dimension of random data
(*except* if the transform has real-valued output),
and a frequency shift in each dimension of random data
(*except* if the transform has real-valued input).

Reference: Funda Ergün,
"Testing multivariate linear functions: overcoming the generator bottleneck",
*Symposium on the Theory of Computing* (1995).

The output gives, for each transform, the maximum relative error found
by all the tests performed on that transform.
Maximum relative error on a test
is the maximum over all points of the absolute value of the
difference between the calculated answer and the correct answer,
divided by the maximum over all points of the absolute value
of the correct answer.

On the tests with random data or a random position,
the number of rounds is specified with the flag `-i` (default 2).

There is also a `verbosity` setting with the flag `-v` that defaults to 0.   
If `verbosity` is at least 1, then also writes out the
maximum relative error for each of the three test categories.   
If `verbosity` is at least 2, then also writes out the
maximum relative error for each type of test within each category.  
If `verbosity` is at least 3, then also writes out the
maximum relative error for every round with random input.

* **testverify**
```
./testverify [-i rounds] [-v verbosity] [-s MMxNNxKK] [-h {print help message}]
```
Runs the verification tests of **FFTX** transforms
on the fixed size `MMxNNxKK`, with default 24x32x40.

* **testverify_device**
```
./testverify_device [-i rounds] [-v verbosity] [-s MMxNNxKK] [-h {print help message}]
```
Runs the verification tests of either cuFFT transforms (on CUDA)
or rocFFT transforms (on HIP)
on the fixed size `MMxNNxKK`, with default 24x32x40.

* **testverify_lib**
```
./testverify_lib [-i rounds] [-v verbosity] [-h {print help message}]
```
Runs the verification tests of **FFTX** transforms
for all 3D sizes in the **FFTX** library.
