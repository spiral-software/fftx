The program **testmdprdft** runs the real-to-complex and complex-to-real
3D discrete Fourier transform (MDPRDFT and IMDPRDFT) in FFTX and reports
their timings.

On a GPU platform, either CUDA or HIP or SYCL, **testmddft* also
runs the same transformations with the vendor FFT library, respectively
either cuFFT or rocFFT or MKL FFT, and reports their timings, too.

To run:

```
./testmdprdft [-i <iterations>] [-s <M>x<N>x<K> ]
```

The ``-i`` flag specifies the number of iterations, with default 2.

The ``-s`` flag specifies the size, with default 32x40x48.

This example has no known execution issues.

If the size provided by -s is not available it will be generated and placed into $FFTX_HOME/cache_jit_files.

If it is available it will be pulled from either the fixed sized library src/library or $FFTX_HOME/cache_jit_files.

For the CPU build, some machines could have timing issues (times vary significantly). Please raise an issue with machine information if you see timing issues. 
