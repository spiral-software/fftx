The following example has no known execution issues.

If the size provided by -s is not available it will be generated and placed into $FFTX_HOME/cache_jit_files

If it is available it will be pulled from either the fixed sized library src/library or $FFTX_HOME/cache_jit_files

For the CPU build some machines could have timing issues (times vary significantly). Please raise an issue with machine information if you see timing issues. 
