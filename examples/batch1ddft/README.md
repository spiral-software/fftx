The following example has some known execution issues for the CPU Backend:
Only the read sequential, write sequential case works (-r0x0 or the default case) all other strides will break


If the size provided by -s is not available it will be generated and placed into $FFTX_HOME/cache_jit_files

If it is available it will be pulled from either the fixed sized library src/library or $FFTX_HOME/cache_jit_files

For the CPU build some machines could have timing issues (times vary significantly). Please raise an issue with machine information if you see timing issues. 
