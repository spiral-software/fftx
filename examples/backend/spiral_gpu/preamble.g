##  preamble for spiral_gpu backend

Load(fftx);
ImportAll(fftx);

# use the configuration for small mutidimensional real convolutions
# later we will have to auto-derive the correct options class

conf := LocalConfig.fftx.confGPU();

##  end of preamble
