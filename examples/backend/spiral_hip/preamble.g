##  preamble for spiral_hip backend

Load(fftx);
ImportAll(fftx);

# use the configuration for small mutidimensional real convolutions
# later we will have to auto-derive the correct options class

conf := FFTXGlobals.defaultHIPConf();

##  end of preamble
