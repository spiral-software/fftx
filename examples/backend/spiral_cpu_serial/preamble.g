##  preamble for spiral_cpu_serial backend

Load(fftx);
ImportAll(fftx);

conf := LocalConfig.fftx.defaultConf();

# use the configuration for small mutidimensional real convolutions
# later we will have to auto-derive the correct options class

