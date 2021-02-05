##  preamble for spiral_cpu_serial backend

comment("right now I don't know what goes in a preamble");

Load(fftx);
ImportAll(fftx);

conf := FFTXGlobals.defaultConf();
opts := FFTXGlobals.getOpts(conf);

# use the configuration for small mutidimensional real convolutions
# later we will have to auto-derive the correct options class

