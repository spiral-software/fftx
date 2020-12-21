##  Start of codegen

c:=opts.fftxGen(transform);

PrintTo(prefix::".fftx.source.cu",opts.prettyPrint(c));
