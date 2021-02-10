##  Start of codegen -- for GPU

opts:=conf.getOpts(transform);
tt:= opts.tagIt(transform);
c:=opts.fftxGen(tt);

PrintTo(prefix::".fftx.source.cu",opts.prettyPrint(c));

