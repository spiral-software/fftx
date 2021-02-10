##  Start of codegen -- for CPU

opts:=conf.getOpts(transform);
tt:= opts.tagIt(transform);
c:=opts.fftxGen(tt);

PrintTo(prefix::".fftx.source.cpp",opts.prettyPrint(c));
