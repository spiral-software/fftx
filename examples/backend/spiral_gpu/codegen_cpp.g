##  Start of codegen -- for GPU

opts:=conf.getOpts(transform);
tt:= opts.tagIt(transform);
if(IsBound(fftx_includes)) then opts.includes:=fftx_includes; fi;
c:=opts.fftxGen(tt);

PrintTo(prefix::".fftx.source.cpp",opts.prettyPrint(c));

