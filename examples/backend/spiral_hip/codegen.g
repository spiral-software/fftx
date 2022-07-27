##  Start of codegen -- for HIP

opts:=conf.getOpts(transform);
tt:= opts.tagIt(transform);
if(IsBound(fftx_includes)) then opts.includes:=fftx_includes; fi;
c:=opts.fftxGen(tt);

PrintTo(prefix::".fftx.source.cpp",opts.prettyPrint(c));

