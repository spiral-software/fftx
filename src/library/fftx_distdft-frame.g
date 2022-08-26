##  Copyright (c) 2018-2022, Carnegie Mellon University
##  See LICENSE for details

##  Distributed 3D complex DFTs

##  Script to generate code, will be driven by a size specification and will write the
##  CUDA/HIP code to a file.  The code will be compiled into a library for applications
##  to link against -- providing pre-compiled distributed FFTs of standard sizes.

Load(fftx);
ImportAll(fftx);
Load(mpi);
ImportAll(mpi);

##  grid := [ d1, d2 ];                 ##  Supplied from distdft-sizes.txt file
##  szcube := [ x, y, z ];              ##  Supplied from distdft-sizes.txt file
##  codefor := { "CUDA" | "HIP" | "CPU" };
##  file_suffix := { ".cu" | ".cpp" };
##  libdir := "lib_fftx_distdft_gpu_srcs";
##  fwd := { true | false };
##  embed := { true | false };

procGrid := MPIProcGridND(grid);

if codefor = "CUDA" then
    conf := LocalConfig.mpi.confMPIGPUDevice(procGrid);    
elif codefor = "CPU" then
    conf := LocalConfig.fftx.defaultConf();
elif codefor = "HIP" then
    conf := LocalConfig.mpi.confMPIGPUDevice(procGrid);    
fi;

name := "fftx_default_name";

if fwd then
    prefix := "fftx_distdft_";
    sign   := -1;
else
    prefix := "fftx_idistdft_";
    sign   := 1;
fi;

emcube := szcube;
if embed then
    emcube := List ( [1, 2, 3], i -> emcube[i] / 2 );
fi;
    
Nlocal := szcube{[1]}::List ( [2, 3], i-> szcube[i] / grid[i-1] );;
localBrick := TArrayND ( TComplex, Nlocal, [dimX, dimY, dimZ] );;
dataLayout := TGlobalArrayND ( procGrid, localBrick );;
Xglobal := tcast ( TPtr ( dataLayout ), X );;
Yglobal := tcast ( TPtr ( dataLayout ), Y );;

if embed then
    t := TFCall (
             TCompose ( [
                 TTensorII ( DFT ( szcube[2], sign ), [szcube[3], szcube[1]], ACubeRot_XYZ, ACubeRot_XYZ ),
                 TPrm ( fCubeEmbed ( Product ( [szcube[2], szcube[3], szcube[1]] ),
                                     "FFTX_MPI_EMBED_2", [Product ( [szcube[1], emcube[2], szcube[3]] ),
                                                          true, [szcube[1], emcube[2], szcube[3]] ] ) ),
                 TPrm ( fCubeTranspose ( Product ( [szcube[1], emcube[2], szcube[3]] ),
                                         "FFTX_MPI_3D_CUFFT_STAGE2", [Product ( [szcube[1], emcube[2], szcube[3]] ) ] ) ),
                 TTensorII ( DFT (szcube[1], sign ), [emcube[2], szcube[3]], ACubeRot_XYZ, ACubeRot_YZX ),
                 TPrm ( fCubeEmbed ( Product ( [szcube[1], emcube[2], szcube[3]] ),
                                     "FFTX_MPI_EMBED_1", [Product ( [szcube[3], emcube[1], emcube[2]] ),
                                                          true, [szcube[3], emcube[1], emcube[2]] ] ) ),
                 TPrm ( fCubeTranspose ( Product ( [szcube[3], emcube[1], emcube[2]] ),
                                         "FFTX_MPI_3D_CUFFT_STAGE1", [Product ( [szcube[3], emcube[1], emcube[2]] ) ] ) ),
                 TTensorII ( DFT ( szcube[3], sign), [emcube[1], emcube[2]], ACubeRot_ZXY, ACubeRot_XYZ )
             ] ),
             rec(fname := name, params := [ ] ) );

    name := prefix::"embed_g"::StringInt(grid[1])::"x"::StringInt(grid[2])::"_c";
    name := name::StringInt(szcube[1])::"x"::StringInt(szcube[2])::"x"::StringInt(szcube[3])::"_"::codefor;
else
    t := TFCall (
             TCompose ( [
                 TTensorII ( DFT ( szcube[2], sign ), [szcube[3], szcube[1]], ACubeRot_XYZ, ACubeRot_XYZ ),
                 TPrm ( fCubeTranspose ( Product (szcube),
                                         "FFTX_MPI_3D_CUFFT_STAGE2", [ Product ( szcube ), false ] ) ),
                 TTensorII ( DFT ( szcube[1], sign ), [szcube[2], szcube[3]], ACubeRot_ZXY, ACubeRot_XYZ ),
                 TPrm ( fCubeTranspose ( Product (szcube),
                                         "FFTX_MPI_3D_CUFFT_STAGE1", [ Product ( szcube ), false ] ) ),
                 TTensorII ( DFT ( szcube[3], sign ), [szcube[1], szcube[2]], ACubeRot_ZXY, ACubeRot_XYZ )
             ] ),
             rec(fname := name, params := [ ] ) );

    name := prefix::"g"::StringInt(grid[1])::"x"::StringInt(grid[2])::"_c";
    name := name::StringInt(szcube[1])::"x"::StringInt(szcube[2])::"x"::StringInt(szcube[3])::"_"::codefor;
fi;

PrintLine ( "fftx_distdft-frame: name = ", name, ", grid = ", grid, ", szcube = ",
            StringInt(szcube[1])::"x"::StringInt(szcube[2])::"x"::StringInt(szcube[3]) );

##    t.params[2].fname := prefix::StringInt(grid[1])::"x"::StringInt(grid[2])::"_c"::StringInt(szcube[1])::"x"::StringInt(szcube[2])::"x"::StringInt(szcube[3])::"_"::codefor;
t.params[2].fname := name;;

opts := conf.getOpts(t);;
if not IsBound ( libdir ) then
    libdir := "srcs";
fi;

##  We need the Spiral functions wrapped in 'extern C' for adding to a library
opts.wrapCFuncs := true;;

tt := opts.tagIt(t);;
if ( IsBound ( fftx_includes ) ) then opts.includes := fftx_includes; fi;
c := opts.fftxGen(tt);;

init_comm_fn_name := var("init_2d_comms");;
if embed then
    lst := [init_comm_fn_name]::grid::emcube::[true];
else
    lst := [init_comm_fn_name]::grid::szcube::[false];
fi;

init_comm := ApplyFunc(call, lst);;
cc := Collect(c, func);;
cc[1].cmd := chain(init_comm, cc[1].cmd);;

destroy_comm_fn_name := var("destroy_2d_comms");;
destroy_comm := ApplyFunc(call, [destroy_comm_fn_name]);;
cc[3].cmd := chain(destroy_comm, cc[3].cmd);;

PrintTo ( libdir::"/"::name::file_suffix, opts.prettyPrint(c));;
