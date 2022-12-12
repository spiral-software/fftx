
##  Copyright (c) 2018-2022, Carnegie Mellon University
##  See LICENSE for details

##  Script to generate code, will be driven by a size specification and will write the
##  CUDA/HIP code to a file.  The code will be compiled into a library for applications
##  to link against -- providing pre-compiled PSATD of standard sizes.

Load(fftx);
ImportAll(fftx);

if codefor = "CUDA" then
    conf := FFTXGlobals.confWarpXCUDADevice();
    ##  conf := LocalConfig.fftx.confGPU();
elif codefor = "HIP" then
    ##  Need a HIP specific version ... TBD
    ##  conf := FFTXGlobals.defaultHIPConf();
    conf := FFTXGlobals.confWarpXCUDADevice();
elif codefor = "CPU" then
    conf := FFTXGlobals.defaultWarpXConf();
fi;

opts := FFTXGlobals.getOpts(conf);

prefix := "fftx_psatd_";
name := prefix::StringInt(szcube[1])::ApplyFunc(ConcatenationString, List(Drop(szcube, 1), s->"x"::StringInt(s)));
name := name::"_"::codefor;

symvar := var("sym", TPtr(TPtr(TReal)));

##  szcube will be passed in

xdim := (szcube[1]+2)/2;
ydim := szcube[2];
zdim := szcube[3];
nx := szcube[1];
ny := szcube[2];
nz := szcube[3];
npx := nx+1;
npy := ny+1;
npz := nz+1;

var_1 := var("var_1", BoxND( [npz, npy, nx], TReal));
var_2 := var("var_2", BoxND( [npz, ny, npx], TReal));
var_3 := var("var_3", BoxND( [nz, npy, npx], TReal));
var_4 := var("var_4", BoxND( [nz, ny, npx], TReal));
var_5 := var("var_5", BoxND( [nz, npy, nx], TReal));
var_6 := var("var_6", BoxND( [npz, ny, nx], TReal));
var_7 := var("var_7", BoxND( [npz, npy, nx], TReal));
var_8 := var("var_8", BoxND( [npz, ny, npx], TReal));
var_9 := var("var_9", BoxND( [nz, npy, npx], TReal));
var_10 := var("var_10", BoxND( [npx, npy, npz], TReal));
var_11 := var("var_11", BoxND( [npx, npy, npz], TReal));
var_12 := var("var_12", BoxND( [1, 1, nx], TReal));
var_13 := var("var_13", BoxND( [1, ny, 1], TReal));
var_14 := var("var_14", BoxND( [nz, 1, 1], TReal));
var_15 := var("var_15", BoxND( [nz, ny, nx], TReal));
var_16 := var("var_16", BoxND( [nz, ny, nx], TReal));
var_17 := var("var_17", BoxND( [nz, ny, nx], TReal));
var_18 := var("var_18", BoxND( [nz, ny, nx], TReal));
var_19 := var("var_19", BoxND( [nz, ny, nx], TReal));
var_20 := var("var_20", BoxND( [npz, npy, nx], TReal));
var_21 := var("var_21", BoxND( [npz, ny, npx], TReal));
var_22 := var("var_22", BoxND( [nz, npy, npx], TReal));
var_23 := var("var_23", BoxND( [nz, ny, npx], TReal));
var_24 := var("var_24", BoxND( [nz, npy, nx], TReal));
var_25 := var("var_25", BoxND( [npz, ny, nx], TReal));
var_26 := var("var_26", BoxND( [11, nz, ny, nx], TReal));
var_27 := var("var_27", BoxND( [11, nz, ny, nx+2], TReal));
var_28 := var("var_28", BoxND( [6, nz, ny, nx+2], TReal));
var_29 := var("var_29", BoxND( [6, nz, ny, nx], TReal));
var_30 := var("var_30", BoxND( [nz, ny, nx], TReal));
var_30 := nth(var_26, 0);
var_31 := var("var_31", BoxND( [nz, ny, nx], TReal));
var_31 := nth(var_26, 1);
var_32 := var("var_32", BoxND( [nz, ny, nx], TReal));
var_32 := nth(var_26, 2);
var_33 := var("var_33", BoxND( [nz, ny, nx], TReal));
var_33 := nth(var_26, 3);
var_34 := var("var_34", BoxND( [nz, ny, nx], TReal));
var_34 := nth(var_26, 4);
var_35 := var("var_35", BoxND( [nz, ny, nx], TReal));
var_35 := nth(var_26, 5);
var_36 := var("var_36", BoxND( [nz, ny, nx], TReal));
var_36 := nth(var_26, 6);
var_37 := var("var_37", BoxND( [nz, ny, nx], TReal));
var_37 := nth(var_26, 7);
var_38 := var("var_38", BoxND( [nz, ny, nx], TReal));
var_38 := nth(var_26, 8);
var_39 := var("var_39", BoxND( [nz, ny, nx], TReal));
var_39 := nth(var_26, 9);
var_40 := var("var_40", BoxND( [nz, ny, nx], TReal));
var_40 := nth(var_26, 10);
var_41 := var("var_41", BoxND( [nz, ny, nx], TReal));
var_41 := nth(var_29, 0);
var_42 := var("var_42", BoxND( [nz, ny, nx], TReal));
var_42 := nth(var_29, 1);
var_43 := var("var_43", BoxND( [nz, ny, nx], TReal));
var_43 := nth(var_29, 2);
var_44 := var("var_44", BoxND( [nz, ny, nx], TReal));
var_44 := nth(var_29, 3);
var_45 := var("var_45", BoxND( [nz, ny, nx], TReal));
var_45 := nth(var_29, 4);
var_46 := var("var_46", BoxND( [nz, ny, nx], TReal));
var_46 := nth(var_29, 5);
var_1 := nth(X, 0);
var_2 := nth(X, 1);
var_3 := nth(X, 2);
var_4 := nth(X, 3);
var_5 := nth(X, 4);
var_6 := nth(X, 5);
var_7 := nth(X, 6);
var_8 := nth(X, 7);
var_9 := nth(X, 8);
var_10 := nth(X, 9);
var_11 := nth(X, 10);
var_20 := nth(Y, 0);
var_21 := nth(Y, 1);
var_22 := nth(Y, 2);
var_23 := nth(Y, 3);
var_24 := nth(Y, 4);
var_25 := nth(Y, 5);


    ix := Ind(xdim);
    iy := Ind(ydim);
    iz := Ind(zdim);
 
    c := 299792458;
    c2 := c^2;
    ep0 := 8.8541878128e-12;
    invep0 := 1.0 / ep0;
    ii := lin_idx(iz, iy, ix);
    var_12 := nth(nth(symvar, 0), ix);
    var_13 := nth(nth(symvar, 1), iy);
    var_14 := nth(nth(symvar, 2), iz);
    var_15 := nth(nth(symvar, 3), ii);
    var_16 := nth(nth(symvar, 4), ii);
    var_17 := nth(nth(symvar, 5), ii);
    var_18 := nth(nth(symvar, 6), ii);
    var_19 := nth(nth(symvar, 7), ii);

    div := nx * ny * nz;
    rmat := TSparseMat( [6, 11], [
        [0, [0, var_15/div],
            [4, cxpack(0, -var_14 * c2 * var_16/div)],
            [5, cxpack(0, var_13 * c2 * var_16/div)],
            [6, -invep0 * var_16/div],
            [9, cxpack(0,   var_12 * var_19/div)],
            [10, cxpack(0, -var_12 * var_18/div)]],
        [1, [1, var_15/div],
            [3, cxpack(0, var_14 * c2 * var_16/div)],
            [5, cxpack(0, -var_12 * c2 * var_16/div)],
            [7, -invep0 * var_16/div],
            [9, cxpack(0,   var_13 * var_19/div)],
            [10, cxpack(0, -var_13 * var_18/div)]],
        [2, [2, var_15/div],
            [3, cxpack(0, -var_13 * c2 * var_16/div)],
            [4,  cxpack(0, var_12 * c2 * var_16/div)],
            [8, -invep0 * var_16/div],
            [9, cxpack(0,   var_14 * var_19/div)],
            [10, cxpack(0, -var_14 * var_18/div)]],
    
        [3, [1, cxpack(0, var_14 * var_16/div)],
            [2, cxpack(0, -var_13 * var_16/div)],
            [3, var_15/div],
            [7, cxpack(0, -var_14 * var_17/div)],
            [8, cxpack(0, var_13 * var_17/div)]],
        [4, [0, cxpack(0, -var_14 * var_16/div)],
            [2, cxpack(0, var_12 * var_16/div)],
            [4, var_15/div],
            [6, cxpack(0, var_14 * var_17/div)],
            [8, cxpack(0, -var_12 * var_17/div)]],
        [5, [0, cxpack(0, var_13 * var_16/div)],      
            [1, cxpack(0, -var_12 * var_16/div)],
            [5, var_15/div],
            [6, cxpack(0, -var_13 * var_17/div)],
            [7, cxpack(0, var_12 * var_17/div)]]
        ]);
    
transform := TFCall(TDecl(TDAG( [
    TDAGNode(TResample( [nz, ny, nx], [npz, npy, nx], [0.00, 0.00, -0.50]), var_30, var_1),
    TDAGNode(TResample( [nz, ny, nx], [npz, ny, npx], [0.00, -0.50, 0.00]), var_31, var_2),
    TDAGNode(TResample( [nz, ny, nx], [nz, npy, npx], [-0.50, 0.00, 0.00]), var_32, var_3),
    TDAGNode(TResample( [nz, ny, nx], [nz, ny, npx], [-0.50, -0.50, 0.00]), var_33, var_4),
    TDAGNode(TResample( [nz, ny, nx], [nz, npy, nx], [-0.50, 0.00, -0.50]), var_34, var_5),
    TDAGNode(TResample( [nz, ny, nx], [npz, ny, nx], [0.00, -0.50, -0.50]), var_35, var_6),
    TDAGNode(TResample( [nz, ny, nx], [npz, npy, nx], [0.00, 0.00, -0.50]), var_36, var_7),
    TDAGNode(TResample( [nz, ny, nx], [npz, ny, npx], [0.00, -0.50, 0.00]), var_37, var_8),
    TDAGNode(TResample( [nz, ny, nx], [nz, npy, npx], [-0.50, 0.00, 0.00]), var_38, var_9),
    TDAGNode(TResample( [nz, ny, nx], [npx, npy, npz], [0.00, 0.00, 0.00]), var_39, var_10),
    TDAGNode(TResample( [nz, ny, nx], [npx, npy, npz], [0.00, 0.00, 0.00]), var_40, var_11),
    TDAGNode(TTensorI(MDPRDFT( [nz, ny, nx], -1), 11, APar, APar),  var_27, var_26),

    TDAGNode(TRC(TMap(rmat,  [iz,  iy,  ix],  AVec,  AVec)), var_28,  var_27),

    TDAGNode(TTensorI(IMDPRDFT( [nz, ny, nx], 1), 6, APar, APar),  var_29, var_28),
    TDAGNode(TResample( [npz, npy, nx], [nz, ny, nx], [0.00, 0.00, 0.50]), var_20, var_41),
    TDAGNode(TResample( [npz, ny, npx], [nz, ny, nx], [0.00, 0.50, 0.00]), var_21, var_42),
    TDAGNode(TResample( [nz, npy, npx], [nz, ny, nx], [0.50, 0.00, 0.00]), var_22, var_43),
    TDAGNode(TResample( [nz, ny, npx], [nz, ny, nx], [0.50, 0.50, 0.00]), var_23, var_44),
    TDAGNode(TResample( [nz, npy, nx], [nz, ny, nx], [0.50, 0.00, 0.50]), var_24, var_45),
    TDAGNode(TResample( [npz, ny, nx], [nz, ny, nx], [0.00, 0.50, 0.50]), var_25, var_46),

]),
   [var_26, var_27, var_28, var_29]
),
rec ( XType := TPtr(TPtr(TReal)), YType := TPtr(TPtr(TReal)), fname := name, params := [symvar])
);

##      opts := conf.getOpts(t);
    if not IsBound ( libdir ) then
        libdir := "srcs";
    fi;

    ##  We need the Spiral functions wrapped in 'extern C' for adding to a library
    opts.wrapCFuncs := true;

    tt := opts.tagIt(transform);
    if ( IsBound ( fftx_includes ) ) then opts.includes := fftx_includes; fi;
    c := opts.fftxGen(tt);
    ##  opts.prettyPrint(c);
    PrintTo ( libdir::"/"::name::file_suffix, opts.prettyPrint(c) );
