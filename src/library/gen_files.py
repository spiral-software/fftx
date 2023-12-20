#! python

##  Copyright (c) 2018-2023, Carnegie Mellon University
##  See LICENSE for details

##  This script reads a file of cube sizes (command line arg), that contains several cube size
##  specifications for the 3D DFT.  This script will:
##      Generate a list of source file names for CMake to build
##      Create the source files (by running Spiral), writing them
##      to directory lib_<transform>_srcs
##      Create the prototype definitions in a private library include file: <transform>_decls.h
##      Create the public header file for the library: <transform>_public.h
##  Compiling the library is handled by CMake.
##
##  Usage:
##    python gen_files.py -t transform -s sizes_file -p platform [-i] [-m master_sizes]
##  where:
##    transform is the base transform to use for the library (e.g., fftx_mddft)
##    sizes_file is the file specifying the sizes to build for transform/platform
##    platform specifies the platform, e.g., cpu | cuda | hip
##    -i specifies the inverse direction -- forward is the default when not specified
##    master_sizes when present tells python to perform a second pass, skipping the Spiral code
##          generation -- used to regenerate the API and CMake files when the code exists

##  gen_files will build a separate library for each transform by option (e.g., separate
##  libraries are built for forward and inverse transforms; for CPU and GPU code (NOTE: We
##  only consider the case where a system will have GPUs of one type, thus any GPU target
##  (cuda | hip | sycl [later]) will be built in a 'gpu' library).  The intent is to have
##  only transforms of a single type and similar signature in a library (NOTE: this
##  implies separate libraries for float vs double and Fortran ordered data vs C ordered).

import sys
import subprocess
import os, stat
import re
import shutil
import argparse
from script import Script
from codebuilder import CodeBuilder

################  Definitions pulled from SpiralPy / constants.py  ############


SP_METADATA_START   = '!!START_METADATA!!'
SP_METADATA_END     = '!!END_METADATA!!'
SP_STR_DOUBLE       = 'Double'
SP_STR_SINGLE       = 'Single'
SP_STR_FORWARD      = 'Forward'
SP_STR_INVERSE      = 'Inverse'
SP_STR_C            = 'C'
SP_STR_FORTRAN      = 'Fortran'

SP_TRANSFORM_BATDFT     = 'BATDFT'
SP_TRANSFORM_BATMDDFT   = 'BATMDDFT'
SP_TRANSFORM_DFT        = 'DFT'
SP_TRANSFORM_MDDFT      = 'MDDFT'
SP_TRANSFORM_MDRCONV    = 'MDRCONV'
SP_TRANSFORM_MDRFSCONV  = 'MDRFSCONV'
SP_TRANSFORM_MDPRDFT    = 'MDPRDFT'
SP_TRANSFORM_UNKNOWN    = 'UNKNOWN'

SP_KEY_BATCHSIZE        = 'BatchSize'
SP_KEY_DESTROY          = 'Destroy'
SP_KEY_DIMENSIONS       = 'Dimensions'
SP_KEY_DIRECTION        = 'Direction'
SP_KEY_EXEC             = 'Exec'
SP_KEY_FILENAME         = 'Filename'
SP_KEY_FUNCTIONS        = 'Functions'
SP_KEY_INIT             = 'Init'
SP_KEY_METADATA         = 'Metadata'
SP_KEY_NAMES            = 'Names'
SP_KEY_ORDER            = 'Order'
SP_KEY_PLATFORM         = 'Platform'
SP_KEY_PRECISION        = 'Precision'
SP_KEY_READSTRIDE       = 'ReadStride'
SP_KEY_SPIRALBUILDINFO  = 'SpiralBuildInfo'
SP_KEY_TRANSFORMS       = 'Transforms'
SP_KEY_TRANSFORMTYPE    = 'TransformType'
SP_KEY_TRANSFORMTYPES   = 'TransformTypes'

###################################


def start_header_file ( type, script ):
    """ Sets up the common stuff for header files:
        Add the function prototype definitions for the library
    """
        
    str = CodeBuilder ( f'#ifndef {script.file_stem}{type}{script.decor_platform}HEADER_INCLUDED' + '\n' )
    str.append ( f'#define {script.file_stem}{type}{script.decor_platform}HEADER_INCLUDED' + '\n\n' )
    str.append ( '//  Copyright (c) 2018-2023, Carnegie Mellon University\n' )
    str.append ( '//  See LICENSE for details\n\n' )

    str.append ( '#include "fftx3.hpp"\n\n' )

    str.append ( '#ifndef INITTRANSFORMFUNC\n' )
    str.append ( '#define INITTRANSFORMFUNC\n' )
    str.append ( 'typedef void ( * initTransformFunc ) ( void );\n' )
    str.append ( '#endif\n\n' )

    str.append ( '#ifndef DESTROYTRANSFORMFUNC\n' )
    str.append ( '#define DESTROYTRANSFORMFUNC\n' )
    str.append ( 'typedef void ( * destroyTransformFunc ) ( void );\n' )
    str.append ( '#endif\n\n' )

    str.append ( '#ifndef RUNTRANSFORMFUNC\n' )
    str.append ( '#define RUNTRANSFORMFUNC\n' )

    str.append ( 'typedef void ( * runTransformFunc ) ' )
    if script.xform_name == 'psatd':
        str.append ( '( double **output, double **input, double **sym );\n' )
    else:
        str.append ( '( double *output, double *input, double *sym );\n' )
    str.append ( '#endif\n\n' )

    str.append ( '#ifndef TRANSFORMTUPLE_T\n' )
    str.append ( '#define TRANSFORMTUPLE_T\n' )
    str.append ( 'typedef struct transformTuple {\n' )
    str.append ( '    initTransformFunc    initfp;\n' )
    str.append ( '    destroyTransformFunc destroyfp;\n' )
    str.append ( '    runTransformFunc     runfp;\n' )
    str.append ( '} transformTuple_t;\n' )
    str.append ( '#endif\n\n' )

    return str.get();


def body_public_header ( script ):
    "Add the body details for the public header file"

    str = CodeBuilder ( '//  Query the list of sizes available from the library; returns a pointer to an\n' )
    str.append ( '//  array of sizes, each element is a struct of type fftx::point_t<3> specifying the X,\n' )
    str.append ( '//  Y, and Z dimensions\n\n' )

    str.append ( f'fftx::point_t<3> * {script.file_stem}{script.decor_platform}QuerySizes ();' + '\n' )
    str.append ( f'#define {script.file_stem}QuerySizes {script.file_stem}{script.decor_platform}QuerySizes' + '\n\n' )

    str.append ( f'//  Run an {script.file_stem} transform once: run the init functions, run the,' + '\n' )
    str.append ( '//  transform and finally tear down by calling the destroy function.\n' )
    str.append ( '//  Accepts fftx::point_t<3> specifying size, and pointers to the output\n' )
    str.append ( '//  (returned) data and the input data.\n\n' )


    str.append ( f'void {script.file_stem}{script.decor_platform}Run ' )
    if script.xform_name == 'psatd':
        str.append ( '( fftx::point_t<3> req, double ** output, double ** input, double ** sym );\n' )
    else:
        str.append ( '( fftx::point_t<3> req, double * output, double * input, double * sym );\n' )

    str.append ( f'#define {script.file_stem}Run {script.file_stem}{script.decor_platform}Run' + '\n\n' )

    str.append ( '//  Get a transform tuple -- a set of pointers to the init, destroy, and run\n' )
    str.append ( f'//  functions for a specific size {script.file_stem} transform.  Using this' + '\n' )
    str.append ( '//  information the user may call the init function to setup for the transform,\n' )
    str.append ( '//  then run the transform repeatedly, and finally tear down (using destroy function).\n\n' )

    str.append ( f'transformTuple_t * {script.file_stem}{script.decor_platform}Tuple ( fftx::point_t<3> req );' + '\n' )
    str.append ( f'#define {script.file_stem}Tuple {script.file_stem}{script.decor_platform}Tuple' + '\n\n' )

    str.append ( '//  The metadata table is compiled into the library (and thus readable by scanning\n' )
    str.append ( '//  the file, without having to load the library).\n' )
    str.append ( '//  Add a simple function to get the metadata (for debug purposes).\n\n' )

    str.append ( f'char * {script.file_stem}{script.decor_platform}GetMetaData ();' + '\n\n' )

    if script.xform_name != 'psatd':
        str.append ( '//  Wrapper functions to allow python to call CUDA/HIP GPU code.\n\n' )
        str.append ( 'extern "C" {\n\n' )
        str.append ( f'int  {script.file_stem}{script.decor_platform}python_init_wrapper ( int * req );' + '\n' )

        str.append ( f'void {script.file_stem}{script.decor_platform}python_run_wrapper ' )
        str.append ( '( int * req, double * output, double * input, double * sym );\n' )

        str.append ( f'void {script.file_stem}{script.decor_platform}python_destroy_wrapper ( int * req );' + '\n\n}\n\n' )

    str.append ( '#endif\n\n' )

    return str.get();


def library_api ( script ):
    """Sets up the API file for the library, includes the following:
           included required header files and defines the functions:
           {script.file_stem}{script.decor_platform}{QuerySizes | Tuple | Run}
    """

    str = CodeBuilder ( '//  Copyright (c) 2018-2023, Carnegie Mellon University\n' )
    str.append ( '//  See LICENSE for details\n\n' )

    str.append ( '#include <stdio.h>\n' )
    str.append ( '#include <stdlib.h>\n' )
    str.append ( '#include <string.h>\n' )
    str.append ( f'#include "{script.file_stem}{script.decor_platform}decls.h"' + '\n' )
    str.append ( f'#include "{script.file_stem}{script.decor_platform}public.h"' + '\n\n' )

    if script.args.platform == 'CUDA':
        str.append ( '#include <helper_cuda.h>\n\n' )
    elif script.args.platform == 'HIP':
        str.append ( '#include <hip/hip_runtime.h>\n\n' )
        str.append ( '#define checkLastHipError(str)   { hipError_t err = hipGetLastError();   ' )
        str.append ( 'if (err != hipSuccess) {  printf("%s: %s\\n", (str), hipGetErrorString(err) );  ' )
        str.append ( 'exit(-1); } }\n\n' )
    elif script.args.platform == 'SYCL':
        str.append ( '#include <CL/sycl.hpp>\n\n' )

    str.append ( '//  Query the list of sizes available from the library; returns a pointer to an\n' )
    str.append ( '//  array of size <N+1>, each element is a struct of type fftx::point_t<3> specifying the X,\n' )
    str.append ( '//  Y, and Z dimensions of a transform in the library.  <N> is the number of sizes defined;\n' )
    str.append ( '//  the last entry in the returned list has all dimensions equal 0.\n\n' )

    str.append ( f'fftx::point_t<3> * {script.file_stem}{script.decor_platform}QuerySizes ()' + '\n{\n' )
    str.append ( '    fftx::point_t<3> *wp = (fftx::point_t<3> *) ' )
    str.append ( f' malloc ( sizeof ( AllSizes3_{script.args.platform} ) );' + '\n' )
    str.append ( '    if ( wp != NULL)\n' )
    str.append ( f'        memcpy ( (void *) wp, (const void *) AllSizes3_{script.args.platform}, ' )
    str.append ( f'sizeof ( AllSizes3_{script.args.platform} ) )' + ';\n\n' )
    str.append ( '    return wp;\n' )
    str.append ( '}\n\n' )

    str.append ( '//  Get a transform tuple -- a set of pointers to the init, destroy, and run\n' )
    str.append ( f'//  functions for a specific size {script.file_stem} transform.  Using this\n' )
    str.append ( '//  information the user may call the init function to setup for the transform,\n' )
    str.append ( '//  then run the transform repeatedly, and finally tear down (using the destroy\n' )
    str.append ( '//  function).  Returns NULL if requested size is not found\n\n' )

    str.append ( f'transformTuple_t * {script.file_stem}{script.decor_platform}Tuple ( fftx::point_t<3> req )' )
    str.append ( '\n{\n' )
    str.append ( '    int indx;\n' )
    str.append ( f'    int numentries = sizeof ( AllSizes3_{script.args.platform} ) /' )
    str.append ( ' sizeof ( fftx::point_t<3> ) - 1;    // last entry is { 0, 0, 0 }\n' )
    str.append ( '    transformTuple_t *wp = NULL;\n\n' )

    str.append ( '    for ( indx = 0; indx < numentries; indx++ ) {\n' )
    str.append ( f'        if ( req[0] == AllSizes3_{script.args.platform}[indx][0] &&' + '\n' )
    str.append ( f'             req[1] == AllSizes3_{script.args.platform}[indx][1] &&' + '\n' )
    str.append ( f'             req[2] == AllSizes3_{script.args.platform}[indx][2] )' + ' {\n' )
    str.append ( '            // found a match\n' )
    str.append ( '            wp = (transformTuple_t *) malloc ( sizeof ( transformTuple_t ) );\n' )
    str.append ( '            if ( wp != NULL) {\n' )
    str.append ( f'                *wp = {script.file_stem}{script.args.platform}_Tuples[indx];' + '\n' )
    str.append ( '            }\n' )
    str.append ( '            break;\n' )
    str.append ( '        }\n' )
    str.append ( '    }\n\n' )

    str.append ( '    return wp;\n' )
    str.append ( '}\n\n' )

    str.append ( f'//  Run an {script.file_stem} transform once: run the init functions, run the' + '\n' )
    str.append ( '//  transform and finally tear down by calling the destroy function.\n' )
    str.append ( '//  Accepts fftx::point_t<3> specifying size, and pointers to the output\n' )
    str.append ( '//  (returned) data and the input data.\n\n' )

    str.append ( f'void {script.file_stem}{script.decor_platform}Run ' )
    if script.xform_name == 'psatd':
        str.append ( '( fftx::point_t<3> req, double ** output, double ** input, double ** sym )\n' )
    else:
        str.append ( '( fftx::point_t<3> req, double * output, double * input, double * sym )\n' )

    str.append ( '{\n' )
    str.append ( f'    transformTuple_t *wp = {script.file_stem}{script.decor_platform}Tuple ( req );' + '\n' )
    str.append ( '    if ( wp == NULL )\n' )
    str.append ( '        //  Requested size not found -- just return\n' )
    str.append ( '        return;\n\n' )

    str.append ( '    //  Call the init function\n' )
    str.append ( '    ( * wp->initfp )();\n' )
    str.append ( '    //  checkCudaErrors ( cudaGetLastError () );\n\n' )

    str.append ( '    ( * wp->runfp ) ( output, input, sym );\n' )
    str.append ( '    //  checkCudaErrors ( cudaGetLastError () );\n\n' )

    str.append ( '    //  Tear down / cleanup\n' )
    str.append ( '    ( * wp->destroyfp ) ();\n' )
    str.append ( '    //  checkCudaErrors ( cudaGetLastError () );\n\n' )

    str.append ( '    return;\n' )
    str.append ( '}\n\n' )

    return str.get();


def python_cuda_api ( script ):
    """Sets up the python wrapper API to allow Python to call C++ CUDA or HIP code.
       For CPU code no data management or marshalling is required
    """

    if script.xform_name == 'psatd':
        return '';                  ## no easy way to test psatd with simple python interface

    str = CodeBuilder ( '//  Host-to-Device C/CUDA/HIP wrapper functions to permit Python to call the kernels.' )
    str.append ( '\n\nextern "C" {\n\n' )

    if script.args.platform == 'CUDA' or script.args.platform == 'HIP' or script.args.platform == 'SYCL':
        str.append ( 'static double *dev_in, *dev_out, *dev_sym;\n\n' )

    str.append ( f'int {script.file_stem}{script.decor_platform}python_init_wrapper ( int * req )' )
    str.append ( '\n{\n    //  Get the tuple for the requested size\n' )
    str.append ( '    fftx::point_t<3> rsz;\n' )
    str.append ( '    rsz[0] = req[0];  rsz[1] = req[1];  rsz[2] = req[2];\n' )
    str.append ( f'    transformTuple_t *wp = {script.file_stem}{script.decor_platform}Tuple ( rsz );' )
    str.append ( '\n    if ( wp == NULL )\n' )
    str.append ( '        //  Requested size not found -- return false\n' )
    str.append ( '        return 0;\n\n' )

    if script.args.platform == 'CUDA':
        _mmalloc = 'cudaMalloc'
        _errchk  = 'checkCudaErrors ( cudaGetLastError () );'
        _mmemcpy = 'cudaMemcpy'
        _cph2dev = 'cudaMemcpyHostToDevice'
        _cpdev2h = 'cudaMemcpyDeviceToHost'
        _memfree = 'cudaFree'
    elif script.args.platform == 'HIP' or script.args.platform == 'SYCL':
        _mmalloc = 'hipMalloc'
        _errchk  = 'checkLastHipError ( "Error: " );'
        _mmemcpy = 'hipMemcpy'
        _cph2dev = 'hipMemcpyHostToDevice'
        _cpdev2h = 'hipMemcpyDeviceToHost'
        _memfree = 'hipFree'

    if script.args.platform == 'CUDA' or script.args.platform == 'HIP' or script.args.platform == 'SYCL':
        ##  Amount of data space to malloc depends on transform:
        ##     MDDFT/IMDDFT:   x * y * z * 2 doubles (for C2C, both input & output)
        ##     MDPRDFT:        x * y * z     doubles (for R2C, input)
        ##                     x * y * ((z/2) + 1) * 2 doubles (for R2C, output)
        ##     IMDPRDFT:       x * y * ((z/2) + 1) * 2 doubles (for C2R, input)
        ##                     x * y * z     doubles (for C2R, output)
        if script.xform_name == 'mddft':
            str.append ( '    int ndoubin  = (int)(req[0] * req[1] * req[2] * 2);\n' )
            str.append ( '    int ndoubout = (int)(req[0] * req[1] * req[2] * 2);\n' )
        elif script.xform_name == 'mdprdft':
            if script.args.inverse:
                str.append ( '    int ndoubin  = (int)(req[0] * req[1] * ((int)(req[2]/2) + 1) * 2);\n' )
                str.append ( '    int ndoubout = (int)(req[0] * req[1] * req[2] );\n' )
            else:
                str.append ( '    int ndoubin  = (int)(req[0] * req[1] * req[2] );\n' )
                str.append ( '    int ndoubout = (int)(req[0] * req[1] * ((int)(req[2]/2) + 1) * 2);\n' )
        elif script.xform_name == 'rconv':
            str.append ( '    int ndoubin  = (int)(req[0] * req[1] * ((int)(req[2]/2) + 1) * 2);\n' )
            str.append ( '    int ndoubout = (int)(req[0] * req[1] * req[2] );\n' )
        elif script.xform_name == 'psatd':
            str.append ( '    int ndoubin  = (int)(req[0] * req[1] * req[2] );\n' )
            str.append ( '    int ndoubout = (int)(req[0] * req[1] * ((int)(req[2]/2) + 1) * 2);\n' )
            
        str.append ( '    if ( ndoubin  == 0 )\n        return 0;\n\n' )
        str.append ( '    ' + _mmalloc + ' ( &dev_in,  sizeof(double) * ndoubin  );\n' )
        str.append ( '    ' + _mmalloc + ' ( &dev_out, sizeof(double) * ndoubout );\n' )
        str.append ( '    ' + _mmalloc + ' ( &dev_sym, sizeof(double) * 1000 );\n' )
        str.append ( '    ' + _errchk +  '\n\n' )

    str.append ( '    //  Call the init function\n' )
    str.append ( '    ( * wp->initfp )();\n' )
    if script.args.platform == 'CUDA' or script.args.platform == 'HIP' or script.args.platform == 'SYCL':
        str.append ( '    ' + _errchk +  '\n\n' )

    str.append ( '    return 1;\n}\n\n' )

    str.append ( f'void {script.file_stem}{script.decor_platform}python_run_wrapper ' )
    if script.xform_name == 'psatd':
        str.append ( '( int * req, double ** output, double ** input, double ** sym )\n{\n' )
    else:
        str.append ( '( int * req, double * output, double * input, double * sym )\n{\n' )

    str.append ( '    //  Get the tuple for the requested size\n' )
    str.append ( '    fftx::point_t<3> rsz;\n' )
    str.append ( '    rsz[0] = req[0];  rsz[1] = req[1];  rsz[2] = req[2];\n' )
    str.append ( f'    transformTuple_t *wp = {script.file_stem}{script.decor_platform}Tuple ( rsz );' )
    str.append ( '\n    if ( wp == NULL )\n' )
    str.append ( '        //  Requested size not found -- just return\n' )
    str.append ( '        return;\n\n' )

    if script.args.platform == 'CUDA' or script.args.platform == 'HIP' or script.args.platform == 'SYCL':
        if script.xform_name == 'mddft':
            str.append ( '    int ndoubin  = (int)(req[0] * req[1] * req[2] * 2);\n' )
            str.append ( '    int ndoubout = (int)(req[0] * req[1] * req[2] * 2);\n' )
        elif script.xform_name == 'mdprdft':
            if script.args.inverse:
                str.append ( '    int ndoubin  = (int)(req[0] * req[1] * ((int)(req[2]/2) + 1) * 2);\n' )
                str.append ( '    int ndoubout = (int)(req[0] * req[1] * req[2] );\n' )
            else:
                str.append ( '    int ndoubin  = (int)(req[0] * req[1] * req[2] );\n' )
                str.append ( '    int ndoubout = (int)(req[0] * req[1] * ((int)(req[2]/2) + 1) * 2);\n' )
        elif script.xform_name == 'rconv':
            str.append ( '    int ndoubin  = (int)(req[0] * req[1] * ((int)(req[2]/2) + 1) * 2);\n' )
            str.append ( '    int ndoubout = (int)(req[0] * req[1] * req[2] );\n' )
        elif script.xform_name == 'psatd':
            str.append ( '    int ndoubin  = (int)(req[0] * req[1] * req[2] );\n' )
            str.append ( '    int ndoubout = (int)(req[0] * req[1] * ((int)(req[2]/2) + 1) * 2);\n' )

        str.append ( '    if ( ndoubin  == 0 )\n        return;\n\n' )
        str.append ( '    ' + _mmemcpy + ' ( dev_in, input, sizeof(double) * ndoubin, ' + _cph2dev + ' );\n\n' )

    str.append ( '    //  Call the run function\n' )
    if script.args.platform == 'CUDA' or script.args.platform == 'HIP' or script.args.platform == 'SYCL':
        str.append ( '    ( * wp->runfp )( dev_out, dev_in, dev_sym );\n' )
        str.append ( '    ' + _errchk  + '\n\n' )
        str.append ( '    ' + _mmemcpy + ' ( output, dev_out, sizeof(double) * ndoubout, ' + _cpdev2h + ' );\n' )
    else:
        str.append ( '    ( * wp->runfp )( output, input, sym );\n' )
        
    str.append ( '    return;\n}\n\n' )

    str.append ( f'void {script.file_stem}{script.decor_platform}python_destroy_wrapper ( int * req )' )
    str.append ( '\n{\n    //  Get the tuple for the requested size\n' )
    str.append ( '    fftx::point_t<3> rsz;\n' )
    str.append ( '    rsz[0] = req[0];  rsz[1] = req[1];  rsz[2] = req[2];\n' )
    str.append ( f'    transformTuple_t *wp = {script.file_stem}{script.decor_platform}Tuple ( rsz );' )
    str.append ( '\n    if ( wp == NULL )\n' )
    str.append ( '        //  Requested size not found -- just return\n' )
    str.append ( '        return;\n\n' )

    if script.args.platform == 'CUDA' or script.args.platform == 'HIP' or script.args.platform == 'SYCL':
        str.append ( '    ' + _memfree + ' ( dev_out );\n' )
        str.append ( '    ' + _memfree + ' ( dev_sym );\n' )
        str.append ( '    ' + _memfree + ' ( dev_in  );\n\n' )

    str.append ( '    //  Tear down / cleanup\n' )
    str.append ( '    ( * wp->destroyfp ) ();\n' )
    if script.args.platform == 'CUDA' or script.args.platform == 'HIP' or script.args.platform == 'SYCL':
        str.append ( '    ' + _errchk + '\n\n' )

    str.append ( '    return;\n}\n\n}\n' )
        
    return str.get();


def create_metadata ( script, metadata ):
    "Create a compileable module to be added to the library that contains the metadata for the library"

    str = CodeBuilder ( '//  Copyright (c) 2018-2023, Carnegie Mellon University\n' )
    str.append ( '//  See LICENSE for details\n\n' )

    str.append ( '#include <stdio.h>\n' )
    str.append ( '#include <stdlib.h>\n' )
    str.append ( '#include <string.h>\n\n' )

    ##  remove last 3 chars of metadata (they are an unwanted ',\\n')
    str.append ( metadata )
    str.erase_last ( 3 )
    str.append ( '    ]\\\n}\\\n' + SP_METADATA_END + '";\n\n' )

    str.append ( '//  The metadata table is compiled into the library (and thus readable by scanning file,\n' )
    str.append ( '//  without having to load the library).\n' )
    str.append ( '//  Add a simple function to get the metadata (for debug purposes).\n\n' )

    str.append ( f'char * {script.file_stem}{script.decor_platform}GetMetaData ()' + '\n{\n' )
    str.append ( f'    char * wp = (char *) malloc ( strlen ( {script.file_stem}MetaData ) + 1 );' + '\n' )
    str.append ( '    if ( wp != NULL )\n' )
    str.append ( f'        strcpy ( wp, {script.file_stem}MetaData );' + '\n\n' )
    str.append ( '    return wp;\n' )
    str.append ( '}\n\n' )

    return str.get();


def cmake_library ( script ):
    """Build the CMakeLists.txt file for the generated library"""
    
    str = CodeBuilder ( '##\n## Copyright (c) 2018-2023, Carnegie Mellon University\n' )
    str.append ( '## All rights reserved.\n##\n## See LICENSE file for full information\n##\n\n' )

    str.append ( 'cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )\n\n' )

    moddecor = re.sub ( '_$', '', script.decor_platform )                 ## remove trailing underscore
    str.append ( f'set ( _lib_root {script.file_stem}{moddecor} )' )
    str.append ( '\nset ( _lib_name ${_lib_root} )\n' )
    str.append ( 'set ( _lib_name ${_lib_root} PARENT_SCOPE )\n\n' )

    if script.args.platform == 'CUDA':
        str.append ( 'set ( CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 )\n\n' )

    str.append ( 'include ( SourceList.cmake )\n' )
    str.append ( 'list    ( APPEND _source_files ${_lib_root}_libentry' + script.file_suffix + ' )\n' )
    str.append ( 'list    ( APPEND _source_files ${_lib_root}_metadata' + script.file_suffix + ' )\n' )
    str.append ( 'set ( _incl_files ${_lib_root}_public.h )\n\n' )

    str.append ( 'add_library                ( ${_lib_name} SHARED ${_source_files} )\n' )
    if script.args.platform == 'CUDA':
        str.append ( 'target_compile_options     ( ${_lib_name} PRIVATE ${CUDA_COMPILE_FLAGS} ${GPU_COMPILE_DEFNS} )\n' )
        str.append ( 'target_compile_options     ( ${_lib_name} PRIVATE ${ADDL_COMPILE_FLAGS} )\n' )
        str.append ( 'set_property        ( TARGET ${_lib_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )\n\n' )
    elif script.args.platform == 'HIP':
        str.append ( 'target_compile_options     ( ${_lib_name} PRIVATE ${HIP_COMPILE_FLAGS} ${ADDL_COMPILE_FLAGS} )\n\n' )
    elif script.args.platform == 'SYCL':
        str.append ( 'target_compile_options     ( ${_lib_name} PRIVATE ${SYCL_COMPILE_FLAGS} ${ADDL_COMPILE_FLAGS} )\n\n' )
    elif script.args.platform == 'CPU':
        str.append ( 'target_compile_options     ( ${_lib_name} PRIVATE ${ADDL_COMPILE_FLAGS} )\n\n' )

    str.append ( 'if ( WIN32 )\n' )
    str.append ( '    set_property    ( TARGET ${_lib_name} PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON )\n' )
    str.append ( 'endif ()\n\n' )

    str.append ( 'install ( TARGETS\n' )
    str.append ( '          ${_lib_name}\n' )
    str.append ( '          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )\n\n' )

    str.append ( 'install ( FILES ${_incl_files}\n' )
    str.append ( '          DESTINATION ${CMAKE_INSTALL_PREFIX}/include )\n\n' )

    return str.get();


def parse_platform ( value ):
    "Return the specified platform, use case-insensitive comparisons"
    
    if re.match ( 'hip', value, re.IGNORECASE ):
        return 'HIP'
    if re.match ( 'cpu', value, re.IGNORECASE ):
        return 'CPU'
    if re.match ( 'cuda', value, re.IGNORECASE ):
        return 'CUDA'
    if re.match ( 'sycl', value, re.IGNORECASE ):
        return 'SYCL'
    return 'HIP'

def get_spiralpy_xform ( name ):
    "Get the SpiralPy transform type corresponding to the working transform"

    if name == 'rconv':
        sp_type = SP_TRANSFORM_MDRCONV
    else:
        sp_type = name.upper()

    return sp_type

def parse_args ():
    "Parse and validate the command line arguments"

    parser = argparse.ArgumentParser (
        description = 'Build FFTX library code with Spiral and transform specifications',
        usage = '%(prog)s -t TRANSFORM -s SIZES_FILE -p {CPU,CUDA,HIP,SYCL} [-i] [-m SIZES_MASTER]'
    )
    ##  Required arguments: <transform> <sizes-file> <platform>
    reqd_group = parser.add_argument_group ( 'required arguments' )
    reqd_group.add_argument ( '-t', '--transform', type=str, required=True,
                              help='transform to use use for building the library' )
    reqd_group.add_argument ( '-s', '--sizes_file', type=str, required=True,
                              help='filename containing the sizes to build' )
    reqd_group.add_argument ( '-p', '--platform',  type=parse_platform, nargs='?', default='HIP',
                              choices=['CPU', 'CUDA', 'HIP', 'SYCL'], required=True,
                              help='Platform: one of {CUDA | HIP | SYCL | CPU}' )
    
    ##  Optional arguments: <direction> <sizes-master>
    parser.add_argument ( '-i', '--inverse', action='store_true',
                          help='False [default], run forward transform; when specified run Inverse transform' )
    parser.add_argument ( '-m', '--sizes_master', type=str,
                          help='Master sizes filename; Regenerate headers & API files [uses existing code files] for the library' )
    args = parser.parse_args()
    return args

def setup_script_options ( args ):
    "Setup the options for the script generation process"
    
    ##  Add a dictionary to hold platform to file extension mapping
    plat_to_file_suffix = {
        'CPU':  '.cpp',
        'CUDA': '.cu',
        'HIP':  '.cpp',
        'SYCL': '.cpp'
    }

    ##  Instantiate 'script', and add the parameters needed for generating source code
    script = Script ( args )
    script.file_suffix = plat_to_file_suffix.get ( args.platform, '.cpp' )            ## default to '.cpp'
    script.regen = True if args.sizes_master is not None else False
    
    ##  Print the options selected
    ##  print ( f'Generate files for:\nTransform:\t{args.transform}\nSizes file:\t{args.sizes_file}\nPlatform:\t{args.platform}' )
    ##  dirt = 'Inverse' if args.inverse else 'Forward'
    ##  print ( f'File suffix:\t{script.file_suffix}\nDirection:\t{dirt}\nRegen library:\t{script.regen}', flush = True )
    ##  if script.regen:
    ##      print ( f'Master sizes:\t{args.sizes_master}', flush = True )

    if args.transform.startswith ( 'fftx_' ):
        script.xform_name = args.transform.split('_')[1]
        script.sp_type = get_spiralpy_xform ( script.xform_name )
        script.file_stem = args.transform + '_'
        script.orig_file_stem = script.file_stem
        script.decor_platform = 'cpu_' if args.platform == 'CPU' else 'gpu_'
        if args.inverse:
            script.file_stem = 'fftx_i' + script.xform_name + '_'
    else:
        ##  transform name not recognized, print message & quit
        print ( f'Transform name {args.transform} is not valid, must start with \'fftx_\'...exiting' )
        sys.exit(-1)

    ##  Create the library sources directory (if it doesn't exist)
    script.srcs_dir  = 'lib_' + script.file_stem + script.decor_platform + 'srcs'
    isdir = os.path.isdir ( script.srcs_dir )
    if not isdir:
        os.mkdir ( script.srcs_dir )

    return script

def build_code_files ( script ):
    "Process the sizes file and build the source code (run Spiral), then buils API files"

    cmake_srcs = open ( os.path.join ( script.srcs_dir, 'SourceList.cmake' ), 'w' )
    cmake_srcs.write ( 'set ( _source_files ${_source_files} \n' )

    ##  Setup basic strings with the code fragments (added to as each transform spec processed)

    extern_decls = CodeBuilder ( '' )           ##  Initially empty
    all_cubes    = CodeBuilder ( f'static fftx::point_t<3> AllSizes3_{script.args.platform}[] = ' + '{\n' )
    tuple_funcs  = CodeBuilder ( f'static transformTuple_t {script.file_stem}{script.args.platform}_Tuples[] = ' + '{\n' )

    metadata     = CodeBuilder ( f'static char {script.file_stem}MetaData[] = ' + '\"' )
    metadata.append ( SP_METADATA_START + '\\\n{\\\n' )
    metadata.append ( '  \\"' + SP_KEY_TRANSFORMTYPES + '\\": [ \\"' + script.sp_type + '\\" ],\\\n' )
    metadata.append ( '  \\"' + SP_KEY_TRANSFORMS + '\\": [ \\\n' )
    
    ##  process the sizes file (script.args.sizes_file) and create a source code file (invoke Spiral) for each size
    with open ( script.args.sizes_file, 'r' ) as file:
        currpid = os.getpid()
        myscrf  = 'myscript_' + str ( currpid ) + '.g'
        testsf  = 'testscript_' + str ( currpid ) + '.g'

        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            testscript = open ( testsf, 'w' )
            testscript.write ( line + '\n' )
            testscript.write ( 'libdir := "' + script.srcs_dir + '"; \n' )
            testscript.write ( 'file_suffix := "' + script.file_suffix + '"; \n' )
            _fwd = 'false' if script.args.inverse else 'true'
            testscript.write ( 'fwd := ' + _fwd + '; \n' )
            testscript.write ( 'codefor := "' + script.args.platform + '"; \n' )
            ##  testscript.write ( 'createJIT := true;\n' )
            testscript.close()

            line = re.sub ( ' ', '', line )                 ## suppress white space
            pattern = r'szcube:=\[(\d+),(\d+),(\d+)\];'
            m = re.match ( pattern, line)
            if not m:
                print ( f'Invalid line format: {line}', flush = True )
                continue

            dims = [int(m.group(i)) for i in range(1, 4)]

            if re.match ( 'rconv', script.xform_name ) and script.args.platform == 'CPU' and dims[0] > 260:
                continue

            ##  Assumption: gap file is named {script.orig_file_stem}-frame.g
            ##  Generate the SPIRAL script: cat testscript_$pid.g & {transform}-frame.g
            script.frame_file = re.sub ( '_$', '', script.orig_file_stem ) + '-frame.g'
            
            _spiralhome = os.environ.get('SPIRAL_HOME')
            _catfils = os.path.join ( _spiralhome, 'gap', 'bin', 'catfiles.py' )
            cmdstr = f'"{sys.executable}" "{_catfils}" "{myscrf}" "{testsf}" "{script.frame_file}"'
            result = subprocess.run ( cmdstr, shell=True, check=True )
            res = result.returncode

            ##  Generate the code by running SPIRAL
            if sys.platform == 'win32':
                cmdstr = os.path.join ( _spiralhome, 'bin', 'spiral.bat' ) + ' < ' + myscrf
            else:
                cmdstr = os.path.join ( _spiralhome, 'bin', 'spiral' ) + ' < ' + myscrf

            script.func_stem = f'{script.file_stem}{dims[0]}x{dims[1]}x{dims[2]}_{script.args.platform}'
            script.file_name = script.func_stem + script.file_suffix
            script.src_file_path = script.srcs_dir + '/' + script.file_name
            failure_written = False
            if not script.regen:
                ##  Not regenerating library, run Spiral to generate the code
                try:
                    result = subprocess.run ( cmdstr, shell=True, check=True )
                    res = result.returncode
                except:
                    ##  Spiral exited with an error (non-zero return code).  Log the failure.
                    ##  Failed to generate file -- note it in build-lib-code-failures.txt
                    print ( 'Spiral code generation failed, error logged to build-lib-code-failures.txt', flush = True )
                    bldf = open ( 'build-lib-code-failures.txt', 'a' )
                    bldf.write  ( 'Failed to generate:   ' + script.src_file_path + '\n' )
                    bldf.close  ()
                    failure_written = True
                    if os.path.exists ( script.src_file_path ):
                        os.remove ( script.src_file_path )

            else:
                ##  Just print a message and skip copde gen (test python process/logic)
                print ( 'Regenerate API & header files, use existing source file: ' + script.file_name, flush = True )

            ##  Add the file name to the list of sources, update declarations etc. if file exists
            if os.path.exists ( script.src_file_path ):
                cmake_srcs.write ( '    ' + script.file_name + '\n' )

                ##  Add the extern declarations and track func name for header file
                extern_decls.append ( 'extern "C" { extern void init_' + script.func_stem + '();  }\n' )
                extern_decls.append ( 'extern "C" { extern void destroy_' + script.func_stem + '();  }\n' )

                extern_decls.append ( 'extern "C" { extern void ' + script.func_stem )
                if script.xform_name == 'psatd':
                    extern_decls.append ( '( double **output, double **input, double **sym );  }\n\n' )
                else:
                    extern_decls.append ( '( double *output, double *input, double *sym );  }\n\n' )

                all_cubes.append ( '    { ' + f'{dims[0]}, {dims[1]}, {dims[2]}' + ' },\n' )
                tuple_funcs.append ( '    { ' + f'init_{script.func_stem}, destroy_{script.func_stem}, ' )
                tuple_funcs.append ( script.func_stem + ' },\n' )
                metadata.append ( '    {  \\"' + SP_KEY_DIMENSIONS + '\\": [ ' f'{dims[0]}, {dims[1]}, {dims[2]}' + ' ],\\\n' )
                metadata.append ( '       \\"' + SP_KEY_DIRECTION + '\\": \\"' )
                if _fwd == 'true':
                    metadata.append ( SP_STR_FORWARD )
                else:
                    metadata.append ( SP_STR_INVERSE )
                metadata.append ( '\\",\\\n' )
                metadata.append ( '       \\"' + SP_KEY_NAMES + '\\": {\\\n' )
                metadata.append ( '         \\"' + SP_KEY_DESTROY + '\\": \\"destroy_' + script.func_stem + '\\",\\\n' )
                metadata.append ( '         \\"' + SP_KEY_EXEC + '\\": \\"' + script.func_stem + '\\",\\\n' )
                metadata.append ( '         \\"' + SP_KEY_INIT + '\\": \\"init_' + script.func_stem + '\\" },\\\n' )
                ##  For now we're only doing C ordering...
                metadata.append ( '       \\"' + SP_KEY_ORDER + '\\": \\"' + SP_STR_C + '\\",\\\n' )
                metadata.append ( '       \\"' + SP_KEY_PLATFORM + '\\": \\"' + script.args.platform + '\\",\\\n' )
                ##  For now all libs generated are double precision -- maybe look at this in future
                metadata.append ( '       \\"' + SP_KEY_PRECISION + '\\": \\"' + SP_STR_DOUBLE + '\\",\\\n' )
                metadata.append ( '       \\"' + SP_KEY_TRANSFORMTYPE + '\\": \\"' + script.sp_type + '\\"\\\n' )
                metadata.append ( '    },\\\n' )

            else:
                ##  File was not successfully created
                if not failure_written:
                    ##  Failed to generate file -- note it in build-lib-code-failures.txt
                    bldf = open ( 'build-lib-code-failures.txt', 'a' )
                    bldf.write  ( 'Failed to generate:   ' + script.src_file_path + '\n' )
                    bldf.close  ()

        ##  All cube sizes processed: close list of sources, create header file
        cmake_srcs.write ( ')\n' )
        cmake_srcs.close()

        ##  Create the declarations header file.  Decls files have either 'cpu' or 'gpu' (when
        ##  target is CUDA | HIP | SYCL) in the name.  Write the file body, the extern
        ##  declarations, the tuple functions table, and the list of sizes to the file in turn.
    
        _hfil = os.path.join ( script.srcs_dir, script.file_stem + script.decor_platform + 'decls.h' )
        _header_fil = open ( _hfil, 'w' )
        _filebody = start_header_file ( 'LIB_', script )
        _header_fil.write ( _filebody )
        _header_fil.write ( extern_decls.get() )
        tuple_funcs.append ( '    { NULL, NULL, NULL }\n};\n\n' )
        _header_fil.write ( tuple_funcs.get() )
        all_cubes.append ( '    { 0, 0, 0 }\n};\n\n' )
        _header_fil.write ( all_cubes.get() )
        _header_fil.write ( '#endif\n\n' )
        _header_fil.close ()

        ##  Create the public header file

        _hfil = os.path.join ( script.srcs_dir, script.file_stem + script.decor_platform + 'public.h' )
        _header_fil = open ( _hfil, 'w' )
        _filebody = start_header_file ( 'PUBLIC_',  script )
        _filebody = _filebody + body_public_header ( script )
        _header_fil.write ( _filebody )
        _header_fil.close ()

        ##  Create the {script.args.platform} library API file.  This is the public API for the target

        _hfil = os.path.join ( script.srcs_dir, script.file_stem + script.decor_platform + 'libentry' + script.file_suffix )
        _api_file = open ( _hfil, 'w' )
        _filebody = library_api ( script )
        if script.args.platform != 'SYCL':
            _filebody = _filebody + python_cuda_api ( script )
        _api_file.write ( _filebody )
        _api_file.close ()

        ##  Create the metadata file.

        _hfil = os.path.join ( script.srcs_dir, script.file_stem + script.decor_platform + 'metadata' + script.file_suffix )
        _api_file = open ( _hfil, 'w' )
        _filebody = create_metadata ( script, metadata.get() )
        _api_file.write ( _filebody )
        _api_file.close ()

        ##  Create the CMakeLists.txt file
        _hfil = os.path.join ( script.srcs_dir, 'CMakeLists.txt' )
        _cmake_file = open ( _hfil, 'w' )
        _filebody = cmake_library ( script )
        _cmake_file.write ( _filebody )
        _cmake_file.close ()

        if os.path.exists ( myscrf ):
            os.remove ( myscrf )
        if os.path.exists ( testsf ):
            os.remove ( testsf )

    return


def main():
    args = parse_args()
    script = setup_script_options ( args )

    ##  We want to potentially run the build_code_files function twice
    ##  Pass 1: build all files specified in the sizes file
    ##  Pass 2: optional, depends on a master sizes file spcified
    pass2 = False
    if script.regen:
        pass2 = script.regen
        script.regen = False

    build_code_files ( script )

    if pass2:
        script.regen = pass2
        script.args.sizes_file = script.args.sizes_master       ##  Force processing on master file
        build_code_files ( script )


if __name__ == '__main__':
    main()


