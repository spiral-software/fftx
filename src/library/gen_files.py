#! python

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

##  This script reads a file, cube-sizes.txt, that contains several cube size
##  specifications for the 3D DFT.  This script will:
##      Generate a list of source file names for CMake to build
##      Create the source files (by running Spiral), writing them
##      to directory lib_<stem>_srcs
##      Create the prototype definitions in a private library include file: <stem>_decls.h
##      Create the public header file for the library: <stem>_public.h
##  Compiling and library is handled by CMake.

import sys
import subprocess
import os, stat
import re
import shutil

##  file stem can be an argument speciying library to build
if len ( sys.argv ) < 2:
    ##  No library name stem provided, default to mddft3d_
    _file_stem = 'mddft3d_'
else:
    ##  Use given argument as the library stem name
    _file_stem = sys.argv[1]
    if not re.match ( '_$', _file_stem ):                ## append an underscore if one is not present
        _file_stem = _file_stem + '_'

_xform_name = _file_stem
_xform_pref = ''

if re.match ( '^.*_.*_', _file_stem ):
    _dims = re.split ( '_', _file_stem )
    _xform_pref = _dims[0]
    _xform_name = _dims[1]
    _xform_root = _dims[1]
    _xform_name = _xform_name + '_'
    _xform_pref = _xform_pref + '_'

_orig_file_stem = _file_stem

##  Code to build -- Hip or CUDA (default) governs file suffix etc.
_code_type = 'CUDA'
_file_suffix = '.cu'

if len ( sys.argv ) >= 3:
    ##  Code type specified
    _code_type = sys.argv[2]
    if re.match ( 'cuda', _code_type, re.IGNORECASE ):
        ##  CUDA selected
        _code_type = 'CUDA'
        _file_suffix = '.cu'

    if re.match ( 'hip', _code_type, re.IGNORECASE ):
        ##  HIP selected
        _code_type = 'HIP'
        _file_suffix = '.cpp'

    if re.match ( 'cpu', _code_type, re.IGNORECASE ):
        ##  CPU code gen selected
        _code_type = 'CPU'
        _file_suffix = '.cpp'

##  If the transform can be forward or inverse accept an argument to specify
_fwd = 'true'             ## default to true or forward

if len ( sys.argv ) >= 4:
    ##  Forward/Inverse parameter specified -- anything except false ==> true
    if re.match ( 'false', sys.argv[3], re.IGNORECASE ):
        _fwd = 'false'
        _file_stem =  _xform_pref + 'i' + _xform_name
        _xform_root = 'i' + _xform_root
        ##  print ( 'File stem = ' + _file_stem )

##  Create the library sources directory (if it doesn't exist)

_srcs_dir  = 'lib_' + _file_stem + 'srcs'
isdir = os.path.isdir ( _srcs_dir )
if not isdir:
    os.mkdir ( _srcs_dir )

_cmake_srcs = open ( _srcs_dir + '/SourceList' + _code_type + '.cmake', 'w' )
_cmake_srcs.write ( 'set ( _source_files ${_source_files} \n' )

##  Build a header file for the library with the declarations and tables to
##  manage the entry points in the library


def start_header_file ( type, codefor ):
    "Sets up the common stuff for header files"

    if codefor != '':
        codefor = codefor + '_'
        
    _str = '#ifndef ' + _file_stem + type + codefor + 'HEADER_INCLUDED\n'
    _str = _str + '#define ' + _file_stem + type + codefor + 'HEADER_INCLUDED\n\n'
    _str = _str + '//  Copyright (c) 2018-2021, Carnegie Mellon University\n'
    _str = _str + '//  See LICENSE for details\n\n'

    if codefor != '':
        _str = _str + '#include "fftx3.hpp"\n\n'

        _str = _str + '#ifndef INITTRANSFORMFUNC\n'
        _str = _str + '#define INITTRANSFORMFUNC\n'
        _str = _str + 'typedef void ( * initTransformFunc ) ( void );\n'
        _str = _str + '#endif\n\n'

        _str = _str + '#ifndef DESTROYTRANSFORMFUNC\n'
        _str = _str + '#define DESTROYTRANSFORMFUNC\n'
        _str = _str + 'typedef void ( * destroyTransformFunc ) ( void );\n'
        _str = _str + '#endif\n\n'

        _str = _str + '#ifndef RUNTRANSFORMFUNC\n'
        _str = _str + '#define RUNTRANSFORMFUNC\n'
        ##  TODO: Allow optional 3rd arg for symbol
        ##  _str = _str + 'typedef void ( * runTransformFunc ) ( double *output, double *input );\n'
        _str = _str + 'typedef void ( * runTransformFunc ) ( double *output, double *input, double *sym );\n'
        _str = _str + '#endif\n\n'

        _str = _str + '#ifndef TRANSFORMTUPLE_T\n'
        _str = _str + '#define TRANSFORMTUPLE_T\n'
        _str = _str + 'typedef struct transformTuple {\n'
        _str = _str + '    initTransformFunc    initfp;\n'
        _str = _str + '    destroyTransformFunc destroyfp;\n'
        _str = _str + '    runTransformFunc     runfp;\n'
        _str = _str + '} transformTuple_t;\n'
        _str = _str + '#endif\n\n'

    else:
        _str = _str + '#include "' + _file_stem + 'CPU_public.h"\n'
        if _code_type == 'CUDA' or _code_type == 'HIP':
            _str = _str + '#include "' + _file_stem + _code_type + '_public.h"\n\n'

        _str = _str + '#ifndef LIB_MODE_CPU\n'
        _str = _str + '#define LIB_MODE_CPU 0                   //  run CPU Spiral code\n'
        _str = _str + '#define LIB_MODE_CUDA 1                  //  run NVIDIA GPU Spiral code\n'
        _str = _str + '#define LIB_MODE_HIP 2                   //  run AMD HIP Spiral code \n'
        _str = _str + '#endif\n\n'

        _str = _str + 'static int ' + _file_stem + 'LibraryMode = LIB_MODE_' + _code_type + ';\n\n'
        _str = _str + 'extern "C" {\n'
        _str = _str + '//  Get the library code mode\n'
        _str = _str + 'int ' + _file_stem + 'GetLibraryMode ();\n\n'
        _str = _str + '//  Set the library code mode -- specify which code to run = { CPU | CUDA | HIP }\n'
        _str = _str + 'void ' + _file_stem + 'SetLibraryMode ( int );\n}\n\n'

    return _str;


def body_public_header ( codefor ):
    "Add the body details for the public header file"
    if codefor != '':
        codefor = codefor + '_'

    _str =        '//  Query the list of sizes available from the library; returns a pointer to an\n'
    _str = _str + '//  array of size, each element is a struct of type fftx::point_t<3> specifying the X,\n'
    _str = _str + '//  Y, and Z dimensions\n\n'

    _str = _str + 'fftx::point_t<3> * ' + _file_stem + codefor + 'QuerySizes ();\n\n'

    _str = _str + '//  Run an ' + _file_stem + ' transform once: run the init functions, run the transform,\n'
    _str = _str + '//  and finally tear down by calling the destroy function.  Accepts fftx::point_t<3>\n'
    _str = _str + '//  specifying size, and pointers to the output (returned) data and the input\n'
    _str = _str + '//  data.\n\n'

    ##  TODO: Allow optional 3rd arg for symbol
    ##  _str = _str + 'void ' + _file_stem + 'Run ( fftx::point_t<3> req, double * output, double * input );\n\n'
    _str = _str + 'void ' + _file_stem + codefor + 'Run ( fftx::point_t<3> req, double * output, double * input, double * sym );\n\n'

    _str = _str + '//  Get a transform tuple -- a set of pointers to the init, destroy, and run\n'
    _str = _str + '//  functions for a specific size ' + _file_stem + ' transform.  Using this information the\n'
    _str = _str + '//  user may call the init function to setup for the transform, then run the\n'
    _str = _str + '//  transform repeatedly, and finally tear down (using destroy function).\n\n'

    _str = _str + 'transformTuple_t * ' + _file_stem + codefor + 'Tuple ( fftx::point_t<3> req );\n\n'

    _str = _str + '//  Wrapper functions to allow python to call CUDA/HIP GPU code.\n\n'
    _str = _str + 'extern "C" {\n\n'
    _str = _str + 'int  ' + _file_stem + codefor + 'python_init_wrapper ( int * req );\n'
    _str = _str + 'void ' + _file_stem + codefor + 'python_run_wrapper ( int * req, double * output, double * input, double * sym );\n'
    _str = _str + 'void ' + _file_stem + codefor + 'python_destroy_wrapper ( int * req );\n\n'
    _str = _str + '}\n\n#endif\n\n'

    return _str;


def library_api ( mkvers, type ):
    "Sets up the API file(s) for the library: one generic file and up to two arch specific versions"
    if type == '':
        codefor = ''
    else:
        codefor = type + '_'

    _str =        '//  Copyright (c) 2018-2021, Carnegie Mellon University\n'
    _str = _str + '//  See LICENSE for details\n\n'

    _str = _str + '#include <stdio.h>\n'
    _str = _str + '#include <stdlib.h>\n'
    _str = _str + '#include <string.h>\n'
    _str = _str + '#include "' + _file_stem + 'decls.h"\n'
    _str = _str + '#include "' + _file_stem + 'public.h"\n\n'

    if mkvers:
        if type == 'CUDA':
            _str = _str + '#include <helper_cuda.h>\n\n'
        elif type == 'HIP':
            _str = _str + '#include <hip/hip_runtime.h>\n\n'
            _str = _str + '#define checkLastHipError(str)   { hipError_t err = hipGetLastError();   if (err != hipSuccess) {  printf("%s: %s\\n", (str), hipGetErrorString(err) );  exit(-1); } }\n\n'

        _str = _str + '//  Query the list of sizes available from the library; returns a pointer to an\n'
        _str = _str + '//  array of size <N+1>, each element is a struct of type fftx::point_t<3> specifying the X,\n'
        _str = _str + '//  Y, and Z dimensions of a transform in the library.  <N> is the number of sizes defined;\n'
        _str = _str + '//  the last entry in the returned list has all dimensions equal 0.\n\n'

        _str = _str + 'fftx::point_t<3> * ' + _file_stem + codefor + 'QuerySizes ()\n{\n'
        _str = _str + '    fftx::point_t<3> *wp = (fftx::point_t<3> *) malloc ( sizeof ( AllSizes3_' + type + ' ) );\n'
        _str = _str + '    if ( wp != NULL)\n'
        _str = _str + '        memcpy ( (void *) wp, (const void *) AllSizes3_' + type + ', sizeof ( AllSizes3_' + type + ' ) );\n\n'
        _str = _str + '    return wp;\n'
        _str = _str + '}\n\n'

        _str = _str + '//  Get a transform tuple -- a set of pointers to the init, destroy, and run\n'
        _str = _str + '//  functions for a specific size ' + _file_stem + ' transform.  Using this information the\n'
        _str = _str + '//  user may call the init function to setup for the transform, then run the\n'
        _str = _str + '//  transform repeatedly, and finally tear down (using destroy function).\n'
        _str = _str + '//  Returns NULL if requested size is not found.\n\n'

        _str = _str + 'transformTuple_t * ' + _file_stem + codefor + 'Tuple ( fftx::point_t<3> req )\n'
        _str = _str + '{\n'
        _str = _str + '    int indx;\n'
        _str = _str + '    int numentries = sizeof ( AllSizes3_' + type + ' ) / sizeof ( fftx::point_t<3> ) - 1;    // last entry is { 0, 0, 0 }\n'
        _str = _str + '    transformTuple_t *wp = NULL;\n\n'

        _str = _str + '    for ( indx = 0; indx < numentries; indx++ ) {\n'
        _str = _str + '        if ( req[0] == AllSizes3_' + type + '[indx][0] &&\n'
        _str = _str + '             req[1] == AllSizes3_' + type + '[indx][1] &&\n'
        _str = _str + '             req[2] == AllSizes3_' + type + '[indx][2] ) {\n'
        _str = _str + '            // found a match\n'
        _str = _str + '            wp = (transformTuple_t *) malloc ( sizeof ( transformTuple_t ) );\n'
        _str = _str + '            if ( wp != NULL) {\n'
        _str = _str + '                *wp = ' + _file_stem + codefor + 'Tuples[indx];\n'
        _str = _str + '            }\n'
        _str = _str + '            break;\n'
        _str = _str + '        }\n'
        _str = _str + '    }\n\n'

        _str = _str + '    return wp;\n'
        _str = _str + '}\n\n'

        _str = _str + '//  Run an ' + _file_stem + ' transform once: run the init functions, run the transform,\n'
        _str = _str + '//  and finally tear down by calling the destroy function.  Accepts fftx::point_t<3>\n'
        _str = _str + '//  specifying size, and pointers to the output (returned) data and the input\n'
        _str = _str + '//  data.\n\n'

        ##  TODO: Allow optional 3rd arg for symbol
        ##  _str = _str + 'void ' + _file_stem + code_type + '_Run ( fftx::point_t<3> req, double * output, double * input )\n'
        _str = _str + 'void ' + _file_stem + type + '_Run ( fftx::point_t<3> req, double * output, double * input, double * sym )\n'
        _str = _str + '{\n'
        _str = _str + '    transformTuple_t *wp = ' + _file_stem + type + '_Tuple ( req );\n'
        _str = _str + '    if ( wp == NULL )\n'
        _str = _str + '        //  Requested size not found -- just return\n'
        _str = _str + '        return;\n\n'

        _str = _str + '    //  Call the init function\n'
        _str = _str + '    ( * wp->initfp )();\n'
        _str = _str + '    //  checkCudaErrors ( cudaGetLastError () );\n\n'

        ##  TODO: Allow optional 3rd arg for symbol
        ##  _str = _str + '    ( * wp->runfp ) ( output, input );\n'
        _str = _str + '    ( * wp->runfp ) ( output, input, sym );\n'
        _str = _str + '    //  checkCudaErrors ( cudaGetLastError () );\n\n'

        _str = _str + '    //  Tear down / cleanup\n'
        _str = _str + '    ( * wp->destroyfp ) ();\n'
        _str = _str + '    //  checkCudaErrors ( cudaGetLastError () );\n\n'

        _str = _str + '    return;\n'
        _str = _str + '}\n\n'

    else:
        _str = _str + 'extern "C" {\n'
        _str = _str + '//  Get the library code mode\n\n'
        _str = _str + 'int ' + _file_stem + 'GetLibraryMode ()\n{\n'
        _str = _str + '    return ' + _file_stem + 'LibraryMode;\n}\n\n'

        _str = _str + '//  Set the library code mode -- specify which code to run = { CPU | CUDA | HIP }\n\n'
        _str = _str + 'void ' + _file_stem + 'SetLibraryMode ( int reqmode )\n{\n'
        _str = _str + '    if ( reqmode < LIB_MODE_CPU || reqmode > LIB_MODE_HIP ) return;      //  ignore invalid requests\n'
        _str = _str + '    ' + _file_stem + 'LibraryMode = reqmode;\n'
        _str = _str + '    return;\n}\n\n}\n\n'

        _str = _str + '//  Call the specific version of each public API function based on library mode...\n\n'
        _str = _str + '//  Get the list of sizes defined in the library.\n\n'
        _str = _str + 'fftx::point_t<3> * ' + _file_stem + 'QuerySizes ()\n{\n'
        _str = _str + '    fftx::point_t<3> *wp = NULL;\n'
        _str = _str + '    if ( ' + _file_stem + 'LibraryMode == LIB_MODE_CPU )\n'
        _str = _str + '        wp = ' + _file_stem + 'CPU_QuerySizes();\n\n'

        if type == 'CUDA' or type == 'HIP':
            _str = _str + '    if ( ' + _file_stem + 'LibraryMode == LIB_MODE_' + type + ' )\n'
            _str = _str + '        wp = ' + _file_stem + codefor + 'QuerySizes();\n\n'

        _str = _str + '    return wp;\n'
        _str = _str + '}\n\n'

        _str = _str + '//  Get a transform tuple.\n\n'
        _str = _str + 'transformTuple_t * ' + _file_stem + 'Tuple ( fftx::point_t<3> req )\n{\n'
        _str = _str + '    transformTuple_t *wp = NULL;\n'
        _str = _str + '    if ( ' + _file_stem + 'LibraryMode == LIB_MODE_CPU )\n'
        _str = _str + '        wp = ' + _file_stem + 'CPU_Tuple( req );\n\n'

        if type == 'CUDA' or type == 'HIP':
            _str = _str + '    if ( ' + _file_stem + 'LibraryMode == LIB_MODE_' + type + ' )\n'
            _str = _str + '        wp = ' + _file_stem + codefor + 'Tuple( req );\n\n'

        _str = _str + '    return wp;\n'
        _str = _str + '}\n\n'

        _str = _str + '//  Run an ' + _file_stem + ' transform once.\n\n'
        _str = _str + 'void ' + _file_stem + 'Run ( fftx::point_t<3> req, double * output, double * input, double * sym )\n{\n'
        _str = _str + '    if ( ' + _file_stem + 'LibraryMode == LIB_MODE_CPU )\n'
        _str = _str + '        ' + _file_stem + 'CPU_Run ( req, output, input, sym );\n\n'

        if type == 'CUDA' or type == 'HIP':
            _str = _str + '    if ( ' + _file_stem + 'LibraryMode == LIB_MODE_' + type + ' )\n'
            _str = _str + '        ' + _file_stem + codefor + 'Run ( req, output, input, sym );\n\n'

        _str = _str + '    return;\n'
        _str = _str + '}\n\n'


    return _str;


def python_cuda_api ( mkvers, type, xfm ):
    "Sets up the python wrapper API to allow Python to call C++ CUDA or HIP code. \
     For CPU code no data management or marshalling is required"

    if type == '':
        codefor = ''
    else:
        codefor = type + '_'

    _str =        '//  Host-to-Device C/CUDA wrapper functions to permit Python to call the CUDA kernels.\n\n'
    _str = _str + 'extern "C" {\n\n'

    if mkvers:
        if type == 'CUDA' or type == 'HIP':
            _str = _str + 'static double *dev_in, *dev_out, *dev_sym;\n\n'

        _str = _str + 'int  ' + _file_stem + codefor + 'python_init_wrapper ( int * req )\n{\n'
        _str = _str + '    //  Get the tuple for the requested size\n'
        _str = _str + '    fftx::point_t<3> rsz;\n'
        _str = _str + '    rsz[0] = req[0];  rsz[1] = req[1];  rsz[2] = req[2];\n'
        _str = _str + '    transformTuple_t *wp = ' + _file_stem + codefor + 'Tuple ( rsz );\n'
        _str = _str + '    if ( wp == NULL )\n'
        _str = _str + '        //  Requested size not found -- return false\n'
        _str = _str + '        return 0;\n\n'

        if type == 'CUDA':
            _mmalloc = 'cudaMalloc'
            _errchk  = 'checkCudaErrors ( cudaGetLastError () );'
            _mmemcpy = 'cudaMemcpy'
            _cph2dev = 'cudaMemcpyHostToDevice'
            _cpdev2h = 'cudaMemcpyDeviceToHost'
            _memfree = 'cudaFree'
        elif type == 'HIP':
            _mmalloc = 'hipMalloc'
            _errchk  = 'checkLastHipError ( "Error: " );'
            _mmemcpy = 'hipMemcpy'
            _cph2dev = 'hipMemcpyHostToDevice'
            _cpdev2h = 'hipMemcpyDeviceToHost'
            _memfree = 'hipFree'

        if type == 'CUDA' or type == 'HIP':
            ##  Amount of data space to malloc depends on transform:
            ##     MDDFT/IMDDFT:   x * y * z * 2 doubles (for C2C, both input & output)
            ##     MDPRDFT:        x * y * z     doubles (for R2C, input)
            ##                     x * y * ((z/2) + 1) * 2 doubles (for R2C, output)
            ##     IMDPRDFT:       x * y * ((z/2) + 1) * 2 doubles (for C2R, input)
            ##                     x * y * z     doubles (for C2R, output)
            if xfm == 'mddft' or xfm == 'imddft':
                _str = _str + '    int ndoubin  = (int)(req[0] * req[1] * req[2] * 2);\n'
                _str = _str + '    int ndoubout = (int)(req[0] * req[1] * req[2] * 2);\n'
            elif xfm == 'mdprdft':
                _str = _str + '    int ndoubin  = (int)(req[0] * req[1] * req[2] );\n'
                _str = _str + '    int ndoubout = (int)(req[0] * req[1] * ((int)(req[2]/2) + 1) * 2);\n'
            elif xfm == 'imdprdft' or xfm == 'rconv':
                _str = _str + '    int ndoubin  = (int)(req[0] * req[1] * ((int)(req[2]/2) + 1) * 2);\n'
                _str = _str + '    int ndoubout = (int)(req[0] * req[1] * req[2] );\n'

            _str = _str + '    if ( ndoubin  == 0 )\n        return 0;\n\n'
            _str = _str + '    ' + _mmalloc + ' ( &dev_in,  sizeof(double) * ndoubin  );\n'
            _str = _str + '    ' + _mmalloc + ' ( &dev_out, sizeof(double) * ndoubout );\n'
            _str = _str + '    ' + _mmalloc + ' ( &dev_sym, sizeof(double) * 1000 );\n'
            _str = _str + '    ' + _errchk +  '\n\n'

        _str = _str + '    //  Call the init function\n'
        _str = _str + '    ( * wp->initfp )();\n'
        if type == 'CUDA' or type == 'HIP':
            _str = _str + '    ' + _errchk +  '\n\n'

        _str = _str + '    return 1;\n}\n\n'

        _str = _str + 'void ' + _file_stem + codefor + 'python_run_wrapper ( int * req, double * output, double * input, double * sym )\n{\n'
        _str = _str + '    //  Get the tuple for the requested size\n'
        _str = _str + '    fftx::point_t<3> rsz;\n'
        _str = _str + '    rsz[0] = req[0];  rsz[1] = req[1];  rsz[2] = req[2];\n'
        _str = _str + '    transformTuple_t *wp = ' + _file_stem + codefor + 'Tuple ( rsz );\n'
        _str = _str + '    if ( wp == NULL )\n'
        _str = _str + '        //  Requested size not found -- just return\n'
        _str = _str + '        return;\n\n'

        if type == 'CUDA' or type == 'HIP':
            if xfm == 'mddft' or xfm == 'imddft':
                _str = _str + '    int ndoubin  = (int)(req[0] * req[1] * req[2] * 2);\n'
                _str = _str + '    int ndoubout = (int)(req[0] * req[1] * req[2] * 2);\n'
            elif xfm == 'mdprdft':
                _str = _str + '    int ndoubin  = (int)(req[0] * req[1] * req[2] );\n'
                _str = _str + '    int ndoubout = (int)(req[0] * req[1] * ((int)(req[2]/2) + 1) * 2);\n'
            elif xfm == 'imdprdft' or xfm == 'rconv':
                _str = _str + '    int ndoubin  = (int)(req[0] * req[1] * ((int)(req[2]/2) + 1) * 2);\n'
                _str = _str + '    int ndoubout = (int)(req[0] * req[1] * req[2] );\n'

            _str = _str + '    if ( ndoubin  == 0 )\n        return;\n\n'
            _str = _str + '    ' + _mmemcpy + ' ( dev_in, input, sizeof(double) * ndoubin, ' + _cph2dev + ' );\n\n'

        _str = _str + '    //  Call the run function\n'
        if type == 'CUDA' or type == 'HIP':
            _str = _str + '    ( * wp->runfp )( dev_out, dev_in, dev_sym );\n'
            _str = _str + '    ' + _errchk  + '\n\n'
            _str = _str + '    ' + _mmemcpy + ' ( output, dev_out, sizeof(double) * ndoubout, ' + _cpdev2h + ' );\n'
        else:
            _str = _str + '    ( * wp->runfp )( output, input, sym );\n'
        
        _str = _str + '    return;\n}\n\n'

        _str = _str + 'void ' + _file_stem + codefor + 'python_destroy_wrapper ( int * req )\n{\n'
        _str = _str + '    //  Get the tuple for the requested size\n'
        _str = _str + '    fftx::point_t<3> rsz;\n'
        _str = _str + '    rsz[0] = req[0];  rsz[1] = req[1];  rsz[2] = req[2];\n'
        _str = _str + '    transformTuple_t *wp = ' + _file_stem + codefor + 'Tuple ( rsz );\n'
        _str = _str + '    if ( wp == NULL )\n'
        _str = _str + '        //  Requested size not found -- just return\n'
        _str = _str + '        return;\n\n'

        if type == 'CUDA' or type == 'HIP':
            _str = _str + '    ' + _memfree + ' ( dev_out );\n'
            _str = _str + '    ' + _memfree + ' ( dev_sym );\n'
            _str = _str + '    ' + _memfree + ' ( dev_in  );\n\n'

        _str = _str + '    //  Tear down / cleanup\n'
        _str = _str + '    ( * wp->destroyfp ) ();\n'
        if type == 'CUDA' or type == 'HIP':
            _str = _str + '    ' + _errchk + '\n\n'

        _str = _str + '    return;\n}\n\n}\n'

    else:
        ##  Generic wrapper calls that choose the appropriate version base don library mode
        ##  ???_python_init_wrapper

        _str = _str + 'int  ' + _file_stem + 'python_init_wrapper ( int * req )\n{\n'
        _str = _str + '    int res = 0;\n'
        _str = _str + '    if ( ' + _file_stem + 'LibraryMode == LIB_MODE_CPU )\n'
        _str = _str + '        res = ' + _file_stem + 'CPU_python_init_wrapper ( req );\n\n'

        if type == 'CUDA' or type == 'HIP':
            _str = _str + '    if ( ' + _file_stem + 'LibraryMode == LIB_MODE_' + type + ' )\n'
            _str = _str + '        res = ' + _file_stem + codefor + 'python_init_wrapper ( req );\n\n'
        
        _str = _str + '    return res;\n'
        _str = _str + '}\n\n'

        ##  ???_python_run_wrapper

        _str = _str + 'void ' + _file_stem + 'python_run_wrapper ( int * req, double * output, double * input, double * sym )\n{\n'
        _str = _str + '    if ( ' + _file_stem + 'LibraryMode == LIB_MODE_CPU )\n'
        _str = _str + '        ' + _file_stem + 'CPU_python_run_wrapper ( req, output, input, sym );\n\n'

        if type == 'CUDA' or type == 'HIP':
            _str = _str + '    if ( ' + _file_stem + 'LibraryMode == LIB_MODE_' + type + ' )\n'
            _str = _str + '        ' + _file_stem + codefor + 'python_run_wrapper ( req, output, input, sym );\n\n'
        
        _str = _str + '    return;\n'
        _str = _str + '}\n\n'

        ##  ???_python_destroy_wrapper

        _str = _str + 'void ' + _file_stem + 'python_destroy_wrapper ( int * req )\n{\n'
        _str = _str + '    if ( ' + _file_stem + 'LibraryMode == LIB_MODE_CPU )\n'
        _str = _str + '        ' + _file_stem + 'CPU_python_destroy_wrapper ( req );\n\n'

        if type == 'CUDA' or type == 'HIP':
            _str = _str + '    if ( ' + _file_stem + 'LibraryMode == LIB_MODE_' + type + ' )\n'
            _str = _str + '        ' + _file_stem + codefor + 'python_destroy_wrapper ( req );\n\n'
        
        _str = _str + '    return;\n}\n\n}\n'
        
    return _str;


def cmake_library ( type ):
    _str =        '##\n## Copyright (c) 2018-2021, Carnegie Mellon University\n'
    _str = _str + '## All rights reserved.\n##\n## See LICENSE file for full information\n##\n\n'

    _str = _str + 'cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )\n\n'

    _modfs = re.sub ( '_$', '', _file_stem )                 ## remove trailing underscore
    _str = _str + 'set ( _lib_root ' + _modfs + ' )\n'
    _str = _str + 'set ( _lib_name ${_lib_root} )\n'
    _str = _str + 'set ( _lib_name ${_lib_root} PARENT_SCOPE )\n\n'

    if type == 'CUDA':
        _str = _str + 'set ( CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 )\n\n'

    _str = _str + 'include ( SourceListCPU.cmake )\n'
    if type == 'CUDA' or type == 'HIP':
        _str = _str + 'include ( SourceList' + type + '.cmake )\n'

    _str = _str + 'list    ( APPEND _source_files ${_lib_root}_CPU_libentry.cpp' + ' )\n'
    _str = _str + 'list    ( APPEND _source_files ${_lib_root}_libentry' + _file_suffix + ' )\n'
    if type == 'CUDA' or type == 'HIP':
        _str = _str + 'list    ( APPEND _source_files ${_lib_root}_' + type + '_libentry' + _file_suffix + ' )\n\n'
    ##  _str = _str + 'message ( STATUS "Source file: ${_source_files}" )\n\n'

    _str = _str + 'set ( _incl_files ${_lib_root}_public.h ${_lib_root}_CPU_public.h )\n'
    if type == 'CUDA' or type == 'HIP':
        _str = _str + 'list    ( APPEND _incl_files ${_lib_root}_' + type + '_public.h )\n\n'

    _str = _str + 'add_library                ( ${_lib_name} SHARED ${_source_files} )\n'
    if type == 'CUDA':
        _str = _str + 'target_compile_options     ( ${_lib_name} PRIVATE ${CUDA_COMPILE_FLAGS} ${GPU_COMPILE_DEFNS} )\n'
        _str = _str + 'target_compile_options     ( ${_lib_name} PRIVATE ${ADDL_COMPILE_FLAGS} )\n'
        _str = _str + 'set_property        ( TARGET ${_lib_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )\n\n'
    else:
        _str = _str + 'target_compile_options     ( ${_lib_name} PRIVATE ${HIP_COMPILE_FLAGS} ${ADDL_COMPILE_FLAGS} )\n\n'

    _str = _str + 'if ( WIN32 )\n'
    _str = _str + '    set_property    ( TARGET ${_lib_name} PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON )\n'
    _str = _str + 'endif ()\n\n'

    _str = _str + 'install ( TARGETS\n'
    _str = _str + '          ${_lib_name}\n'
    _str = _str + '          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )\n\n'

    _str = _str + 'install ( FILES ${_incl_files}\n'
    _str = _str + '          DESTINATION ${CMAKE_INSTALL_PREFIX}/include )\n\n'

    return _str;


_extern_decls  = ''
_all_cubes     = 'static fftx::point_t<3> AllSizes3_' + _code_type + '[] = {\n'
_tuple_funcs   = 'static transformTuple_t ' + _file_stem + _code_type + '_Tuples[] = {\n'


with open ( 'cube-sizes.txt', 'r' ) as fil:
    currpid = os.getpid()
    myscrf  = 'myscript_' + str ( currpid ) + '.g'
    testsf  = 'testscript_' + str ( currpid ) + '.g'

    for line in fil.readlines():
        ##  print ( 'Line read = ' + line )
        if re.match ( '[ \t]*#', line ):                ## ignore comment lines
            continue

        if re.match ( '[ \t]*$', line ):                ## skip lines consisting of whitespace
            continue

        testscript = open ( testsf, 'w' )
        testscript.write ( line )
        testscript.write ( 'libdir := "' + _srcs_dir + '"; \n' )
        testscript.write ( 'file_suffix := "' + _file_suffix + '"; \n' )
        testscript.write ( 'fwd := ' + _fwd + '; \n' )
        testscript.write ( 'codefor := "' + _code_type + '"; \n' )
        testscript.close()

        line = re.sub ( '.*\[', '', line )               ## drop "szcube := ["
        line = re.sub ( '\].*', '', line )               ## drop "];"
        line = re.sub ( ' *', '', line )                 ## compress out white space
        line = line.rstrip()                             ## remove training newline
        dims = re.split ( ',', line )
        _dimx = dims[0]
        _dimy = dims[1]
        _dimz = dims[2]

        if re.match ( 'rconv', _xform_root ) and _code_type == 'CPU' and int ( _dimx ) > 260:
            continue

        ##  TODO: Allow a way to specify different gap file(s)
        ##  Assume gap file is named {_orig_file_stem}-frame.g
        ##  Generate the SPIRAL script: cat testscript_$pid.g & {transform}-frame.g
        _frame_file = re.sub ( '_$', '', _orig_file_stem ) + '-frame' + '.g'
        _spiralhome = os.environ.get('SPIRAL_HOME')
        _catfils = _spiralhome + '/gap/bin/catfiles.py'
        cmdstr = sys.executable + ' ' + _catfils + ' ' + myscrf + ' ' + testsf + ' ' + _frame_file
        result = subprocess.run ( cmdstr, shell=True, check=True )
        res = result.returncode

        ##  Generate the code by running SPIRAL
        if sys.platform == 'win32':
            cmdstr = _spiralhome + '/bin/spiral.bat < ' + myscrf
        else:
            cmdstr = _spiralhome + '/bin/spiral < ' + myscrf

        _func_stem = _file_stem + _dimx + 'x' + _dimy + 'x' + _dimz + '_' + _code_type
        _file_name = _func_stem + _file_suffix
        src_file_path = _srcs_dir + '/' + _file_name
        if len ( sys.argv ) < 5:
            ##  No optional argument, generate the code
            result = subprocess.run ( cmdstr, shell=True, check=True )
            res = result.returncode
        else:
            ##  Just print a message and skip copde gen (test python process/logic)
            print ( 'run spiral to create source file: ' + _file_name, flush = True )

        ##  Add the file name to the list of sources, update declarations etc. if file exists
        if os.path.exists ( src_file_path ):
            _cmake_srcs.write ( '    ' + _file_name + '\n' )

            ##  Add the extern declarations and track func name for header file
            ##  FUTURE: Need a way to handle functions with different signatures
            _extern_decls = _extern_decls + 'extern "C" { extern void init_' + _func_stem + '();  }\n'
            _extern_decls = _extern_decls + 'extern "C" { extern void destroy_' + _func_stem + '();  }\n'
            ##  TODO: Allow optional 3rd arg for symbol
            ##  _extern_decls = _extern_decls + 'extern "C" { extern void ' + _func_stem + '( double *output, double *input );  }\n\n'
            _extern_decls = _extern_decls + 'extern "C" { extern void ' + _func_stem + '( double *output, double *input, double *sym );  }\n\n'
            _all_cubes = _all_cubes + '    { ' + _dimx + ', ' + _dimy + ', ' + _dimz + ' },\n'
            _tuple_funcs = _tuple_funcs + '    { init_' + _func_stem + ', destroy_' + _func_stem + ', '
            _tuple_funcs = _tuple_funcs + _func_stem + ' },\n'

        else:
            ## Failed to generate file -- note it in build-lib-code-failures.txt
            bldf = open ( 'build-lib-code-failures.txt', 'a' )
            bldf.write  ( 'Failed to generate:   ' + src_file_path + '\n' )
            bldf.close  ()

    ##  All cube sizes processed: close list of sources, create header file
    _cmake_srcs.write ( ')\n' )
    _cmake_srcs.close()

    _hfil = _srcs_dir + '/' + _file_stem + _code_type + '_decls.h'
    _header_fil = open ( _hfil, 'w' )
    _filebody = start_header_file ( 'LIB_', _code_type )
    _header_fil.write ( _filebody )
    _header_fil.write ( _extern_decls )
    _header_fil.write ( _tuple_funcs + '    { NULL, NULL, NULL }\n};\n\n' )
    _header_fil.write ( _all_cubes + '    { 0, 0, 0 }\n};\n\n' )
    _header_fil.write ( '#endif\n\n' )
    _header_fil.close ()
    _hfil = _srcs_dir + '/' + _file_stem + 'decls.h'
    if _code_type == 'CPU':
        _omode = 'w'
    else:
        _omode = 'a'

    _header_fil = open ( _hfil, _omode )
    _header_fil.write ( '#include "' + _file_stem + _code_type + '_decls.h"\n\n' )
    _header_fil.close ()

    _hfil = _srcs_dir + '/' + _file_stem + _code_type + '_public.h'
    _header_fil = open ( _hfil, 'w' )
    _filebody = start_header_file ( 'PUBLIC_',  _code_type )
    _filebody = _filebody + body_public_header ( _code_type )
    _header_fil.write ( _filebody )
    _header_fil.close ()

    _hfil = _srcs_dir + '/' + _file_stem + 'public.h'
    _header_fil = open ( _hfil, 'w' )
    _filebody = start_header_file ( 'PUBLIC_',  '' )
    _filebody = _filebody + body_public_header ( '' )
    _header_fil.write ( _filebody )
    _header_fil.close ()

    _hfil = _srcs_dir + '/' + _file_stem + _code_type + '_libentry' + _file_suffix
    _api_file = open ( _hfil, 'w' )
    _filebody = library_api ( True, _code_type )
    _filebody = _filebody + python_cuda_api ( True, _code_type, _xform_root )
    _api_file.write ( _filebody )
    _api_file.close ()

    
    _hfil = _srcs_dir + '/' + _file_stem + 'libentry.cpp'
    if os.path.exists ( _hfil ):
        os.remove ( _hfil )
        
    _hfil = _srcs_dir + '/' + _file_stem + 'libentry' + _file_suffix
    _api_file = open ( _hfil, 'w' )
    _filebody = library_api ( False, _code_type )
    _filebody = _filebody + python_cuda_api ( False, _code_type, _xform_root )
    _api_file.write ( _filebody )
    _api_file.close ()

    _hfil = _srcs_dir + '/CMakeLists.txt'
    _cmake_file = open ( _hfil, 'w' )
    _filebody = cmake_library ( _code_type )
    _cmake_file.write ( _filebody )
    _cmake_file.close ()

    if os.path.exists ( myscrf ):
        os.remove ( myscrf )
    if os.path.exists ( testsf ):
        os.remove ( testsf )

sys.exit (0)
