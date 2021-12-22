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

_cmake_srcs = open ( _srcs_dir + '/SourceList.cmake', 'w' )
_cmake_srcs.write ( 'set ( _source_files ${_source_files} \n' )

##  Build a header file for the library with the declarations and tables to
##  manage the entry points in the library

_lib_hdrfname = _srcs_dir + '/' + _file_stem + 'decls.h'
_lib_pubfname = _srcs_dir + '/' + _file_stem + 'public.h'
_lib_apifname = _srcs_dir + '/' + _file_stem + 'libentry' + _file_suffix
_lib_cmakefil = _srcs_dir + '/CMakeLists.txt'


def start_header_file ( type ):
    "Sets up the common stuff for both header files"
    _str = '#ifndef ' + _file_stem + type + '_HEADER_INCLUDED\n'
    _str = _str + '#define ' + _file_stem + type + '_HEADER_INCLUDED\n\n'
    _str = _str + '//  Copyright (c) 2018-2021, Carnegie Mellon University\n'
    _str = _str + '//  See LICENSE for details\n\n'

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

##    _str = _str + '#ifndef CUBESIZE_T\n'
##    _str = _str + '#define CUBESIZE_T\n'
##    _str = _str + 'typedef struct cubesize { int dimx, dimy, dimz; } cubesize_t;\n'
##    _str = _str + '#endif\n\n'

    _str = _str + '#ifndef TRANSFORMTUPLE_T\n'
    _str = _str + '#define TRANSFORMTUPLE_T\n'
    _str = _str + 'typedef struct transformTuple {\n'
    _str = _str + '    initTransformFunc    initfp;\n'
    _str = _str + '    destroyTransformFunc destroyfp;\n'
    _str = _str + '    runTransformFunc     runfp;\n'
    _str = _str + '} transformTuple_t;\n'
    _str = _str + '#endif\n\n'

    return _str;


def body_public_header ():
    "Add the body details for the public header file"
    _str =        '//  Query the list of sizes available from the library; returns a pointer to an\n'
    _str = _str + '//  array of size, each element is a struct of type fftx::point_t<3> specifying the X,\n'
    _str = _str + '//  Y, and Z dimensions\n\n'

    _str = _str + 'fftx::point_t<3> * ' + _file_stem + 'QuerySizes ();\n\n'

    _str = _str + '//  Run an ' + _file_stem + ' transform once: run the init functions, run the transform,\n'
    _str = _str + '//  and finally tear down by calling the destroy function.  Accepts fftx::point_t<3>\n'
    _str = _str + '//  specifying size, and pointers to the output (returned) data and the input\n'
    _str = _str + '//  data.\n\n'

    ##  TODO: Allow optional 3rd arg for symbol
    ##  _str = _str + 'void ' + _file_stem + 'Run ( fftx::point_t<3> req, double * output, double * input );\n\n'
    _str = _str + 'void ' + _file_stem + 'Run ( fftx::point_t<3> req, double * output, double * input, double * sym );\n\n'

    _str = _str + '//  Get a transform tuple -- a set of pointers to the init, destroy, and run\n'
    _str = _str + '//  functions for a specific size ' + _file_stem + ' transform.  Using this information the\n'
    _str = _str + '//  user may call the init function to setup for the transform, then run the\n'
    _str = _str + '//  transform repeatedly, and finally tesr down (using destroy function).\n\n'

    _str = _str + 'transformTuple_t * ' + _file_stem + 'Tuple ( fftx::point_t<3> req );\n\n'

    _str = _str + '//  Wrapper functions to allow python to call CUDA/HIP GPU code.\n\n'
    _str = _str + 'extern "C" {\n\n'
    _str = _str + 'int  ' + _file_stem + 'python_init_wrapper ( int * req );\n'
    _str = _str + 'void ' + _file_stem + 'python_run_wrapper ( int * req, double * output, double * input, double * sym );\n'
    _str = _str + 'void ' + _file_stem + 'python_destroy_wrapper ( int * req );\n\n'
    _str = _str + '}\n\n#endif\n\n'

    return _str;


def library_api ( type ):
    "Sets up the public API for the library"
    _str =        '//  Copyright (c) 2018-2021, Carnegie Mellon University\n'
    _str = _str + '//  See LICENSE for details\n\n'

    _str = _str + '#include <stdio.h>\n'
    _str = _str + '#include <stdlib.h>\n'
    _str = _str + '#include <string.h>\n'
    _str = _str + '#include "' + _file_stem + 'decls.h"\n'
    _str = _str + '#include "' + _file_stem + 'public.h"\n'
    if type == 'CUDA':
        _str = _str + '#include <helper_cuda.h>\n\n'
    elif type == 'HIP':
        _str = _str + '#include <hip/hip_runtime.h>\n\n'
        _str = _str + '#define checkLastHipError(str)	{ hipError_t err = hipGetLastError();	if (err != hipSuccess) {  printf("%s: %s\\n", (str), hipGetErrorString(err) );  exit(-1); } }\n\n'
        ##  _str = _str + '#include <hipfft.h>\n#include "rocfft.h"\n\n'

    _str = _str + '//  Query the list of sizes available from the library; returns a pointer to an\n'
    _str = _str + '//  array of size, each element is a struct of type fftx::point_t<3> specifying the X,\n'
    _str = _str + '//  Y, and Z dimensions\n\n'

    _str = _str + 'fftx::point_t<3> * ' + _file_stem + 'QuerySizes ()\n'
    _str = _str + '{\n'
    _str = _str + '    fftx::point_t<3> *wp = (fftx::point_t<3> *) malloc ( sizeof ( AllSizes3 ) );\n'
    _str = _str + '    if ( wp != NULL)\n'
    _str = _str + '        memcpy ( (void *) wp, (const void *) AllSizes3, sizeof ( AllSizes3 ) );\n\n'
    _str = _str + '    return wp;\n'
    _str = _str + '}\n\n'

    _str = _str + '//  Get a transform tuple -- a set of pointers to the init, destroy, and run\n'
    _str = _str + '//  functions for a specific size ' + _file_stem + ' transform.  Using this information the\n'
    _str = _str + '//  user may call the init function to setup for the transform, then run the\n'
    _str = _str + '//  transform repeatedly, and finally tear down (using destroy function).\n'
    _str = _str + '//  Returns NULL if requested size is not found\n\n'

    _str = _str + 'transformTuple_t * ' + _file_stem + 'Tuple ( fftx::point_t<3> req )\n'
    _str = _str + '{\n'
    _str = _str + '    int indx;\n'
    _str = _str + '    int numentries = sizeof ( AllSizes3 ) / sizeof ( fftx::point_t<3> ) - 1;    // last entry in { 0, 0, 0 }\n'
    _str = _str + '    transformTuple_t *wp = NULL;\n\n'

    _str = _str + '    for ( indx = 0; indx < numentries; indx++ ) {\n'
    _str = _str + '        if ( req[0] == AllSizes3[indx][0] &&\n'
    _str = _str + '             req[1] == AllSizes3[indx][1] &&\n'
    _str = _str + '             req[2] == AllSizes3[indx][2] ) {\n'
    _str = _str + '            // found a match\n'
    _str = _str + '            wp = (transformTuple_t *) malloc ( sizeof ( transformTuple_t ) );\n'
    _str = _str + '            if ( wp != NULL) {\n'
    _str = _str + '                *wp = ' + _file_stem + 'Tuples[indx];\n'
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
    ##  _str = _str + 'void ' + _file_stem + 'Run ( fftx::point_t<3> req, double * output, double * input )\n'
    _str = _str + 'void ' + _file_stem + 'Run ( fftx::point_t<3> req, double * output, double * input, double * sym )\n'
    _str = _str + '{\n'
    _str = _str + '    transformTuple_t *wp = ' + _file_stem + 'Tuple ( req );\n'
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

    return _str;


def python_cuda_api ( type, xfm ):
    "Sets up the python wrapper API to allow Python to call C++ CUDA or HIP code. \
     For CPU code no data management or marshalling is required"
    _str =        '//  Host-to-Device C/CUDA wrapper functions to permit Python to call the CUDA kernels.\n\n'
    if type == 'CUDA' or type == 'HIP':
        _str = _str + 'static double *dev_in, *dev_out, *dev_sym;\n\n'
    _str = _str + 'extern "C" {\n\n'

    _str = _str + 'int  ' + _file_stem + 'python_init_wrapper ( int * req )\n{\n'
    _str = _str + '    //  Get the tuple for the requested size\n'
    _str = _str + '    fftx::point_t<3> rsz;\n'
    _str = _str + '    rsz[0] = req[0];  rsz[1] = req[1];  rsz[2] = req[2];\n'
    _str = _str + '    transformTuple_t *wp = ' + _file_stem + 'Tuple ( rsz );\n'
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
        ##  Amount of data space to mlloc depends on transform:
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

    _str = _str + 'void ' + _file_stem + 'python_run_wrapper ( int * req, double * output, double * input, double * sym )\n{\n'
    _str = _str + '    //  Get the tuple for the requested size\n'
    _str = _str + '    fftx::point_t<3> rsz;\n'
    _str = _str + '    rsz[0] = req[0];  rsz[1] = req[1];  rsz[2] = req[2];\n'
    _str = _str + '    transformTuple_t *wp = ' + _file_stem + 'Tuple ( rsz );\n'
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

    _str = _str + 'void ' + _file_stem + 'python_destroy_wrapper ( int * req )\n{\n'
    _str = _str + '    //  Get the tuple for the requested size\n'
    _str = _str + '    fftx::point_t<3> rsz;\n'
    _str = _str + '    rsz[0] = req[0];  rsz[1] = req[1];  rsz[2] = req[2];\n'
    _str = _str + '    transformTuple_t *wp = ' + _file_stem + 'Tuple ( rsz );\n'
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

    return _str;


def cmake_library ( type ):
    _str =        '##\n## Copyright (c) 2018-2021, Carnegie Mellon University\n'
    _str = _str + '## All rights reserved.\n##\n## See LICENSE file for full information\n##\n\n'

    _str = _str + 'cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )\n\n'

    _str = _str + 'set ( _lib_root ' + _file_stem + ' )\n'
    _str = _str + 'set ( _lib_name ${_lib_root}precomp )\n'
    _str = _str + 'set ( _lib_name ${_lib_root}precomp PARENT_SCOPE )\n\n'

    if type == 'CUDA':
        _str = _str + 'set ( CMAKE_CUDA_ARCHITECTURES 70 )\n\n'

    _str = _str + 'include ( SourceList.cmake )\n'
    _str = _str + 'list    ( APPEND _source_files ${_lib_root}libentry' + _file_suffix + ' )\n\n'
    ##  _str = _str + 'message ( STATUS "Source file: ${_source_files}" )\n\n'

    _str = _str + 'add_library                ( ${_lib_name} SHARED ${_source_files} )\n'
    if type == 'CUDA':
        _str = _str + 'target_compile_options     ( ${_lib_name} PRIVATE ${CUDA_COMPILE_FLAGS} ${GPU_COMPILE_DEFNS} )\n'
        _str = _str + 'set_property        ( TARGET ${_lib_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )\n\n'
    else:
        _str = _str + 'target_compile_options     ( ${_lib_name} PRIVATE ${HIP_COMPILE_FLAGS} ${ADDL_COMPILE_FLAGS} )\n\n'

    _str = _str + 'if ( WIN32 )\n'
    _str = _str + '    set_property    ( TARGET ${_lib_name} PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON )\n'
    _str = _str + 'endif ()\n\n'

    _str = _str + 'install ( TARGETS\n'
    _str = _str + '          ${_lib_name}\n'
    _str = _str + '          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )\n\n'

    return _str;


_extern_decls  = ''
_all_cubes     = 'static fftx::point_t<3> AllSizes3[] = {\n'
_tuple_funcs   = 'static transformTuple_t ' + _file_stem + 'Tuples[] = {\n'


with open ( 'cube-sizes.txt', 'r' ) as fil:
    for line in fil.readlines():
        ##  print ( 'Line read = ' + line )
        if re.match ( '[ \t]*#', line ):                ## ignore comment lines
            continue

        if re.match ( '[ \t]*$', line ):                ## skip lines consisting of whitespace
            continue

        testscript = open ( 'testscript.g', 'w' )
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

        ##  Add the file name to the list of sources
        _func_stem = _file_stem + _dimx + 'x' + _dimy + 'x' + _dimz
        _file_name = _func_stem + _file_suffix
        _cmake_srcs.write ( '    ' + _file_name + '\n' )

        ##  Add the extern declarations and tranck func name for header file
        ##  FUTURE: Need a way to handle functions with different signatures
        _extern_decls = _extern_decls + 'extern "C" { extern void init_' + _func_stem + '();  }\n'
        _extern_decls = _extern_decls + 'extern "C" { extern void destroy_' + _func_stem + '();  }\n'
        ##  TODO: Allow optional 3rd arg for symbol
        ##  _extern_decls = _extern_decls + 'extern "C" { extern void ' + _func_stem + '( double *output, double *input );  }\n\n'
        _extern_decls = _extern_decls + 'extern "C" { extern void ' + _func_stem + '( double *output, double *input, double *sym );  }\n\n'
        _all_cubes = _all_cubes + '    { ' + _dimx + ', ' + _dimy + ', ' + _dimz + ' },\n'
        _tuple_funcs = _tuple_funcs + '    { init_' + _func_stem + ', destroy_' + _func_stem + ', '
        _tuple_funcs = _tuple_funcs + _func_stem + ' },\n'

        ##  TODO: Allow a way to specify different gap file(s)
        ##  Assume gap file is named {_orig_file_stem}-frame.g
        ##  Generate the SPIRAL script: cat testscript.g & {transform}-frame.g
        _frame_file = re.sub ( '_$', '', _orig_file_stem ) + '-frame' + '.g'
        _spiralhome = os.environ.get('SPIRAL_HOME')
        _catfils = _spiralhome + '/gap/bin/catfiles.py'
        cmdstr = 'python ' + _catfils + ' myscript.g testscript.g ' + _frame_file
        result = subprocess.run ( cmdstr, shell=True, check=True )
        res = result.returncode

        ##  Generate the code by running SPIRAL
        if sys.platform == 'win32':
            cmdstr = _spiralhome + '/bin/spiral.bat < myscript.g'
        else:
            cmdstr = _spiralhome + '/bin/spiral < myscript.g'

        if len ( sys.argv ) < 5:
            ##  No optional argument, generate the code
            result = subprocess.run ( cmdstr, shell=True, check=True )
            res = result.returncode
        else:
            ##  Just print a message and skip copde gen (test python process/logic)
            print ( 'run spiral to create source file: ' + _file_name, flush = True )

    ##  All cube sizes processed: close list of sources, create header file
    _cmake_srcs.write ( ')\n' )
    _cmake_srcs.close()

    _header_fil = open ( _lib_hdrfname, 'w' )
    _filebody = start_header_file ( 'LIB' )
    _header_fil.write ( _filebody )
    _header_fil.write ( _extern_decls )
    _header_fil.write ( _tuple_funcs + '    { NULL, NULL, NULL }\n};\n\n' )
    _header_fil.write ( _all_cubes + '    { 0, 0, 0 }\n};\n\n' )
    _header_fil.write ( '#endif\n\n' )
    _header_fil.close ()

    _header_fil = open ( _lib_pubfname, 'w' )
    _filebody = start_header_file ( 'PUBLIC' )
    _filebody = _filebody + body_public_header ()
    _header_fil.write ( _filebody )
    _header_fil.close ()

    _api_file = open ( _lib_apifname, 'w' )
    _filebody = library_api ( _code_type )
    _filebody = _filebody + python_cuda_api ( _code_type, _xform_root )
    _api_file.write ( _filebody )
    _api_file.close ()

    _cmake_file = open ( _lib_cmakefil, 'w' )
    _filebody = cmake_library ( _code_type )
    _cmake_file.write ( _filebody )
    _cmake_file.close ()


sys.exit (0)

