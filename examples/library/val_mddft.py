#! python

##  Validate FFTX built libraries against numpy (scipy) computed versions of the transforms
##  Exercise all the sizes in the library (read cube-sizes or dftbatch-sizes) and call both
##  forward and inverse transforms

import ctypes
import sys
import re
import os
import numpy as np

if len(sys.argv) < 3:
    print ('Usage: ' + sys.argv[0] + ' libdir xform_name' )
    sys.exit ('missing argument(s)')

_libdir = sys.argv[1]
_xform  = sys.argv[2]
_xform  = _xform.rstrip()                             ## remove training newline

_under = '_'
if not re.match ( '_$', _xform ):                ## append an underscore if one is not present
    _xform = _xform + _under

if re.match ( '^.*_.*_', _xform ):
    _xfmseg = re.split ( _under, _xform )
    _libfwd = _xfmseg[0] + _under + _xfmseg[1]
    _libinv = _xfmseg[0] + _under + 'i' + _xfmseg[1]

if sys.platform == 'win32':
    _libfwd = _libfwd + '.dll'
    _libinv = _libinv + '.dll'
else if sys.platform == 'darwin':
    _libfwd = 'lib' + _libfwd + '.dylib'
    _libinv = 'lib' + _libinv + '.dylib'
else:
    _libfwd = 'lib' + _libfwd + '.so'
    _libinv = 'lib' + _libinv + '.so'

print ( 'library for fwd xform = ' + _libfwd + ' inv xform = ' + _libinv, flush = True )

##  Default mode for library (get by calling the library <root>GetLibraryMode() func
_def_libmode = 0
_do_once = True
lmode = [ 'CPU', 'CUDA', 'HIP' ]

def exec_xform ( segnams, dims, fwd, libmode ):
    "Run a transform specified by segment names and fwd flag of size dims"

    dx = int ( dims[0] )
    dy = int ( dims[1] )
    dz = int ( dims[2] )
    dz_adj = dz // 2 + 1

    _dftsz      = np.zeros(3).astype(ctypes.c_int)
    _dftsz[0] = dx
    _dftsz[1] = dy
    _dftsz[2] = dz

    froot = segnams[0] + _under
    if not fwd:
        froot = froot + 'i'
    froot = froot + segnams[1] + _under
    pywrap = froot + 'python' + _under

    if fwd:
        uselib = _libfwd
    else:
        uselib = _libinv

    _sharedLibPath = os.path.join ( os.path.realpath ( _libdir ), uselib )
    _sharedLibAccess = ctypes.CDLL ( _sharedLibPath )

    global _do_once
    global _def_libmode
    if _do_once:
        func = froot + 'GetLibraryMode'
        _libFuncAttr = getattr ( _sharedLibAccess, func, None)
        if _libFuncAttr is None:
            ##  No library mode functions -- just run without attempting to set CPU/GPU
            msg = 'Could not find function: ' + func
            ##  raise RuntimeError(msg)
            print ( msg + ';  No CPU/GPU switching available', flush = True )
        
        _status = _libFuncAttr ( )
        ##  print ( 'Initial, default Library mode = ' + str ( _status ) )
        _def_libmode = _status
        _do_once = False
        
    if libmode == 'CPU':
        setlibmode = 0
    else:
        setlibmode = _def_libmode
        ##  Want to run library default mode (will be CUDA / HIP if available)
        ##  Only do so if present
        if _def_libmode == 0:
            ##  No GPU functions in library
            print ( 'Library only has CPU code - no GPU function available' )
            return

    func = froot + 'SetLibraryMode'
    _libFuncAttr = getattr ( _sharedLibAccess, func, None)
    if _libFuncAttr is None:
        ##  No library mode functions -- just run without attempting to set CPU/GPU
        msg = 'Could not find function: ' + func
        ##  raise RuntimeError(msg)
        print ( msg + ';  No CPU/GPU switching available', flush = True )

    global lmode
    _libFuncAttr ( setlibmode )
    ##  print ( 'Library mode set to ' + lmode[setlibmode] )
    
    func = pywrap + 'init' + _under + 'wrapper'
    _libFuncAttr = getattr ( _sharedLibAccess, func, None)
    if _libFuncAttr is None:
        msg = 'could not find function: ' + func
        raise RuntimeError(msg)
    _status = _libFuncAttr ( _dftsz.ctypes.data_as ( ctypes.c_void_p ) )
    if not _status:
        print ( 'Size: ' + str(_dftsz) + ' was not found in library - continue' )
        return

    ##  Setup source (input) data -- fill with random values
    _src        = np.random.rand(dx, dy, dz).astype(complex)
    _src_spiral = _src.view(dtype=np.double)
    for ix in range ( dx ):
        for iy in range ( dy ):
            for iz in range ( dz ):
                vr = np.random.random()
                vi = np.random.random()
                _src[ix, iy, iz] = vr + vi * 1j

    ##  Destination (output) -- fill with zeros, one for spiral, one for python
    _dst_python = np.zeros(shape=(dx, dy, dz), dtype=complex)
    _dst_spiral = np.zeros(shape=(dx, dy, dz), dtype=complex)
    _xfm_sym    = np.ones (shape=(10, 10, 10), dtype=complex)       ## dummy symbol

    ##  Evaluate using Spiral generated code in library.  Use the python wrapper funcs in
    ##  the library (these setup/teardown GPU resources when using GPU libraries).
    ##  Call the library function
    func = pywrap + 'run' + _under + 'wrapper'
    _libFuncAttr = getattr ( _sharedLibAccess, func )
    _libFuncAttr ( _dftsz.ctypes.data_as ( ctypes.c_void_p ),
                   _dst_spiral.ctypes.data_as ( ctypes.c_void_p ),
                   _src.ctypes.data_as ( ctypes.c_void_p ),
                   _xfm_sym.ctypes.data_as ( ctypes.c_void_p ) )
    ##  Normalize Spiral result
    _dst_spiral = _dst_spiral / np.size ( _dst_spiral )

    ##  Call the transform's destroy function
    func = pywrap + 'destroy' + _under + 'wrapper'
    _libFuncAttr = getattr ( _sharedLibAccess, func )
    _libFuncAttr ( _dftsz.ctypes.data_as ( ctypes.c_void_p ) )

    ##  Evaluate using numpy
    if fwd:
        ##  Normalize the results of the forward xfrom...
        _dst_python = np.fft.fftn ( _src )
        _dst_python = _dst_python / np.size ( _dst_python )
    else:
        ##  Result of inverse xform is already normalized...
        _dst_python = np.fft.ifftn ( _src )

    ##  Check difference
    diff = np.max ( np.absolute ( _dst_spiral - _dst_python ) )
    if fwd:
        dir = 'forward'
    else:
        dir = 'inverse'

    print ( 'Difference between Python / Spiral(' + lmode[setlibmode] + ') [' + dir + '] transforms = ' + str ( diff ), flush = True )

    return;


if len(sys.argv) == 4:
    ##  Optional problem size is specified:  e.g., 80x80x80
    _probsz = sys.argv[3]
    _probsz = _probsz.rstrip()                  ##  remove training newline
    _dims = re.split ( 'x', _probsz )

    print ( 'Size = ' + _dims[0] + 'x' + _dims[1] + 'x' + _dims[2], flush = True )
    exec_xform ( _xfmseg, _dims, True, 'CPU' )
    exec_xform ( _xfmseg, _dims, False, 'CPU' )
    exec_xform ( _xfmseg, _dims, True, '' )
    exec_xform ( _xfmseg, _dims, False, '' )

    exit ()

    
with open ( 'cube-sizes.txt', 'r' ) as fil:
    for line in fil.readlines():
        ##  print ( 'Line read = ' + line )
        if re.match ( '[ \t]*#', line ):                ## ignore comment lines
            continue

        if re.match ( '[ \t]*$', line ):                ## skip lines consisting of whitespace
            continue

        line = re.sub ( '.*\[', '', line )              ## drop "szcube := ["
        line = re.sub ( '\].*', '', line )              ## drop "];"
        line = re.sub ( ' *', '', line )                ## compress out white space
        line = line.rstrip()                            ## remove training newline
        _dims = re.split ( ',', line )

        print ( 'Size = ' + _dims[0] + 'x' + _dims[1] + 'x' + _dims[2], flush = True )
        exec_xform ( _xfmseg, _dims, True, 'CPU' )
        exec_xform ( _xfmseg, _dims, False, 'CPU' )
        exec_xform ( _xfmseg, _dims, True, '' )
        exec_xform ( _xfmseg, _dims, False, '' )

    exit()
