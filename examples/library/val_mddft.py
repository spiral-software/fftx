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
    _libfwd = _xfmseg[0] + _under + _xfmseg[1] + _under + 'precomp'
    _libinv = _xfmseg[0] + _under + 'i' + _xfmseg[1] + _under + 'precomp'

if sys.platform == 'win32':
    _libfwd = _libfwd + '.dll'
    _libinv = _libinv + '.dll'
else:
    _libfwd = 'lib' + _libfwd + '.so'
    _libinv = 'lib' + _libinv + '.so'

print ( 'library for fwd xform = ' + _libfwd + ' inv xform = ' + _libinv, flush = True )


def exec_xform ( segnams, dims, fwd ):
    "Run a transform specified by segment names and fwd flag of size dims"

    dx = int ( dims[0] )
    dy = int ( dims[1] )
    dz = int ( dims[2] )
    dz_adj = dz // 2 + 1

    pywrap = segnams[0] + _under
    if not fwd:
        pywrap = pywrap + 'i'
    pywrap = pywrap + segnams[1] + _under + 'python' + _under

    if fwd:
        print ( 'Size = ' + dims[0] + 'x' + dims[1] + 'x' + dims[2], flush = True )

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
    _dftsz      = np.zeros(3).astype(ctypes.c_int)
    _dftsz[0] = dx
    _dftsz[1] = dy
    _dftsz[2] = dz

    ##  Evaluate using numpy
    if fwd:
        ##  Normalize the results of the forward xfrom...
        _dst_python = np.fft.fftn ( _src )
        _dst_python = _dst_python / np.size ( _dst_python )
    else:
        ##  Result of inverse xform is already normalized...
        _dst_python = np.fft.ifftn ( _src )

    ##  Evaluate using Spiral generated code in library.  Use the python wrapper funcs in
    ##  the library (these setup/teardown GPU resources when using GPU libraries).
    if fwd:
        uselib = _libfwd
    else:
        uselib = _libinv

    _sharedLibPath = os.path.join ( os.path.realpath ( _libdir ), uselib )
    _sharedLibAccess = ctypes.CDLL ( _sharedLibPath )
    func = pywrap + 'init' + _under + 'wrapper'

    _libFuncAttr = getattr ( _sharedLibAccess, func, None)
    if _libFuncAttr is None:
        msg = 'could not find function: ' + func
        raise RuntimeError(msg)
    _status = _libFuncAttr ( _dftsz.ctypes.data_as ( ctypes.c_void_p ) )
    if not _status:
        print ( 'Size: ' + str(_dftsz) + ' was not found in library - continue' )
        return

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

    ##  Check difference
    diff = np.max ( np.absolute ( _dst_spiral - _dst_python ) )
    if fwd:
        dir = 'forward'
    else:
        dir = 'inverse'
    print ( 'Difference between Python / Spiral [' + dir + '] transforms = ' + str ( diff ), flush = True )

    return;


if len(sys.argv) == 4:
    ##  Optional problem size is specified:  e.g., 80x80x80
    _probsz = sys.argv[3]
    _probsz = _probsz.rstrip()                  ##  remove training newline
    _dims = re.split ( 'x', _probsz )

    exec_xform ( _xfmseg, _dims, True )
    exec_xform ( _xfmseg, _dims, False )

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

        exec_xform ( _xfmseg, _dims, True )
        exec_xform ( _xfmseg, _dims, False )

    exit()
