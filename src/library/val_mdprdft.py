#! python

##  Validate FFTX built libraries against numpy computed versions of the transforms.
##  Exercise all the sizes in the library (read cube-sizes file) and call both forward and
##  inverse transforms.  Optionally, specify a single cube size to validate.

import ctypes
import sys
import re
import os
import numpy as np

if len(sys.argv) < 2:
    print ('Usage: ' + sys.argv[0] + ': libdir [-s cube_size] [-f sizes_file]' )
    sys.exit ('missing argument(s), NOTE: Only one of -s or -f should be specified')

_under = '_'
_libdir = sys.argv[1]
fftxpre = 'fftx_'
myvar   = sys.argv[0]
if re.match ( 'val_', myvar ):
    _xfmseg = re.split ( _under, myvar )
    _xfmseg = re.split ( '\.', _xfmseg[1] )
    _xform  = fftxpre + _xfmseg[0]

_xform  = _xform.rstrip()                             ## remove training newline

if not re.match ( '_$', _xform ):                ## append an underscore if one is not present
    _xform = _xform + _under

if re.match ( '^.*_.*_', _xform ):
    _xfmseg = re.split ( _under, _xform )
    _libfwd = _xfmseg[0] + _under + _xfmseg[1]
    _libinv = _xfmseg[0] + _under + 'i' + _xfmseg[1]

if sys.platform == 'win32':
    _libfwd = _libfwd + '.dll'
    _libinv = _libinv + '.dll'
    _libext = '.dll'
elif sys.platform == 'darwin':
    _libfwd = 'lib' + _libfwd + '.dylib'
    _libinv = 'lib' + _libinv + '.dylib'
    _libext = '.dylib'
else:
    _libfwd = 'lib' + _libfwd + '.so'
    _libinv = 'lib' + _libinv + '.so'
    _libext = '.so'

print ( 'library stems for fwd/inv xforms = ' + _libfwd + ' / ' + _libinv + ' lib ext = ' + _libext, flush = True )


def exec_xform ( segnams, dims, fwd, libext, typecode ):
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
    froot = froot + segnams[1]
    pywrap = froot + libext + _under + 'python' + _under

    if fwd:
        _libsegs = re.split ( '\.', _libfwd )
        uselib = _libsegs[0] + libext + '.' + _libsegs[1]
    else:
        _libsegs = re.split ( '\.', _libinv )
        uselib = _libsegs[0] + libext + '.' + _libsegs[1]

    _sharedLibPath = os.path.join ( os.path.realpath ( _libdir ), uselib )
    if not os.path.exists ( _sharedLibPath ):
        print ( 'library file: ' + uselib + ' does not exist - continue', flush = True )
        return

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

    if fwd:
        ##  Real-to-Complex, R2C  (MDPRDFT)
        ##  Setup source (Real input) data -- fill with random values
        _src        = np.zeros(shape=(dx, dy, dz)).astype(np.double)
        for ix in range ( dx ):
            for iy in range ( dy ):
                for iz in range ( dz ):
                    vr = np.random.random()
                    _src[ix, iy, iz] = vr

        ##  Destination (output) -- fill with zeros, one for spiral, one for python
        _dst_py     = np.zeros(shape=(dx, dy, dz_adj)).astype(complex)
        _dst_spiral = np.zeros(shape=(dx, dy, dz_adj)).astype(complex)
        
    else:
        ##  Complex-to-Real, C2R (IMDPRDFT)              ----------------------------
        ##  Setup and repeat for the inverse transform
        if dz % 2:
            ##  Z dimension is odd -- not handling inverse transform -- skip
            print ( 'Inverse [C2R] transform skipped when Z dimension is odd' )
            return

        _src        = np.zeros(shape=(dx, dy, dz_adj)).astype(complex)
        for ix in range ( dx ):
            for iy in range ( dy ):
                for iz in range ( dz_adj ):
                    vr = np.random.random()
                    vi = np.random.random()
                    _src[ix, iy, iz] = vr + vi * 1j
        
        _dst_py     = np.zeros(shape=(dx, dy, dz)).astype(np.double)
        _dst_spiral = np.zeros(shape=(dx, dy, dz)).astype(np.double)

    _xfm_sym    = np.ones (shape=(10, 10, 10), dtype=complex)       ## dummy symbol

    ##  Evaluate using numpy
    if fwd:
        ##  Normalize results of forward xform...
        _dst_py    = np.fft.rfftn ( _src )
        _dst_py    = _dst_py / np.size ( _dst_py )
    else:
        ##  Result of inverse xform is already normalized...
        _dst_py    = np.fft.irfftn ( _src )

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

    ##  Check difference
    diff = np.max ( np.absolute ( _dst_spiral - _dst_py ) )
    if fwd:
        dir = 'forward'
    else:
        dir = 'inverse'

    print ( 'Difference between Python / Spiral(' + typecode + ') [' + dir + '] transforms = ' + str ( diff ), flush = True )

    return;


_sizesfile = 'cube-sizes.txt'

if len(sys.argv) > 2:
    ##  Optional problem size or file specified:
    if sys.argv[2] == '-s':
        ##  problem size is specified:  e.g., 80x80x80
        _probsz = sys.argv[3]
        _probsz = _probsz.rstrip()                  ##  remove training newline
        _dims = re.split ( 'x', _probsz )

        print ( 'Size = ' + _dims[0] + 'x' + _dims[1] + 'x' + _dims[2], flush = True )
        exec_xform ( _xfmseg, _dims, True, '_cpu', 'CPU' )
        exec_xform ( _xfmseg, _dims, False, '_cpu', 'CPU' )
        exec_xform ( _xfmseg, _dims, True, '_gpu', 'GPU' )
        exec_xform ( _xfmseg, _dims, False, '_gpu', 'GPU' )

        exit ()

    elif sys.argv[2] == '-f':
        ##  Option sizes file is specified:  use the supplied filename to loop over desired sizes
        _sizesfile = sys.argv[3]
        _sizesfile = _sizesfile.rstrip()

    else:
        print ( 'Unrecognized argument: ' + sys.argv[2] + ' ignoring remainder of command line', flush = True )


with open ( _sizesfile, 'r' ) as fil:
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
        exec_xform ( _xfmseg, _dims, True, '_cpu', 'CPU' )
        exec_xform ( _xfmseg, _dims, False, '_cpu', 'CPU' )
        exec_xform ( _xfmseg, _dims, True, '_gpu', 'GPU' )
        exec_xform ( _xfmseg, _dims, False, '_gpu', 'GPU' )

    exit()
