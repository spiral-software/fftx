#! python

##  Validate FFTX built libraries against numpy computed versions of the transforms
##  Exercise all the sizes in the library (read dftbatch-sizes) and call both forward and
##  inverse transforms.  Optionally, specify a single cube size to validate.

import ctypes
import sys
import re
import os
import numpy as np

if len(sys.argv) < 2:
    print ('Usage: ' + sys.argv[0] + ': libdir [-s <batch>x<length>] [-f sizes_file]' )
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

    nbat = int ( dims[0] )
    dz   = int ( dims[1] )

    froot = segnams[0] + _under
    if not fwd:
        froot = froot + 'i'
    froot = froot + segnams[1]
    pywrap = froot + libext + _under + 'python' + _under

    ##  Setup source (input) data -- fill with random values
    _src        = np.zeros(shape=(nbat, dz)).astype(complex)
    _src_spiral = _src.view(dtype=np.double)
    for ib in range ( nbat ):
        for ii in range ( dz ):
            vr = np.random.random()
            vi = np.random.random()
            _src[ib, ii] = vr + vi * 1j

    ##  Destination (output) -- fill with zeros, one for spiral, one for python
    _dst_python = np.zeros(shape=(nbat, dz)).astype(complex)
    _dst_spiral = np.zeros((nbat * dz * 2), dtype=np.double)
    _xfmsz      = np.zeros(2).astype(ctypes.c_int)
    _xfmsz[0] = nbat
    _xfmsz[1] = dz

    ##  Evaluate using numpy ... 
    if fwd:
        for ii in range ( nbat ):
            _biidft = np.fft.fft ( _src[ii,:] )
            _dst_python[ii,:] = _biidft
    else:
        for ii in range ( nbat ):
            _biidft = np.fft.ifft ( _src[ii,:] )
            _dst_python[ii,:] = _biidft

    _dst_python_IL = _dst_python.view(dtype=np.double).flatten()

    ##  Evaluate using Spiral generated code in library.  Use the python wrapper funcs in
    ##  the library (these setup/teardown GPU resources when using GPU libraries).
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
    _status = _libFuncAttr ( _xfmsz.ctypes.data_as ( ctypes.c_void_p ) )
    if not _status:
        print ( 'Size: ' + str(_xfmsz) + ' was not found in library - continue' )
        return

    ##  Call the library function
    func = pywrap + 'run' + _under + 'wrapper'
    _libFuncAttr = getattr ( _sharedLibAccess, func )
    _libFuncAttr ( _xfmsz.ctypes.data_as ( ctypes.c_void_p ),
                   _dst_spiral.ctypes.data_as ( ctypes.c_void_p ),
                   _src.ctypes.data_as ( ctypes.c_void_p ) )
    if not fwd:
        ##  Normalize Spiral result: numpy result is normalized
        ##  Divide by Length (dz) as size _dst_spiral increases with number batches
        _dst_spiral = _dst_spiral / dz  ##  np.size ( _dst_python )

    ##  Call the transform's destroy function
    func = pywrap + 'destroy' + _under + 'wrapper'
    _libFuncAttr = getattr ( _sharedLibAccess, func )
    _libFuncAttr ( _xfmsz.ctypes.data_as ( ctypes.c_void_p ) )

    ##  Check difference
    _diff = np.max ( np.absolute ( _dst_spiral - _dst_python_IL ) )
    if fwd:
        dir = 'forward'
    else:
        dir = 'inverse'

    print ( 'Difference between Python / Spiral(' + typecode + ') [' + dir + '] transforms = ' + str ( _diff ), flush = True )

    return;


_sizesfile = 'dftbatch-sizes.txt'

if len(sys.argv) > 2:
    ##  Optional problem size or file specified:
    if sys.argv[2] == '-s':
        ##  problem size is specified:  e.g., 80x80x80
        _probsz = sys.argv[3]
        _probsz = _probsz.rstrip()                  ##  remove training newline
        _dims = re.split ( 'x', _probsz )

        print ( 'Size = ' + _dims[0] + ' (batches) x ' + _dims[1] + ' (length)', flush = True )
        exec_xform ( _xfmseg, _dims, True, '_cpu', 'CPU' )
        exec_xform ( _xfmseg, _dims, False, '_cpu', 'CPU' )

        # exec_xform ( _xfmseg, _dims, True, '_gpu', 'GPU' )
        # exec_xform ( _xfmseg, _dims, False, '_gpu', 'GPU' )

        exit ()

    elif sys.argv[2] == '-f':
        ##  Option sizes file is specified:  use the supplied filename to loop over desired sizes
        _sizesfile = sys.argv[3]
        _sizesfile = _sizesfile.rstrip()

    else:
        print ( 'Unrecognized argument: ' + sys.argv[2] + ' ignoring remainder of command line', flush = True )


with open ( _sizesfile, 'r' ) as fil:
    for line in fil.readlines():
        ##  print ( 'Line read = ' + line, flush = True )
        if re.match ( '[ \t]*#', line ):                ## ignore comment lines
            continue

        if re.match ( '[ \t]*$', line ):                ## skip lines consisting of whitespace
            continue

        _dims = [ 0, 0 ]
        _nbat = re.sub ( '.*nbatch :=', '', line )      ## get number batches
        _nbat = re.sub ( ';.*', '', _nbat )
        _nbat = re.sub ( ' *', '', _nbat )              ## compress out white space
        _nbat = _nbat.rstrip()                          ## remove training newline
        _dims[0] = _nbat
        
        line = re.sub ( '.*\[', '', line )              ## drop "szns := ["
        line = re.sub ( '\].*', '', line )              ## drop "];"
        line = re.sub ( ' *', '', line )                ## compress out white space
        line = line.rstrip()                            ## remove training newline
        _dims[1] = line

        print ( 'Size = ' + _dims[0] + ' (batches) x ' + _dims[1] + ' (length)', flush = True )
        exec_xform ( _xfmseg, _dims, True, '_cpu', 'CPU' )
        exec_xform ( _xfmseg, _dims, False, '_cpu', 'CPU' )

        # exec_xform ( _xfmseg, _dims, True, '_gpu', 'GPU' )
        # exec_xform ( _xfmseg, _dims, False, '_gpu', 'GPU' )

    exit()
