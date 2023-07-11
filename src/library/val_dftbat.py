#! python

##  Validate FFTX built libraries against numpy computed versions of the transforms
##  Exercise all the sizes in the library (read dftbatch-sizes) and call both forward and
##  inverse transforms.  Optionally, specify a single cube size to validate.

import ctypes
import sys
import re
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser ( description = 'Validate FFTX built libraries against NumPy computed versions of the transforms' )

##  Positional argument: libdir
parser.add_argument ( 'libdir', type=str, help='directory of the library' )

##  Optional arguments
group = parser.add_mutually_exclusive_group()
group.add_argument ( '-s', '--size', type=str, help='size of the transform (e.g., 16x80x2)' )
group.add_argument ( '-f', '--file', type=str, help='file containing sizes to loop over' )

args = parser.parse_args()

_libdir = args.libdir

_under = '_'
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

print ( 'library stems for fwd/inv xforms = {} / {} lib ext = {}'.format(_libfwd, _libinv, _libext), flush = True )


def exec_xform ( segnams, dims, fwd, libext, typecode ):
    "Run a transform specified by segment names and fwd flag of size dims"

    nbat = int ( dims[0] )
    dz   = int ( dims[1] )
    styp = int ( dims[2] )

    froot = segnams[0] + _under
    if not fwd:
        froot = froot + 'i'
    froot = froot + segnams[1]
    pywrap = froot + libext + _under + 'python' + _under

    ##  Setup source (input) data -- fill with random values
    _src        = np.zeros(shape=(nbat, dz)).astype(complex)
    for ib in range ( nbat ):
        for ii in range ( dz ):
            vr = np.random.random()
            vi = np.random.random()
            _src[ib, ii] = vr + vi * 1j

    ##  Destination (output) -- fill with zeros, one for spiral, one for python
    _dst_python = np.zeros(shape=(nbat, dz)).astype(complex)
    _dst_spiral = np.zeros(shape=(nbat, dz)).astype(complex)
    _xfmsz      = np.zeros(3).astype(ctypes.c_int)
    _xfmsz[0] = nbat
    _xfmsz[1] = dz
    _xfmsz[2] = styp

    ##  Evaluate using numpy ... 
    if fwd:
        for ii in range ( nbat ):
            _biidft = np.fft.fft ( _src[ii,:] )
            _dst_python[ii,:] = _biidft
    else:
        for ii in range ( nbat ):
            _biidft = np.fft.ifft ( _src[ii,:] )
            _dst_python[ii,:] = _biidft

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
        print ( 'library file: {} does not exist - continue'.format(uselib), flush = True )
        return

    _sharedLibAccess = ctypes.CDLL ( _sharedLibPath )

    func = pywrap + 'init' + _under + 'wrapper'
    _libFuncAttr = getattr ( _sharedLibAccess, func, None)
    if _libFuncAttr is None:
        msg = 'could not find function: ' + func
        raise RuntimeError(msg)
    _status = _libFuncAttr ( _xfmsz.ctypes.data_as ( ctypes.c_void_p ) )
    if not _status:
        print ( 'Size: {} was not found in library - continue'.format(_xfmsz), flush = True )
        return

    ##  Call the library function
    func = pywrap + 'run' + _under + 'wrapper'
    try:
        _libFuncAttr = getattr ( _sharedLibAccess, func )
        _libFuncAttr ( _xfmsz.ctypes.data_as ( ctypes.c_void_p ),
                       _dst_spiral.ctypes.data_as ( ctypes.c_void_p ),
                       _src.ctypes.data_as ( ctypes.c_void_p ) )
        print ( 'Called library func: {}'.format(func), flush = True )
    except Exception as e:
        print("Error occurred during library function call:", type(e).__name__)
        print("Exception details:", str(e))
        return

    if not fwd:
        ##  Normalize Spiral result: numpy result is normalized
        ##  Divide by Length (dz) as size _dst_spiral increases with number batches
        _dst_spiral = _dst_spiral / dz

    ##  Call the transform's destroy function
    func = pywrap + 'destroy' + _under + 'wrapper'
    _libFuncAttr = getattr ( _sharedLibAccess, func )
    _libFuncAttr ( _xfmsz.ctypes.data_as ( ctypes.c_void_p ) )

    ##  Check difference
    _diff = np.max ( np.absolute ( _dst_spiral - _dst_python ) )
    if fwd:
        dir = 'forward'
    else:
        dir = 'inverse'

    print ( 'Difference between Python / Spiral({}) [{}] transforms = {}'.format(typecode, dir, _diff), flush = True )

    return;


_sizesfile = 'dftbatch-sizes.txt'

if args.size:
    _probsz = args.size.rstrip()
    _dims = re.split ( 'x', _probsz )
    print ( 'Size = {} (batches) x {} (length) x {} (stride type)'.format ( _dims[0], _dims[1], _dims[2] ), flush = True )
    exec_xform ( _xfmseg, _dims, True, '_cpu', 'CPU' )
    exec_xform ( _xfmseg, _dims, False, '_cpu', 'CPU' )

    exec_xform ( _xfmseg, _dims, True, '_gpu', 'GPU' )
    exec_xform ( _xfmseg, _dims, False, '_gpu', 'GPU' )

    exit ()

elif args.file:
    _sizesfile = args.file.rstrip()


with open ( _sizesfile, 'r' ) as fil:
    for line in fil.readlines():
        ##  print ( 'Line read = {}'.format(line), flush = True )
        if re.match ( '[ \t]*#', line ):                ## ignore comment lines
            continue

        if re.match ( '[ \t]*$', line ):                ## skip lines consisting of whitespace
            continue

        # _dims = [ 0, 0 ]
        # _nbat = re.sub ( '.*nbatch :=', '', line )      ## get number batches
        # _nbat = re.sub ( ';.*', '', _nbat )
        # _nbat = re.sub ( ' *', '', _nbat )              ## compress out white space
        # _nbat = _nbat.rstrip()                          ## remove training newline
        # _dims[0] = _nbat
        
        # line = re.sub ( '.*\[', '', line )              ## drop "szns := ["
        # line = re.sub ( '\].*', '', line )              ## drop "];"
        # line = re.sub ( ' *', '', line )                ## compress out white space
        # line = line.rstrip()                            ## remove training newline
        # _dims[1] = line

        _dims = [ 0, 0, 0 ]

        line = re.sub ( ' ', '', line )                 ## suppress white space
        segs = re.split ( ';', line )                   ## expect 3 segments
        chnk = re.split ( '=', segs[0] )
        _dims[0] = chnk[1]

        segs[1] = re.sub ( '\[', '', segs[1] )
        segs[1] = re.sub ( '\]', '', segs[1] )
        chnk = re.split ( '=', segs[1] )
        _dims[1] = chnk[1]
        
        chnk = re.split ( '=', segs[2] )
        _dims[2] = chnk[1]
        
        print ( 'Size = {} (batches) x {} (length) x {} (stride type)'.format ( _dims[0], _dims[1], _dims[2] ), flush = True )
        exec_xform ( _xfmseg, _dims, True, '_cpu', 'CPU' )
        exec_xform ( _xfmseg, _dims, False, '_cpu', 'CPU' )

        exec_xform ( _xfmseg, _dims, True, '_gpu', 'GPU' )
        exec_xform ( _xfmseg, _dims, False, '_gpu', 'GPU' )

    exit()
