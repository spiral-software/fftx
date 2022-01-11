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

##  Default mode for library (get by calling the library <root>GetLibraryMode() func
_def_libmode = 0
_do_once = True
lmode = [ 'CPU', 'CUDA', 'HIP' ]


def exec_xform ( segnams, dims, fwd, libmode ):
    "Run a transform specified by segment names and fwd flag of size dims"

    nbat = int ( dims[0] )
    dz   = int ( dims[1] )
    dz_adj = dz // 2 + 1

    froot = segnams[0] + _under
    if not fwd:
        froot = froot + 'i'
    froot = froot + segnams[1] + _under
    pywrap = froot + 'python' + _under

    if fwd:
        ##  Forward transform: Real-to-Complex, R2C (PRDFT)
        ##  Setup source (input) data -- fill with random values
        _src        = np.zeros(shape=(nbat, dz)).astype(np.double)
        _src_spiral = _src.view(dtype=np.double)
        for ib in range ( nbat ):
            for ii in range ( dz ):
                vr = np.random.random()
                _src[ib, ii] = vr               ##   + vi * 1j

        ##  Destination (output) -- fill with zeros, one for spiral, one for python
        ##  Destination is nbat * dz // 2 + 1 complex (and * 2 double for spiral)
        _dst_python = np.zeros(shape=(nbat, dz_adj)).astype(complex)
        _dst_spiral = np.zeros((nbat * dz_adj * 2), dtype=np.double)
    else:
        ##  Complex-to-Real, C2R (IPRDFT)
        ##  Setup source (input) data for the inverse transform
        _src        = np.zeros(shape=(nbat, dz_adj)).astype(complex)
        _src_spiral = _src.view(dtype=np.double)
        for ib in range ( nbat ):
            for ii in range ( dz_adj ):
                vr = np.random.random()
                vi = np.random.random()
                _src[ib, ii] = vr + vi * 1j
        
        ##  Destination (output) -- fill with zeros, one for spiral, one for python
        _dst_python = np.zeros(shape=(nbat, dz)).astype(np.double)
        _dst_spiral = np.zeros((nbat * dz), dtype=np.double)

    xfmsz    = np.zeros(2).astype(ctypes.c_int)         ## dimensions for library lookup
    xfmsz[0] = nbat
    xfmsz[1] = dz

    ##  Evaluate using numpy ... 
    if fwd:
        for ii in range ( nbat ):
            _biidft = np.fft.rfft ( _src[ii,:] )
            _dst_python[ii,:] = _biidft

        _dst_python_IL = _dst_python.view(dtype=np.double).flatten()
    else:
        for ii in range ( nbat ):
            _biidft = np.fft.irfft ( _src[ii,:] )
            _dst_python[ii,:] = _biidft

        _dst_python_IL = _dst_python.view().flatten()  ##  dtype=np.double

    ##  Evaluate using Spiral generated code in library.  Use the python wrapper funcs in
    ##  the library (these setup/teardown GPU resources when using GPU libraries).
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
    _status = _libFuncAttr ( xfmsz.ctypes.data_as ( ctypes.c_void_p ) )
    if not _status:
        print ( 'Size: ' + str(xfmsz) + ' was not found in library - continue' )
        return

    ##  Call the library function
    func = pywrap + 'run' + _under + 'wrapper'
    _libFuncAttr = getattr ( _sharedLibAccess, func )
    _libFuncAttr ( xfmsz.ctypes.data_as ( ctypes.c_void_p ),
                   _dst_spiral.ctypes.data_as ( ctypes.c_void_p ),
                   _src_spiral.ctypes.data_as ( ctypes.c_void_p ) )
    if not fwd:
        ##  Normalize Spiral result: numpy result is normalized
        ##  Divide by Length (dz) as size _dst_spiral increases with number batches
        _dst_spiral = _dst_spiral / dz  ##  np.size ( _dst_python )

    ##  Call the transform's destroy function
    func = pywrap + 'destroy' + _under + 'wrapper'
    _libFuncAttr = getattr ( _sharedLibAccess, func )
    _libFuncAttr ( xfmsz.ctypes.data_as ( ctypes.c_void_p ) )

    ##  Check difference
    _diff = np.max ( np.absolute ( _dst_spiral - _dst_python_IL ) )
    if fwd:
        dir = 'forward'
    else:
        dir = 'inverse'

    print ( 'Difference between Python / Spiral(' + lmode[setlibmode] + ') [' + dir + '] transforms = ' + str ( _diff ), flush = True )

    return;



##  open the right file of sizes: dftbat | prdftbat ==> dftbatch-sizes.txt

with open ( 'dftbatch-sizes.txt', 'r' ) as fil:
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

        print ( 'Batches = ' + _dims[0] + ';  Size = ' + _dims[1] + ';', flush = True )
        exec_xform ( _xfmseg, _dims, True, 'CPU' )
        exec_xform ( _xfmseg, _dims, False, 'CPU' )

        exec_xform ( _xfmseg, _dims, True, '' )
        exec_xform ( _xfmseg, _dims, False, '' )

    exit()
