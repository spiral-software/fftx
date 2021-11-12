#! python

##  Validate FFTX built libraries against numpy (scipy) computed versions of the transforms
##  Exercise all the sizes in the library (read cube-sizes or dftbatch-sizes) and call both
##  forward and inverse transforms

import numpy as np
import ctypes
import sys
import re
import os

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

print ( 'library for fwd xform = ' + _libfwd + ' inv xform = ' + _libinv )

##  open the right file of sizes: dftbat | prdftbat ==> dftbatch-sizes.txt
##                                mddft | mdprdft ==> cube-sizes.txt

with open ( 'dftbatch-sizes.txt', 'r' ) as fil:
    for line in fil.readlines():
        ##  print ( 'Line read = ' + line )
        if re.match ( '[ \t]*#', line ):                ## ignore comment lines
            continue

        if re.match ( '[ \t]*$', line ):                ## skip lines consisting of whitespace
            continue

        _dims = re.sub ( '.*nbatch :=', '', line )      ## get number batches
        _dims = re.sub ( ';.*', '', _dims )
        _dims = re.sub ( ' *', '', _dims )              ## compress out white space
        _dims = _dims.rstrip()                          ## remove training newline
        _nbat = _dims
        _nbt  = int ( _nbat )
        ##  if ( _nbt != 1 ):
        ##      continue
        
        line = re.sub ( '.*\[', '', line )              ## drop "szns := ["
        line = re.sub ( '\].*', '', line )              ## drop "];"
        line = re.sub ( ' *', '', line )                ## compress out white space
        line = line.rstrip()                            ## remove training newline
        _dims = line
        _N = int ( _dims )
        _NN = _N
        if ( _xfmseg[1] == 'prdftbat' ):
            _NN = _N // 2 + 1
        
        _funcname = _xfmseg[0] + _under + _xfmseg[1] + _under + _nbat + _under + _dims + _under + '1d'
        _invfunc  = _xfmseg[0] + _under + 'i' + _xfmseg[1] + _under + _nbat + _under + _dims + _under + '1d'
        print ( 'Batches = ' + str ( _nbt ) + ';  Size = ' + str ( _N ) + ';' )

        ##  Setup source (input) data -- fill with random values
        _src        = np.zeros(shape=(_nbt, _N)).astype(np.double)
        _src_spiral = _src.view(dtype=np.double)
        for ib in range ( _nbt ):
            for ii in range ( _N ):
                vr = np.random.random()
                _src[ib, ii] = vr               ##   + vi * 1j

        ##  Destination (output) -- fill with zeros, one for spiral, one for python
        _dst_python = np.zeros(shape=(_nbt, _NN)).astype(complex)
        _dst_spiral = np.zeros((_nbt * _NN * 2), dtype=np.double)
        
        ##  Evaluate using numpy ... 
        if ( _xfmseg[1] == 'prdftbat' ):
            for ii in range ( _nbt ):
                _biidft = np.fft.rfft ( _src[ii,:] )
                _dst_python[ii,:] = _biidft
        else:
            for ii in range ( _nbt ):
                _biidft = np.fft.fft ( _src[ii,:] )
                _dst_python[ii,:] = _biidft

        _dst_python_IL = _dst_python.view(dtype=np.double).flatten()

        ##  Access library and call Spiral generated code...
        _sharedLibPath = os.path.join ( os.path.realpath ( _libdir ), _libfwd )
        _sharedLibAccess = ctypes.CDLL ( _sharedLibPath )
        ##  Call the SPIRAL generated init function
        _inifunc = 'init_' + _funcname
        
        _libFuncAttr = getattr ( _sharedLibAccess, _inifunc, None)
        if _libFuncAttr == None:
            msg = 'could not find function: ' + _inifunc
            raise RuntimeError(msg)
        _libFuncAttr ()
        
        ##  Call the library function
        print ( 'Function name: ' + _funcname )   ## 'Using Library = ' + _sharedLibPath
        _libFuncAttr = getattr ( _sharedLibAccess, _funcname )
        _libFuncAttr ( _dst_spiral.ctypes.data_as ( ctypes.c_void_p ),
                       _src_spiral.ctypes.data_as ( ctypes.c_void_p ) )
        ##  print ( 'Spiral result = ' )
        ##  print ( _dst_spiral )
        
        ##  Check difference
        _diff = np.max ( np.absolute ( _dst_spiral - _dst_python_IL ) )
        print ( 'Difference between Python / Spiral transforms = ' + str ( _diff ) )

        ##  Complex-to-Real, C2R (IMDPRDFT)              ----------------------------
        ##  Setup and repeat for the inverse transform
        _src        = np.zeros(shape=(_nbt, _NN)).astype(complex)
        _src_spiral = _src.view(dtype=np.double)
        for ib in range ( _nbt ):
            for ii in range ( _NN ):
                vr = np.random.random()
                vi = np.random.random()
                _src[ib, ii] = vr + vi * 1j
        
        ##  Destination (output) -- fill with zeros, one for spiral, one for python
        _dst_python = np.zeros(shape=(_nbt, _N)).astype(np.double)
        _dst_spiral = np.zeros((_nbt * _N), dtype=np.double)
        
        ##  Evaluate using numpy ... numpy already normalizes output
        if ( _xfmseg[1] == 'prdftbat' ):
            for ii in range ( _nbt ):
                _biidft = np.fft.irfft ( _src[ii,:] )
                _dst_python[ii,:] = _biidft
        else:
            for ii in range ( _nbt ):
                _biidft = np.fft.ifft ( _src[ii,:] )
                _dst_python[ii,:] = _biidft

        _dst_python_IL = _dst_python.view().flatten()  ##  dtype=np.double

        ##  Access library and call Spiral generated code...
        _sharedLibPath = os.path.join ( os.path.realpath ( _libdir ), _libinv )
        _sharedLibAccess = ctypes.CDLL ( _sharedLibPath )
        ##  Call the SPIRAL generated init function
        _inifunc = 'init_' + _invfunc
        
        _libFuncAttr = getattr ( _sharedLibAccess, _inifunc, None)
        if _libFuncAttr == None:
            msg = 'could not find function: ' + _inifunc
            raise RuntimeError(msg)
        _libFuncAttr ()

        ##  Call the library function, normalize Spiral output
        print ( 'Function name: ' + _invfunc )     ##  'Using Library = ' + _sharedLibPath
        _libFuncAttr = getattr ( _sharedLibAccess, _invfunc )
        _libFuncAttr ( _dst_spiral.ctypes.data_as ( ctypes.c_void_p ),
                       _src.ctypes.data_as ( ctypes.c_void_p ) )
        _dst_spiral = _dst_spiral / _N   ## np.size ( _dst_spiral  )
        
        ##  Check difference
        _diff = np.max ( np.absolute ( _dst_spiral - _dst_python_IL ) )
        print ( 'Difference between Python / Spiral transforms = ' + str ( _diff ) )



## ------------------------  Additional debug if needed
        # if ( _N < 40):
        #     print ( 'Python output = ' )
        #     print ( _dst_python )
        
        # if ( _N < 40):
        #     print ( 'Spiral output = ' )
        #     print ( _dst_spiral )
