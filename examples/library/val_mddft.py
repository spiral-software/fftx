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

##  open the right file of sizes: mddft | mdprdft ==> cube-sizes.txt

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
        _dimx = _dims[0]
        _dimy = _dims[1]
        _dimz = _dims[2]
        _dx = int ( _dimx )
        _dy = int ( _dimy )
        _dz = int ( _dimz )
        
        _funcname = _xfmseg[0] + _under + _xfmseg[1] + _under + _dimx + 'x' + _dimy + 'x' + _dimz
        _pystemfw = _xfmseg[0] + _under + _xfmseg[1] + _under + 'python' + _under
        _invfunc  = _xfmseg[0] + _under + 'i' + _xfmseg[1] + _under + _dimx + 'x' + _dimy + 'x' + _dimz
        _pystemin = _xfmseg[0] + _under + 'i' + _xfmseg[1] + _under + 'python' + _under
        print ( 'Size = ' + _dimx + 'x' + _dimy + 'x' + _dimz )

        ##  Setup source (input) data -- fill with random values
        _src        = np.random.rand(_dx, _dy, _dz).astype(complex)
        _src_spiral = _src.view(dtype=np.double)
        for ix in range ( _dx ):
            for iy in range ( _dy ):
                for iz in range ( _dz ):
                    vr = np.random.random()
                    vi = np.random.random()
                    _src[ix, iy, iz] = vr + vi * 1j

        ##  Destination (output) -- fill with zeros, one for spiral, one for python
        _dst_python = np.zeros(shape=(_dx, _dy, _dz), dtype=complex)
        _dst_spiral = np.zeros(shape=(_dx, _dy, _dz), dtype=complex)
        _xfm_sym    = np.ones (shape=(10, 10, 10), dtype=complex)       ## dummy symbol
        _dftsz      = np.zeros(3).astype(ctypes.c_int)
        _dftsz[0] = _dx
        _dftsz[1] = _dy
        _dftsz[2] = _dz
        print ( _dftsz )
        
        ##  Evaluate using numpy (forward transform first) ... 
        _dst_python    = np.fft.fftn ( _src )
        _dst_python = _dst_python / np.size ( _dst_python )
        
        ##  Access library and call Spiral generated code... we use the python wrapper funcs in the library
        _sharedLibPath = os.path.join ( os.path.realpath ( _libdir ), _libfwd )
        _sharedLibAccess = ctypes.CDLL ( _sharedLibPath )
        ##  Call the SPIRAL generated init function
        _inifunc = _pystemfw + 'init' + _under + 'wrapper'
        print ( 'Python wrapper init func = ' + _inifunc )

        _libFuncAttr = getattr ( _sharedLibAccess, _inifunc, None)
        if _libFuncAttr == None:
            msg = 'could not find function: ' + _inifunc
            raise RuntimeError(msg)
        _libFuncAttr ( _dftsz.ctypes.data_as ( ctypes.c_void_p ) )
        
        ##  Call the library function
        _funcname = _pystemfw + 'run' + _under + 'wrapper'
        print ( 'Function name: ' + _funcname )   ## 'Using Library = ' + _sharedLibPath
        _libFuncAttr = getattr ( _sharedLibAccess, _funcname )
        _libFuncAttr ( _dftsz.ctypes.data_as ( ctypes.c_void_p ),
                       _dst_spiral.ctypes.data_as ( ctypes.c_void_p ),
                       _src.ctypes.data_as ( ctypes.c_void_p ),
                       _xfm_sym.ctypes.data_as ( ctypes.c_void_p ) )
        _dst_spiral = _dst_spiral / np.size ( _dst_spiral )
        
        ##  Call the transform's destroy function
        _funcname = _pystemfw + 'destroy' + _under + 'wrapper'
        _libFuncAttr = getattr ( _sharedLibAccess, _funcname )
        _libFuncAttr ( _dftsz.ctypes.data_as ( ctypes.c_void_p ) )
        
        ##  Check difference
        _diff = np.max ( np.absolute ( _dst_spiral - _dst_python ) )
        print ( 'Difference between Python / Spiral transforms = ' + str ( _diff ), flush = True )
        ## sys.stdout.flush()

        ##  Setup and repeat for the inverse transform
        ##  Destination (output) -- fill with zeros, one for spiral, one for python
        _dst_python = np.zeros(shape=(_dx, _dy, _dz), dtype=complex)
        _dst_spiral = np.zeros(shape=(_dx, _dy, _dz), dtype=complex)
        
        ##  Evaluate using numpy ... 
        _dst_python = np.fft.ifftn ( _src )
        ##  _dst_python = _dst_python / np.size ( _dst_python )

        ##  Access library and call Spiral generated code... use the python wrapper funcs in the library
        _sharedLibPath = os.path.join ( os.path.realpath ( _libdir ), _libinv )
        _sharedLibAccess = ctypes.CDLL ( _sharedLibPath )
        ##  Call the SPIRAL generated init function
        _inifunc = _pystemin + 'init' + _under + 'wrapper'
        print ( 'Python wrapper init func = ' + _inifunc )
        
        _libFuncAttr = getattr ( _sharedLibAccess, _inifunc, None)
        if _libFuncAttr == None:
            msg = 'could not find function: ' + _inifunc
            raise RuntimeError(msg)
        _libFuncAttr ( _dftsz.ctypes.data_as ( ctypes.c_void_p ) )

        ##  Call the library function, normalize Spiral output
        _funcname = _pystemin + 'run' + _under + 'wrapper'
        print ( 'Function name: ' + _funcname )   ## 'Using Library = ' + _sharedLibPath
        _libFuncAttr = getattr ( _sharedLibAccess, _funcname )
        _libFuncAttr ( _dftsz.ctypes.data_as ( ctypes.c_void_p ),
                       _dst_spiral.ctypes.data_as ( ctypes.c_void_p ),
                       _src.ctypes.data_as ( ctypes.c_void_p ),
                       _xfm_sym.ctypes.data_as ( ctypes.c_void_p ) )
        _dst_spiral = _dst_spiral / np.size ( _dst_spiral )
        
        ##  Check difference
        _diff = np.max ( np.absolute ( _dst_spiral - _dst_python ) )
        print ( 'Difference between Python / Spiral transforms = ' + str ( _diff ), flush = True )


## ---  extra debugging ...
        # print ( '_src type = ' + str(type(_src)) + ', shape = ' + str(_src.shape) + ', dtype = ' + str(_src.dtype) )
        # print ( '_src_spiral type = ' + str(type(_src_spiral)) + ', shape = ' + str(_src_spiral.shape) + ', dtype = ' + str(_src_spiral.dtype) )
        
        # print ( '_dst_spiral type = ' + str(type(_dst_spiral)) + ', shape = ' + str(_dst_spiral.shape) + ', dtype = ' + str(_dst_spiral.dtype) )
        # print ( '_dst_python type = ' + str(type(_dst_python)) + ', shape = ' + str(_dst_python.shape) + ', dtype = ' + str(_dst_python.dtype) )
        # print ( '_dst_python_IL type = ' + str(type(_dst_python_IL)) + ', shape = ' + str(_dst_python_IL.shape) + ', dtype = ' + str(_dst_python_IL.dtype) )
        
        # _dims = re.sub ( '.*nbatch :=', '', line )      ## get number batches
        # _dims = re.sub ( ';.*', '', _dims )
        # _dims = re.sub ( ' *', '', _dims )              ## compress out white space
        # _dims = _dims.rstrip()                          ## remove training newline
        # _nbat = _dims
        # _nbt  = int ( _nbat )
        # ##  if ( _nbt != 1 ):
        # ##      continue
        
        
        # _N = int ( _dims )
        # _NN = _N
        # if ( _xfmseg[1] == 'prdftbat' ):
        #     _NN = _N // 2 + 1
