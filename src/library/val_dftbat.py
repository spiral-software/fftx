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


_under = '_'

def setup_input_array ( array_shape ):
    "Create an input array based on the array shape tuple"

    ##  Setup source (input) data -- fill with random values
    src = np.random.random(array_shape) + np.random.random(array_shape) * 1j
    return src;


def run_python_version ( src, styp, fwd ):
    "Create the output array by calling python"

    ##  Stride type 1 and 3 read sequentially (2 & 4 read strided)
    ax = -1 if styp in (1, 3) else 0

    if fwd:
        dst = np.fft.fft ( src, axis=ax )
    else:
        dst = np.fft.ifft ( src, axis=ax )
 
    if styp in (2, 3):
        ##  read & write strides are different (i.e., read seq/write stride or read stride/write seq)
        if styp == 3:
            ##  read sequential, write strided
            ##  new shape is inverse of src
            revdims = src.shape[::-1]
            tmp = dst.reshape(np.asarray(revdims[1:]).prod(), revdims[0]).transpose()
            dst = tmp.reshape(revdims)
        else:
            ##  read strided, write sequential
            samedims = dst.shape
            tmp = dst.reshape(samedims[0], np.asarray(samedims[1:]).prod()).transpose()
            dst = tmp.reshape(samedims[::-1])

    return dst;


def exec_xform ( libdir, libfwd, libinv, libext, dims, fwd, platform, typecode ):
    "Run a transform specified by segment names and fwd flag of size dims"

    ##  First, ensure the library we need exists...
    uselib = libfwd + platform + libext if fwd else libinv + platform + libext
    _sharedLibPath = os.path.join ( os.path.realpath ( libdir ), uselib )
    if not os.path.exists ( _sharedLibPath ):
        print ( f'library file: {uselib} does not exist - continue', flush = True )
        return

    nbat = dims[0]
    dz   = dims[1]
    styp = dims[2]

    ##  Stride type 1 and 3 read sequentially (2 & 4 read strided)
    array_shape = ( nbat, 1, dz ) if styp in (1, 3) else ( dz, nbat, 1 )
    _src = setup_input_array ( array_shape )

    ##  Setup function name and specify size to find in library
    if sys.platform == 'win32':
        froot = libfwd if fwd else libinv
    else:
        froot = libfwd[3:] if fwd else libinv[3:]

    pywrap = froot + platform + _under + 'python' + _under

    _xfmsz    = np.zeros(3).astype(ctypes.c_int)
    _xfmsz[0] = nbat
    _xfmsz[1] = dz
    _xfmsz[2] = styp

    ##  Evaluate using Spiral generated code in library.  Use the python wrapper funcs in
    ##  the library (these setup/teardown GPU resources when using GPU libraries).
    _sharedLibAccess = ctypes.CDLL ( _sharedLibPath )

    func = pywrap + 'init' + _under + 'wrapper'
    _libFuncAttr = getattr ( _sharedLibAccess, func, None)
    if _libFuncAttr is None:
        msg = 'could not find function: ' + func
        raise RuntimeError(msg)
    _status = _libFuncAttr ( _xfmsz.ctypes.data_as ( ctypes.c_void_p ) )
    if not _status:
        print ( f'Size: {_xfmsz} was not found in library - continue', flush = True )
        return

    ##  If the strides for read & write are different then reverse the dimensions for the output array.
    if styp in (2, 3):
        ##  reverse dims
        revdims = _src.shape[::-1]
        _dst_spiral = np.zeros ( revdims, _src.dtype )
    else:
        _dst_spiral = np.zeros_like( _src )

    ##  Call the library function
    func = pywrap + 'run' + _under + 'wrapper'
    try:
        _libFuncAttr = getattr ( _sharedLibAccess, func )
        _libFuncAttr ( _xfmsz.ctypes.data_as ( ctypes.c_void_p ),
                       _dst_spiral.ctypes.data_as ( ctypes.c_void_p ),
                       _src.ctypes.data_as ( ctypes.c_void_p ) )
    except Exception as e:
        print ( 'Error occurred during library function call:', type(e).__name__ )
        print ( 'Exception details:', str(e) )
        return

    ##  Call the transform's destroy function
    func = pywrap + 'destroy' + _under + 'wrapper'
    _libFuncAttr = getattr ( _sharedLibAccess, func )
    _libFuncAttr ( _xfmsz.ctypes.data_as ( ctypes.c_void_p ) )

    ##  Get the python answer using the same input
    _dst_python = run_python_version ( _src, styp, fwd )

    ##  Check difference
    _diff = np.max ( np.absolute ( _dst_spiral - _dst_python ) )
    dir = 'forward' if fwd else 'inverse'
    msg = 'are' if _diff < 1e-7 else 'are NOT'
    print ( f'Python / Spiral({typecode}) [{dir}] transforms {msg} equivalent, difference = {_diff}', flush = True )

    return;


def main():
    parser = argparse.ArgumentParser ( description = 'Validate FFTX built libraries against NumPy computed versions of the transforms' )
    ##  Positional argument: libdir
    parser.add_argument ( 'libdir', type=str, help='directory of the library' )
    ##  Optional arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument ( '-s', '--size', type=str, help='size of the transform (e.g., 16x80x2)' )
    group.add_argument ( '-f', '--file', type=str, help='file containing sizes to loop over' )
    args = parser.parse_args()

    script_name = os.path.splitext ( os.path.basename ( sys.argv[0] ) )[0]
    if script_name.startswith('val_'):
        xfmseg = script_name.split('_')[1]
        xform = xfmseg

    libdir = args.libdir.rstrip('/')
    libprefix = '' if sys.platform == 'win32' else 'lib'
    libfwd = libprefix + 'fftx_' + xform
    libinv = libprefix + 'fftx_i' + xform
    libext = '.dll' if sys.platform == 'win32' else '.dylib' if sys.platform == 'darwin' else '.so'
    print ( f'library stems for fwd/inv xforms = {libfwd} / {libinv} lib ext = {libext}', flush = True )

    if args.size:
        dims = [int(d) for d in args.size.split('x')]
        print ( f'Size = {dims[0]} (batches) x {dims[1]} (length) x {dims[2]} (stride type)', flush = True )
        exec_xform ( libdir, libfwd, libinv, libext, dims, True, '_cpu', 'CPU' )
        exec_xform ( libdir, libfwd, libinv, libext, dims, False, '_cpu', 'CPU' )

        exec_xform ( libdir, libfwd, libinv, libext, dims, True, '_gpu', 'GPU' )
        exec_xform ( libdir, libfwd, libinv, libext, dims, False, '_gpu', 'GPU' )

        sys.exit ()

    if args.file:
        sizesfile = args.file.rstrip()
    else:
        sizesfile = 'dftbatch-sizes.txt'

    ##  Process the sizes file, extracting the dimentions and running the transform.  The
    ##  sizes file contains records of the form:
    ##      nbatch := 16; szns := 60; stridetype := 1;
    ##  where nbatch is the batch size, szns is the DFT length, and stridetype inidcates if
    ##  input, output, or both are read/written sequentially or strided.  Stride type can take
    ##  value 1 - 4, with the following meaning:
    ##      1:      Read sequantial, write sequential
    ##      2:      Read strided, write sequential
    ##      3:      Read sequantial, write strided
    ##      4:      Read strided, write strided

    with open ( sizesfile, 'r' ) as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            line = re.sub ( ' ', '', line )                 ## suppress white space
            m = re.match ( r'nbatch:=(\d+);szns:=(\d+);stridetype:=(\d+);', line)
            if not m:
                print ( f'Invalid line format: {line}', flush = True )
                continue

            dims = [int(m.group(i)) for i in range(1, 4)]
            print ( f'Size = {dims[0]} (batches) x {dims[1]} (length) x {dims[2]} (stride type)', flush = True )
            exec_xform ( libdir, libfwd, libinv, libext, dims, True, '_cpu', 'CPU' )
            exec_xform ( libdir, libfwd, libinv, libext, dims, False, '_cpu', 'CPU' )

            exec_xform ( libdir, libfwd, libinv, libext, dims, True, '_gpu', 'GPU' )
            exec_xform ( libdir, libfwd, libinv, libext, dims, False, '_gpu', 'GPU' )

if __name__ == '__main__':
    main()
