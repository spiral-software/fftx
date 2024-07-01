#! python

##  Validate FFTX built libraries against numpy computed versions of the transforms.
##  Exercise all the sizes in the library (read cube-sizes file) and call both forward and
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


def run_python_version ( src, fwd ):
    "Create the output array by calling NumPy"

    if fwd:
        dst = np.fft.fftn ( src )
        dst = dst / np.size ( dst )             ##  Normalize after performing forward transform
    else:
        dst = np.fft.ifftn ( src )              ##  NumPy inverse is already normalized
 
    return dst;


def exec_xform ( libdir, libfwd, libinv, libext, dims, fwd, platform, typecode, warn ):
    "Run a transform from the requested library of size dims"

    ##  First, ensure the library we need exists...
    uselib = libfwd + platform + libext if fwd else libinv + platform + libext
    _sharedLibPath = os.path.join ( os.path.realpath ( libdir ), uselib )
    if not os.path.exists ( _sharedLibPath ):
        if warn:
            print ( f'library file: {uselib} does not exist - continue', flush = True )
        return

    array_shape = tuple(dims)
    src = setup_input_array ( array_shape )

    ##  Setup function name and specify size to find in library
    if sys.platform == 'win32':
        froot = libfwd if fwd else libinv
    else:
        froot = libfwd[3:] if fwd else libinv[3:]

    pywrap = froot + platform + _under + 'python' + _under

    _xfmsz    = np.zeros(3).astype(ctypes.c_int)
    _xfmsz[0] = dims[0]
    _xfmsz[1] = dims[1]
    _xfmsz[2] = dims[2]

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

    ##  Destination (output) -- fill with zeros
    _dst_spiral = np.zeros_like ( src )
    _xfm_sym    = np.ones ( shape=(10, 10, 10), dtype=complex )       ## dummy symbol

    ##  Call the library function
    func = pywrap + 'run' + _under + 'wrapper'
    try:
        _libFuncAttr = getattr ( _sharedLibAccess, func )
        _libFuncAttr ( _xfmsz.ctypes.data_as ( ctypes.c_void_p ),
                       _dst_spiral.ctypes.data_as ( ctypes.c_void_p ),
                       src.ctypes.data_as ( ctypes.c_void_p ),
                       _xfm_sym.ctypes.data_as ( ctypes.c_void_p ) )
        ##  Normalize Spiral result
        _dst_spiral = _dst_spiral / np.size ( _dst_spiral )

    except Exception as e:
        print ( 'Error occurred during library function call:', type(e).__name__ )
        print ( 'Exception details:', str(e) )
        return

    ##  Call the transform's destroy function
    func = pywrap + 'destroy' + _under + 'wrapper'
    _libFuncAttr = getattr ( _sharedLibAccess, func )
    _libFuncAttr ( _xfmsz.ctypes.data_as ( ctypes.c_void_p ) )

    ##  Get the python answer using the same input
    _dst_python = run_python_version ( src, fwd )

    ##  Check difference
    diff = np.max ( np.absolute ( _dst_spiral - _dst_python ) )
    dir = 'forward' if fwd else 'inverse'
    msg = 'are' if diff < 1e-7 else 'are NOT'
    print ( f'Python / Spiral({typecode}) [{dir}] transforms {msg} equivalent, difference = {diff}', flush = True )

    return;


def main():
    parser = argparse.ArgumentParser ( description = 'Validate FFTX built libraries against NumPy computed versions of the transforms' )
    ##  Positional argument: libdir
    parser.add_argument ( 'libdir', type=str, help='directory containing the library' )
    ##  Optional argument: -e or --emit
    parser.add_argument ( '-e', '--emit', action='store_true', help='emit warnings when True, default is False' )
    ##  mutually exclusive optional arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument ( '-s', '--size', type=str, help='3D size specification of the transform (e.g., 80x80x120)' )
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
        print ( f'Size = {dims[0]} x {dims[1]} x {dims[2]}', flush = True )
        exec_xform ( libdir, libfwd, libinv, libext, dims, True, '_cpu', 'CPU', args.emit )
        exec_xform ( libdir, libfwd, libinv, libext, dims, False, '_cpu', 'CPU', args.emit )

        exec_xform ( libdir, libfwd, libinv, libext, dims, True, '_gpu', 'GPU', args.emit )
        exec_xform ( libdir, libfwd, libinv, libext, dims, False, '_gpu', 'GPU', args.emit )

        sys.exit ()

    if args.file:
        sizesfile = args.file.rstrip()
    else:
        sizesfile = 'cube-sizes-gpu.txt'

    ##  Process the sizes file, extracting the dimentions and running the transform.  The
    ##  sizes file contains records of the form:
    ##      szcube := [ 128, 128, 128 ];
    ##  where:
    ##      the X, Y, and Z dimensions are specified (dimension need not be cubic)

    with open ( sizesfile, 'r' ) as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            line = re.sub ( ' ', '', line )                 ## suppress white space
            pattern = r'szcube:=\[(\d+),(\d+),(\d+)\];'
            m = re.match ( pattern, line)
            if not m:
                print ( f'Invalid line format: {line}', flush = True )
                continue

            dims = [int(m.group(i)) for i in range(1, 4)]
            print ( f'Size = {dims[0]} x {dims[1]} x {dims[2]}', flush = True )
            exec_xform ( libdir, libfwd, libinv, libext, dims, True, '_cpu', 'CPU', args.emit )
            exec_xform ( libdir, libfwd, libinv, libext, dims, False, '_cpu', 'CPU', args.emit )

            exec_xform ( libdir, libfwd, libinv, libext, dims, True, '_gpu', 'GPU', args.emit )
            exec_xform ( libdir, libfwd, libinv, libext, dims, False, '_gpu', 'GPU', args.emit )

if __name__ == '__main__':
    main()

