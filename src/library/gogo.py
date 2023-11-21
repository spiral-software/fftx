#! python

import sys
import subprocess
import os, stat
import re
import shutil
import argparse


def parse_platform(value):
    if re.match ( 'hip', value, re.IGNORECASE ):
        return 'HIP'
    if re.match ( 'cpu', value, re.IGNORECASE ):
        return 'CPU'
    if re.match ( 'cuda', value, re.IGNORECASE ):
        return 'CUDA'
    if re.match ( 'sycl', value, re.IGNORECASE ):
        return 'SYCL'
    return 'HIP'


def main():
    parser = argparse.ArgumentParser (
        description = 'Build FFTX library code with Spiral and transform specifications',
        usage = '%(prog)s -t TRANSFORM -s SIZES_FILE -p {CPU,CUDA,HIP,SYCL} [-i] [-m SIZES_MASTER]'
    )
    ##  Required arguments: <transform> <sizes-file> <platform>
    reqd_group = parser.add_argument_group ( 'required arguments' )
    reqd_group.add_argument ( '-t', '--transform', type=str, required=True,
                              help='transform to use use for building the library' )
    reqd_group.add_argument ( '-s', '--sizes_file', type=str, required=True,
                              help='filename containing the sizes to build' )
    reqd_group.add_argument ( '-p', '--platform',  type=parse_platform, nargs='?', default='HIP',
                              choices=['CPU', 'CUDA', 'HIP', 'SYCL'], required=True,
                              help='Platform: CPU or GPU [{CUDA | HIP | SYCL}, default = HIP]' )
    
    ##  Optional arguments: <direction> <sizes-master>
    parser.add_argument ( '-i', '--inverse', action='store_true',
                          help='False [default], run forward transform; when specified run Inverse transform' )
    parser.add_argument ( '-m', '--sizes_master', type=str,
                          help='Master sizes filename; Regenerate headers & API files [uses existing code files] for the library' )
    args = parser.parse_args()

    ##  Add a dictionary to hold platform to file extension mapping
    plat_to_file_suffix = {
        'CPU':  '.cpp',
        'CUDA': '.cu',
        'HIP':  '.cpp',
        'SYCL': '.cpp'
    }

    args.file_suffix = plat_to_file_suffix.get ( args.platform, '.cpp' )            ## default to '.cpp'
    args.regen = True if args.sizes_master is not None else False
    
    ##  Print the options selected
    print ( f'Generate files for:\nTransform:\t{args.transform}\nSizes file:\t{args.sizes_file}\nPlatform:\t{args.platform}' )
    dirt = 'Inverse' if args.inverse else 'Forward'
    print ( f'File suffix:\t{args.file_suffix}\nDirection:\t{dirt}\nRegen library:\t{args.regen}' )
    if args.regen:
        print ( f'Master sizes:\t{args.sizes_master}' )


if __name__ == '__main__':
    main()
