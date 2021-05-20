#! python

##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

##  This script reads a file, process-sizes.txt, that contains several cube size
##  specifications for the 3D DFT.  The script will run the FFTX generation process for
##  each size, using CMake; the final executable will be renamed to append the size spec.

import os
import re
import shutil
import subprocess
import sys
import time

##  Run cmake and build to build the code.  CMake isn't being called from the top
##  level FFTX folder, so it won't know the FFTX Project source directory -- we'll
##  assume its 2 levels up from this [current] directory, define it as a variable
##  to pass to CMake

_cdir = os.getcwd()
_base = os.path.basename ( _cdir )
_fftx = os.path.join ( _cdir, '../..' )
_fftx = os.path.abspath (_fftx )
##  print ( 'CWD: ' + _cdir + '; Basename: ' + _base + '; FFTX Project dir: ' + _fftx )

##  CMake will normally run with ${_codegen} defined (if undefined it defaults to CPU)
##  Check for a command line parameter specifying ${_codegen}, default to CPU if missing

_mode = "CPU"                     ## define mode as CPU [default]
if len ( sys.argv ) < 2:
    print ( 'Usage: ' + sys.argv[0] + ' [ CPU | GPU | HIP ]; defaulting to CPU' )
else:
    _mode = sys.argv[1]
    if _mode != "CPU" and _mode != "GPU":
        if _mode == "HIP":
            print ( sys.argv[0] + ': HIP mode requested - support coming soon, defaulting to CPU' )
            _mode = "CPU"
        else:
            print ( sys.argv[0] + ': unknown mode: ' + _mode + ' exiting...' )
            sys.exit (-1)

##  Setup 'empty' timing script (would need bash or cygwin or similar to run on Windows)
_timescript = _mode + '-timescript.sh'
timefd = open ( _timescript, 'w' )
timefd.write ( '#! /bin/bash \n\n' )
timefd.write ( '##  Timing script to run the various transform sizes \n\n' )
timefd.close()
if sys.platform != 'win32':
    _filmode = stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    os.chmod ( _timescript, _filmode )

with open ( 'process-sizes.txt', 'r' ) as fil:
    for line in fil.readlines():
        ##  print ( 'Line read = ' + line )
        if re.match ( '[ \t]*#', line ):                ## ignore comment lines
            continue

        if re.match ( '[ \t]*$', line ):                ## skip lines consisting of whitespace
            continue
        
        line = re.sub ( '.*\[', '', line )               ## drop "szcube := ["
        line = re.sub ( '\].*', '', line )               ## drop "];"
        line = re.sub ( ' *', '', line )                 ## compress out white space
        line = line.rstrip()                             ## remove training newline
        dims = re.split ( ',', line )
        _dimx = dims[0]
        _dimy = dims[1]
        _dimz = dims[2]

        ##  Use these dimensions in defines passed on the CMake command line -- CMake in
        ##  turn will generate the needed C language defines to be used when compiling
        ##  Create the build directory (if it doesn't exist)
        build_dir = 'build'
        isdir = os.path.isdir ( build_dir )
        if not isdir:
            os.mkdir ( build_dir )

        ##  We'll keep executables we build in an 'executables' folder, create if necessary
        exec_dir = _mode + '-executables'
        isdir = os.path.isdir ( exec_dir )
        if not isdir:
            os.mkdir ( exec_dir )
        
        ##  Build the commnd line for CMake...
        cmdstr = 'rm -rf * && cmake -DFFTX_PROJ_DIR=' + _fftx + ' -D_codegen=' + _mode
        cmdstr = cmdstr + ' -DDIM_X=' + _dimx + ' -DDIM_Y=' + _dimy + ' -DDIM_Z=' + _dimz

        os.chdir ( build_dir )

        if sys.platform == 'win32':
            cmdstr = cmdstr + ' .. && cmake --build . --config Release --target install'
        else:
            cmdstr = cmdstr + ' .. && make install'

        ##  print ( cmdstr )
        ##  continue                    ## testing script, stop here

        result = subprocess.run ( cmdstr, shell=True, check=True )
        res = result.returncode

        if (res != 0):
            print ( result )
            sys.exit ( res )

        ##  CMake has built the example [_target] and placed it in the ./bin folder
        ##  Rename the executable to have a cube size spec on the end and move to
        ##  ../exeutables folder
        
        _suffix = '-' + _dimx + 'x' + _dimy + 'x' + _dimz
        _target = 'bin/test' + _base
        _newloc = exec_dir + '/test' + _base + _suffix
        if sys.platform == 'win32':
            _target = _target + '.exe'
            _newloc = _newloc + '.exe'

        ##  print ( 'Target built: ' + _target + ' Move to: ' + _newloc )
        shutil.copy2 ( _target, '../' + _newloc )

        os.chdir ( '..' )

        timefd = open ( _timescript, 'a' )
        timefd.write ( '##  Cube = [' + _dimx + ', ' + _dimy + ', ' + _dimz + ' ]\n' )
        timefd.write ( './' + _newloc +  '\n\n' )
        timefd.close()

        time.sleep(1)

sys.exit (0)


