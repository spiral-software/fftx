##  script.py
##  Define a simple class to hold items used when generating the library code for an FFTX library

##  Copyright (c) 2018-2023, Carnegie Mellon University
##  See LICENSE for details

import argparse

class Script:
    """A class to hold items needed for generating FFTX library code

    Attributes:
        xform_name (str):       The name of the transform.
        sp_type (str):          The corresponding transform type in SpiralPy (used for metadata).
        file_stem (str):        The stem used for creating file names (e.g., 'fftx_mddft_').
        orig_file_stem (str):   The original file stem (from the Spiral frame code file).
        decor_platform (str):   The platform name 'decoration' (e.g., 'cpu_' or 'gpu_').
        srcs_dir (str):         The library sources directory.
        func_stem (str):        The qualified function name stem (e.g., 'fftx_mddft_64x64x64_HIP')
        file_name (str):        The file name of the source code file to generate.
        src_file_path (str):    The fully qualified path of the filename.
        frame_file (str):       The transform framework file (used to generate the Spiral script).
        file_suffix (str):      The file suffix to use with the generated code.
        regen (boolean):        Whether or not to regenerate the API & header files.
        args (argparse.Namespace): Parsed command-line arguments.
    """

    def __init__ ( self, args ):
        """Initialize a Script instance with default attribute values."""
        self.xform_name = None
        self.sp_type = None
        self.file_stem = None
        self.orig_file_stem = None
        self.decor_platform = None
        self.srcs_dir = None
        self.func_stem = None
        self.file_name = None
        self.src_file_path = None
        self.frame_file = None
        self.file_suffix = None
        self.regen = False
        self.args = args

