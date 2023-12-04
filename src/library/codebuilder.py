##  codebuilder.py
##  Define a class to hold code generated/added while building a library

##  Copyright (c) 2018-2023, Carnegie Mellon University
##  See LICENSE for details

class CodeBuilder:
    """A class to hold code items generated while building source code for an FFTX libray

    Attributes:
        code_str (str):         The code string built up as processing continues.
    """

    def __init__ ( self, init="" ):
        """Initialize an empty string to hold generated code items."""
        self.code_str = [init]

    def append ( self, code ):
        self.code_str.append ( code )

    def get ( self ):
        return ''.join ( self.code_str )

    def erase_last ( self, n ):
        """Erase the last N characters from the last code string."""
        if n >= 0:
            self.code_str[-1] = self.code_str[-1][:-n]

