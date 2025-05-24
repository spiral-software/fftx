##
##  Copyright (c) 2018-2025, Carnegie Mellon University
##  All rights reserved.
##
##  See LICENSE file for full information.
##

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import subprocess, os
import re
import datetime


def extract_fftx_version ( header_file_path ):
    """Extract the FFTX version from the specified header file."""
    if not os.path.exists ( header_file_path ):
        print ( "Header file does not exist at specified path" )

    try:
        with open ( header_file_path, "r" ) as file:
            content = file.read()
            match = re.search ( r"FFTX_VERSION\s+([\d\.]+)", content )
            if match:
                return match.group(1)
    except Exception as e:
        print ( f"Error extracting FFTX version: {e}" )
    return "Unknown Version"

# The FFTX version is defined in .../src/include/fftx.hpp
header_file_path = os.path.abspath ( os.path.normpath ( os.path.join ( os.path.dirname(__file__),
                                                                       "../src/include/fftx.hpp" ) ) )
print ( f"Resolved header file path: {header_file_path}" )
fftx_version = extract_fftx_version ( header_file_path )

# Get current time and format a 'docs generated on' message

current_date = datetime.datetime.now().strftime("%B %d, %Y")

# Check if we're running on Read the Docs' servers
read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'

breathe_projects = {}
if read_the_docs_build:
	input_dir = '../include'
	output_dir = 'build'
	configureDoxyfile(input_dir, output_dir)
	subprocess.call('doxygen', shell=True)
	breathe_projects['FFTX'] = output_dir + '/xml'


# -- Project information -----------------------------------------------------

project = 'FFTX'
copyright = f"2025, FFTX Team.     Version: {fftx_version};     Documentation generated on {current_date}"
author = 'FFTX Team'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [ "breathe", "sphinx.ext.graphviz" ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Breathe Configuration
breathe_default_project = 'FFTX'
# breathe_default_members = ('members', 'protected-members', 'private-members')
# breathe_default_members = ('members')
breathe_default_members = ()
breathe_show_include = False
