##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

cmake_minimum_required ( VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION} )

project ( examples )

##  All examples are expected to support CPU code generation, and ultimately
##  most (if not all) will support GPU code generation also.  Call
##  manage_add_subdir with the directory name and True/False for building for
##  CPU and GPU

##                  subdir name   CPU       GPU

manage_add_subdir ( test_plan_dft TRUE      TRUE )
manage_add_subdir ( hockney       TRUE      FALSE )
manage_add_subdir ( rconv         TRUE      TRUE )
manage_add_subdir ( verify        TRUE      TRUE )
manage_add_subdir ( warpx         TRUE      FALSE )
