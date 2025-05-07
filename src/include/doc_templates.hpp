//  Copyright (c) 2018-2025, Carnegie Mellon University
//  See LICENSE for details
//
//  doc_templates.hpp
//  This file exists only to help Doxygen generate documentation
//  for FFTX template classes that would otherwise be skipped.
//  It is never used or compiled into code.

#pragma once

#include "fftx.hpp"

namespace fftx {

    // Force template instantiations for documentation
    template struct point_t<3>;
    template struct box_t<3>;
    template struct array_t<3, double>;
    template struct global_ptr<3, double>;

} // namespace fftx
