/**
 *  Copyright (c) 2018-2025, Carnegie Mellon University
 *  See LICENSE for details
 *
 *  @file doc_templates.hpp
 *  @brief Explicit FFTX template wrappers for Doxygen visibility.
 *         This file is used **only** during documentation build and not included in actual code.
 */

#pragma once

#ifdef FFTX_DOXYGEN

#include "fftx.hpp"

namespace fftx {

/// \brief 3-dimensional specialization of point_t.
/// \ingroup api
/// \copydoc fftx::point_t<3>
struct point3 : public point_t<3> { };

/// \brief 3-dimensional specialization of box_t.
/// \ingroup api
/// \copybrief fftx::box_t
/// \copydetails fftx::box_t
struct box3 : public box_t<3> { };

/// \brief 3-dimensional array of doubles (array_t<3, double>).
/// \ingroup api
/// \copybrief fftx::array_t
/// \copydetails fftx::array_t
struct array3d : public array_t<3, double> { };

/// \brief Global pointer to double (global_ptr<double>).
/// \ingroup api
/// \copybrief fftx::global_ptr
/// \copydetails fftx::global_ptr
struct globalptrd : public global_ptr<double> { };

/// \cond FFTX_DOXYGEN_METHODS

// -- Doxygen-only method stubs to enable @copydoc and @copydetails references --

template<int DIM>
void point_t<DIM>::operator=(int a_default) { }

template<int DIM>
bool point_t<DIM>::operator==(const point_t<DIM>& a_rhs) const { return true; }

template<int DIM>
point_t<DIM> point_t<DIM>::operator*(int scale) const { return *this; }

template<int DIM>
int point_t<DIM>::product() { return 1; }

template<int DIM>
point_t<DIM - 1> point_t<DIM>::project() const { return point_t<DIM - 1>(); }

template<int DIM>
point_t<DIM - 1> point_t<DIM>::projectC() const { return point_t<DIM - 1>(); }

template<int DIM>
point_t<DIM> point_t<DIM>::Unit() { return point_t<DIM>(); }

template<int DIM>
point_t<DIM> point_t<DIM>::Zero() { return point_t<DIM>(); }

/// \endcond

} // namespace fftx

#endif // FFTX_DOXYGEN
