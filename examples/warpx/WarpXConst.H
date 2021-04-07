/* Copyright 2019 Andrew Myers, Luca Fedeli, Maxence Thevenet
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_CONST_H_
#define WARPX_CONST_H_

//#include <AMReX_REAL.H>
//typedef amrex::Real T
typedef double T;

#include <cmath>

// Math constants
namespace MathConst
{
    static constexpr T pi = 3.14159265358979323846;
}

// Physical constants. Values are the 2018 CODATA recommended values
// https://physics.nist.gov/cuu/Constants/index.html
namespace PhysConst
{
    static constexpr T c     = 299'792'458.;
    static constexpr T ep0   = 8.8541878128e-12;
    static constexpr T mu0   = 1.25663706212e-06;
    static constexpr T q_e   = 1.602176634e-19;
    static constexpr T m_e   = 9.1093837015e-31;
    static constexpr T m_p   = 1.67262192369e-27;
    static constexpr T hbar  = 1.054571817e-34;
    static constexpr T alpha = mu0/(4*MathConst::pi)*q_e*q_e*c/hbar;
    static constexpr T r_e   = 1./(4*MathConst::pi*ep0) * q_e*q_e/(m_e*c*c);
    static constexpr double xi         = (2.*alpha*alpha*ep0*ep0*hbar*hbar*hbar)/
                                         (45.*m_e*m_e*m_e*m_e*c*c*c*c*c); // SI value is 1.3050122.e-52
    static constexpr T xi_c2 = xi * c * c; // This should be usable for single precision, though
    // very close to smallest number possible: smallest number = 1.2e-38, xi_c2 = 1.1e-35
}

#endif
