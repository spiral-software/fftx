!!
!!  Copyright (c) 2018-2025, Carnegie Mellon University
!!  All rights reserved.
!!
!!  See LICENSE file for full information.
!!

module data_functions_mod
  use problem_dimensions_mod
  use, intrinsic :: iso_c_binding
  implicit none
  
contains

  complex(C_DOUBLE_COMPLEX) function inputComplexFun(ix, iy, iz)
    implicit none
    integer, intent(in) :: ix, iy, iz
    inputComplexFun = exp( &
         (0._C_DOUBLE,  7._C_DOUBLE) * ix / real(nx_global, C_DOUBLE) + &
         (0._C_DOUBLE, -5._C_DOUBLE) * iy / real(ny_global, C_DOUBLE) + &
         (0._C_DOUBLE, -3._C_DOUBLE) * iz / real(nz_global, C_DOUBLE) )
  end function inputComplexFun

  real(C_DOUBLE) function inputRealFun(ix, iy, iz)
    implicit none
    integer, intent(in) :: ix, iy, iz
    inputRealFun = cos( &
         ( 7._C_DOUBLE) * ix / real(nx_global, C_DOUBLE) + &
         (-5._C_DOUBLE) * iy / real(ny_global, C_DOUBLE) + &
         ( 3._C_DOUBLE) * iz / real(nz_global, C_DOUBLE) )
  end function inputRealFun

  real(C_DOUBLE) function inputRealPoisson(ix, iy, iz)
    implicit none
    integer, intent(in) :: ix, iy, iz
    real(C_DOUBLE) :: centerx, centery, centerz, radius2
    real(C_DOUBLE) :: dist2
    integer :: n_shortest

    n_shortest = min(nx_global, ny_global, nz_global)
    radius2 = (n_shortest * n_shortest) * 1._C_DOUBLE / 8._C_DOUBLE
    centerx = ((1 + nx_global) * 1._C_DOUBLE) / 2._C_DOUBLE
    centery = ((1 + ny_global) * 1._C_DOUBLE) / 2._C_DOUBLE
    centerz = ((1 + nz_global) * 1._C_DOUBLE) / 2._C_DOUBLE
    dist2 = (ix - centerx)**2 + (iy - centery)**2 + (iz - centerz)**2
    if (dist2 .lt. radius2) then
       ! For periodicity, need sum over all points to be zero.
       inputRealPoisson = ix - centerx
    else
       ! Outside the circle
       inputRealPoisson = 0._C_DOUBLE
    endif
    ! DEBUG:  this works OK too.
    inputRealPoisson = 2.*(ix - centerx) * (3.*(iy - centery)**2 + 5.*cos(iz - centerz))
  end function inputRealPoisson

  real(C_DOUBLE) function inputRealMessy(ix, iy, iz)
    implicit none
    integer, intent(in) :: ix, iy, iz

    inputRealMessy = sin(ix * 1._C_DOUBLE) * (3._C_DOUBLE * (iy**2) + 5._C_DOUBLE * cos(iz * 1._C_DOUBLE))
  end function inputRealMessy

  real(C_DOUBLE) function inputRealSymmetric(ix, iy, iz)
    implicit none
    integer, intent(in) :: ix, iy, iz
    real(C_DOUBLE) :: center

    center = ((1 + nx_global) * 1._C_DOUBLE) / 2._C_DOUBLE
    inputRealSymmetric = sin(ix * 1._C_DOUBLE - center) * (3._C_DOUBLE * (iy**2) + 5._C_DOUBLE * cos(iz * 1._C_DOUBLE))
  end function inputRealSymmetric

  real(C_DOUBLE) function testfun0real(k, n)
    implicit none
    integer, intent(in) :: k, n

    if (k .eq. 0) then
       testfun0real = 1.5_C_DOUBLE
    elseif (2*k .eq. n) then
       testfun0real = 2.2_C_DOUBLE
    elseif (2*k .lt. n) then
       testfun0real = sin(k * 1._C_DOUBLE)
    elseif (2*k .gt. n) then
       testfun0real = sin((n - k) * 1._C_DOUBLE)
    endif
  end function testfun0real
  
  real(C_DOUBLE) function testfun1real(k, n)
    implicit none
    integer, intent(in) :: k, n

    if (k .eq. 0) then
       testfun1real = -0.9_C_DOUBLE
    elseif (2*k .eq. n) then
       testfun1real = 1.3_C_DOUBLE
    elseif (2*k .lt. n) then
       testfun1real = cos(k * 1._C_DOUBLE)
    elseif (2*k .gt. n) then
       testfun1real = sin((n - k) * 1._C_DOUBLE)
    endif
  end function testfun1real
  
  real(C_DOUBLE) function testfun2real(k, n)
    implicit none
    integer, intent(in) :: k, n

    if (k .eq. 0) then
       testfun2real = 1.1_C_DOUBLE
    elseif (2*k .eq. n) then
       testfun2real = -0.7_C_DOUBLE
    elseif (2*k .lt. n) then
       testfun2real = exp(-abs(k * 1._C_DOUBLE))
    elseif (2*k .gt. n) then
       testfun2real = exp(-abs((n - k) * 1._C_DOUBLE))
    endif
  end function testfun2real
  
  real(C_DOUBLE) function testfun0imag(k, n)
    implicit none
    integer, intent(in) :: k, n

    if ((k .eq. 0) .or. (2*k .eq. n)) then
       testfun0imag = 0._C_DOUBLE
    elseif (2*k .lt. n) then
       testfun0imag = log((1 + k) * 1._C_DOUBLE)
    elseif (2*k .gt. n) then
       testfun0imag = -log((1 + n - k) * 1._C_DOUBLE)
    endif
  end function testfun0imag
  
  real(C_DOUBLE) function testfun1imag(k, n)
    implicit none
    integer, intent(in) :: k, n

    if ((k .eq. 0) .or. (2*k .eq. n)) then
       testfun1imag = 0._C_DOUBLE
    elseif (2*k .lt. n) then
       testfun1imag = tan(1._C_DOUBLE + (k * 1._C_DOUBLE)/(n * 2._C_DOUBLE))
    elseif (2*k .gt. n) then
       testfun1imag = -tan(1._C_DOUBLE + ((n - k) * 1._C_DOUBLE)/(n * 2._C_DOUBLE))
    endif
  end function testfun1imag
  
  real(C_DOUBLE) function testfun2imag(k, n)
    implicit none
    integer, intent(in) :: k, n

    if ((k .eq. 0) .or. (2*k .eq. n)) then
       testfun2imag = 0._C_DOUBLE
    elseif (2*k .lt. n) then
       testfun2imag = atan(1._C_DOUBLE + (k * 1._C_DOUBLE)/(n * 1._C_DOUBLE))
    elseif (2*k .gt. n) then
       testfun2imag = -atan(1._C_DOUBLE + ((n - k) * 1._C_DOUBLE)/(n * 1._C_DOUBLE))
    endif
  end function testfun2imag

  real(C_DOUBLE) function testfunreal(k0, k1, k2, n0, n1, n2)
    implicit none
    integer, intent(in) :: k0, k1, k2, n0, n1, n2
    integer :: p0, p1, p2, psum, s
    real(C_DOUBLE) :: fun0(2), fun1(2), fun2(2), tot

    fun0(1) = testfun0real(k0, n0)
    fun0(2) = testfun0imag(k0, n0)
    fun1(1) = testfun1real(k1, n1)
    fun1(2) = testfun1imag(k1, n1)
    fun2(1) = testfun2real(k2, n2)
    fun2(2) = testfun2imag(k2, n2)
    tot = 0._C_DOUBLE
    do p0 = 0, 1
       do p1 = 0, 1
          do p2 = 0, 1
             psum = p0 + p1 + p2
             if (mod(psum, 2) .eq. 0) then
                s = 1 - 2 * (psum/2)
                tot = tot + s * fun0(p0+1) * fun1(p1+1) * fun2(p2+1)
             endif
          enddo
       enddo
    enddo
    testfunreal = fun2(1) ! FIXME: tot
  end function testfunreal
  
  real(C_DOUBLE) function testfunimag(k0, k1, k2, n0, n1, n2)
    implicit none
    integer, intent(in) :: k0, k1, k2, n0, n1, n2
    integer :: p0, p1, p2, psum, s
    real(C_DOUBLE) :: fun0(2), fun1(2), fun2(2), tot

    fun0(1) = testfun0real(k0, n0)
    fun0(2) = testfun0imag(k0, n0)
    fun1(1) = testfun1real(k1, n1)
    fun1(2) = testfun1imag(k1, n1)
    fun2(1) = testfun2real(k2, n2)
    fun2(2) = testfun2imag(k2, n2)
    tot = 0._C_DOUBLE
    do p0 = 0, 1
       do p1 = 0, 1
          do p2 = 0, 1
             psum = p0 + p1 + p2
             if (mod(psum, 2) .eq. 1) then
                s = 1 - 2 * (psum/2)
                tot = tot + s * fun0(p0+1) * fun1(p1+1) * fun2(p2+1)
             endif
          enddo
       enddo
    enddo
    testfunimag = fun2(2) ! FIXME: tot
  end function testfunimag
  
  complex(C_DOUBLE_COMPLEX) function testfun(k0, k1, k2, n0, n1, n2)
    implicit none
    integer, intent(in) :: k0, k1, k2, n0, n1, n2

    testfun = &
         DCMPLX(testfun0real(k0, n0), testfun0imag(k0, n0)) * &
         DCMPLX(testfun1real(k1, n1), testfun1imag(k1, n1)) * &
         DCMPLX(testfun2real(k2, n2), testfun2imag(k2, n2))
  end function testfun
    
end module data_functions_mod
