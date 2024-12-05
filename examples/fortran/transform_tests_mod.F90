module transform_tests_mod
  use problem_dimensions_mod
  use, intrinsic :: iso_c_binding
  implicit none
  
contains

  subroutine singleMDDFTTest()
    use data_functions_mod, only : inputRealSymmetric
    use fft_mod, only : fftx_3D_mddft
    implicit none
    integer :: ix, iy, iz ! loop indices
    ! For the single-node FFTX transforms, dimension order as usual.
    complex(C_DOUBLE_COMPLEX), dimension(nx_global, ny_global, nz_global) :: in_array, out_array
    type(fftx_3D_mddft) :: tfm

    ! Set in_array.
    do iz = 1, nz_global
       do iy = 1, ny_global
          do ix = 1, nx_global
             in_array(ix, iy, iz) = inputRealSymmetric(ix, iy, iz)
          enddo
       enddo
    enddo
    ! in_array = in_array - sum(in_array) * point_weight
    
    ! print 200, sum(in_array), maxval(abs(in_array)), sum(abs(in_array))
200 format ('Single MDDFT input array sum =', 2es12.4, ' max =', es12.4, ' 1norm =', es12.4)

    call tfm%init()

    call tfm%execute(out_array, in_array)

    call tfm%finalize()
  end subroutine singleMDDFTTest

  subroutine singleIMDDFTTest()
    use data_functions_mod, only : inputRealSymmetric
    use fft_mod, only : fftx_3D_imddft
    implicit none
    integer :: ix, iy, iz ! loop indices
    ! For the single-node FFTX transforms, dimension order as usual.
    complex(C_DOUBLE_COMPLEX), dimension(nx_global, ny_global, nz_global) :: in_array, out_array
    type(fftx_3D_imddft) :: tfm

    ! Set in_array.
    do iz = 1, nz_global
       do iy = 1, ny_global
          do ix = 1, nx_global
             in_array(ix, iy, iz) = inputRealSymmetric(ix, iy, iz)
          enddo
       enddo
    enddo
    ! in_array = in_array - sum(in_array) * point_weight
    
    ! print 200, sum(in_array), maxval(abs(in_array)), sum(abs(in_array))
200 format ('Single IMDDFT input array sum =', 2es12.4, ' max =', es12.4, ' 1norm =', es12.4)

    call tfm%init()

    call tfm%execute(out_array, in_array)

    call tfm%finalize()
  end subroutine singleIMDDFTTest

  subroutine singleMDPRDFTTest()
    use data_functions_mod, only : inputRealSymmetric
    use fft_mod, only : fftx_3D_mdprdft
    implicit none
    integer :: ix, iy, iz ! loop indices
    ! For the single-node FFTX transforms, dimension order as usual.
    real(C_DOUBLE), dimension(nx_global, ny_global, nz_global) :: in_array
    complex(C_DOUBLE_COMPLEX), dimension(dimsc_global(1), dimsc_global(2), dimsc_global(3)) :: out_array
    type(fftx_3D_mdprdft) :: tfm

    ! Set in_array.
    do iz = 1, nz_global
       do iy = 1, ny_global
          do ix = 1, nx_global
             in_array(ix, iy, iz) = inputRealSymmetric(ix, iy, iz)
          enddo
       enddo
    enddo
    ! in_array = in_array - sum(in_array) * point_weight
    
    ! print 200, sum(in_array), maxval(abs(in_array)), sum(abs(in_array))
200 format ('Single MDPRDFT input array sum =', es12.4, ' max =', es12.4, ' 1norm =', es12.4)

    call tfm%init()

    call tfm%execute(out_array, in_array)

    call tfm%finalize()
  end subroutine singleMDPRDFTTest

  subroutine singleIMDPRDFTTest()
    use data_functions_mod, only : inputRealSymmetric
    use fft_mod, only : fftx_3D_imdprdft
    implicit none
    integer :: ix, iy, iz ! loop indices
    ! For the single-node FFTX transforms, dimension order as usual.
    complex(C_DOUBLE_COMPLEX), dimension(dimsc_global(1), dimsc_global(2), dimsc_global(3)) :: in_array
    real(C_DOUBLE), dimension(nx_global, ny_global, nz_global) :: out_array
    type(fftx_3D_imdprdft) :: tfm

    ! Set in_array.
    do iz = 1, dimsc_global(3)
       do iy = 1, dimsc_global(2)
          do ix = 1, dimsc_global(1)
             ! TODO: Figure out what input should be. Needs Hermitian symmetry.
             ! Could set it to something and then force symmetry.
             in_array(ix, iy, iz) = inputRealSymmetric(ix, iy, iz)
          enddo
       enddo
    enddo
    ! in_array = in_array - sum(in_array) * point_weight

    ! Sum for IMDPRDFT doesn't make sense, because input domain is truncated.
    ! print 200, sum(in_array), maxval(abs(in_array)), sum(abs(in_array))
200 format ('Single IMDPRDFT input array sum =', 2es12.4, ' max =', es12.4, ' 1norm =', es12.4)

    call tfm%init()

    call tfm%execute(out_array, in_array)

    call tfm%finalize()
  end subroutine singleIMDPRDFTTest

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
  
  subroutine distMDDFTTest
    use data_functions_mod, only : inputRealSymmetric
    use fft_mod, only : fftx_3D_mddft_dist
    use mpi_utils_mod, only : mpi_rank, i_am_mpi_master, MPISumComplex, MPIMax
    implicit none
    type(fftx_3D_mddft_dist) :: tfm
    ! FFTX layouts:  input [(Z0), Z1, Y, X], output [(X0), Y, X1, Z],
    ! and remember that in Fortran, first dimension changes fastest.
    complex(C_DOUBLE_COMPLEX), dimension(nx_rank, ny_rank, nz_rank) :: in_array
    complex(C_DOUBLE_COMPLEX), dimension(nz_out_rank, nx_out_rank, ny_out_rank) :: out_array
    complex(C_DOUBLE_COMPLEX) :: sum_all
    real(C_DOUBLE) :: max_all
    integer :: ix, iy, iz ! loop indices
    integer :: ix_global, iy_global, iz_global

    ! print 380, nx_global, ny_global, nz_global
380 format ('centered sine x on ', i2, ', squared y on ', i2, ' cos on ', i2)
    ! Set in_array.                                                             
    do iz = 1, nz_rank
       iz_global = iz + z_offset_rank
       do iy = 1, ny_rank
          iy_global = iy + y_offset_rank
          do ix = 1, nx_rank
             ix_global  = ix + x_offset_rank
             in_array(ix, iy, iz) = inputRealSymmetric(ix_global, iy_global, iz_global)
          enddo
       enddo
    enddo
    ! Get the sum to zero.
    ! Already sums to zero when it's inputRealSymmetric.
    ! in_array = in_array - sum(in_array) * point_weight
    sum_all = MPISumComplex(sum(in_array))
    max_all = MPIMax(maxval(abs(in_array)))
    if (i_am_mpi_master) then
       ! print 390, sum_all, max_all
390    format ('Dist MDDFT input array sum =', 2es12.4, ' max abs ', es12.4)
    endif

    call tfm%init()

    call tfm%execute(out_array, in_array)

    call tfm%finalize()
  end subroutine distMDDFTTest

  subroutine distIMDDFTTest
    use data_functions_mod, only : inputRealSymmetric
    use fft_mod, only : fftx_3D_imddft_dist
    use mpi_utils_mod, only : mpi_rank, i_am_mpi_master, MPISumComplex, MPIMax
    implicit none
    type(fftx_3D_imddft_dist) :: tfm
    ! FFTX layouts:  input [(X0), Y, X1, Z], output [(Z0), Z1, Y, X],
    ! and remember that in Fortran, first dimension changes fastest.
    complex(C_DOUBLE_COMPLEX), dimension(nz_out_rank, nx_out_rank, ny_out_rank) :: in_array
    complex(C_DOUBLE_COMPLEX), dimension(nx_rank, ny_rank, nz_rank) :: out_array
    complex(C_DOUBLE_COMPLEX) :: sum_all
    real(C_DOUBLE) :: max_all
    integer :: ix, iy, iz ! loop indices
    integer :: ix_global, iy_global, iz_global

    ! print 380, nx_global, ny_global, nz_global
380 format ('centered sine x on ', i2, ', squared y on ', i2, ' cos on ', i2)
    ! Set in_array.                                                             
    do iy = 1, ny_out_rank
       iy_global = iy + y_off_out_rank
       do ix = 1, nx_out_rank
          ix_global  = ix + x_off_out_rank
          do iz = 1, nz_out_rank
             iz_global = iz + z_off_out_rank
             in_array(iz, ix, iy) = inputRealSymmetric(ix_global, iy_global, iz_global)
          enddo
       enddo
    enddo
    ! Get the sum to zero.
    ! Already sums to zero when it's inputRealSymmetric.
    ! in_array = in_array - sum(in_array) * point_weight
    sum_all = MPISumComplex(sum(in_array))
    max_all = MPIMax(maxval(abs(in_array)))
    if (i_am_mpi_master) then
       ! print 390, sum_all, max_all
390    format ('Dist IMDDFT input array sum =', 2es12.4, ' max abs ', es12.4)
    endif

    call tfm%init()

    call tfm%execute(out_array, in_array)

    call tfm%finalize()

  end subroutine distIMDDFTTest

  subroutine distMDPRDFTTest
    use data_functions_mod, only : inputRealSymmetric
    use fft_mod, only : fftx_3D_mdprdft_dist
    use mpi_utils_mod, only : mpi_rank, i_am_mpi_master, MPISumReal, MPIMax
    implicit none
    type(fftx_3D_mdprdft_dist) :: tfm
    ! FFTX layouts:  input [(Z0), Z1, Y, X], output [(X0), Y, X1, Z],
    ! and remember that in Fortran, first dimension changes fastest.
    real(C_DOUBLE), dimension(nx_rank, ny_rank, nz_rank) :: in_array
    complex(C_DOUBLE_COMPLEX), dimension(nz_trunc_out_rank, nx_trunc_out_rank, ny_trunc_out_rank) :: out_array
    real(C_DOUBLE) :: sum_all, max_all
    integer :: ix, iy, iz ! loop indices
    integer :: ix_global, iy_global, iz_global

    ! print 380, nx_global, ny_global, nz_global
380 format ('centered sine x on ', i2, ', squared y on ', i2, ' cos on ', i2)
    ! Set in_array.                                                             
    do iz = 1, nz_rank
       iz_global = iz + z_offset_rank
       do iy = 1, ny_rank
          iy_global = iy + y_offset_rank
          do ix = 1, nx_rank
             ix_global  = ix + x_offset_rank
             in_array(ix, iy, iz) = inputRealSymmetric(ix_global, iy_global, iz_global)
          enddo
       enddo
    enddo
    ! Get the sum to zero.
    ! Already sums to zero when it's inputRealSymmetric.
    ! in_array = in_array - sum(in_array) * point_weight
    sum_all = MPISumReal(sum(in_array))
    max_all = MPIMax(maxval(abs(in_array)))
    if (i_am_mpi_master) then
       ! print 390, sum_all, max_all
390    format ('Dist MDPRDFT input array sum =', es12.4, ' max abs ', es12.4)
    endif

    call tfm%init()

    call tfm%execute(out_array, in_array)

    call tfm%finalize()
  end subroutine distMDPRDFTTest

  subroutine distIMDPRDFTTest
    use data_functions_mod, only : inputRealSymmetric
    use fft_mod, only : fftx_3D_imdprdft_dist
    use mpi_utils_mod, only : mpi_rank, i_am_mpi_master, MPISumComplex, MPIMax
    implicit none
    type(fftx_3D_imdprdft_dist) :: tfm
    ! FFTX layouts:  input [(Z0), Z1, Y, X], output [(X0), Y, X1, Z],
    ! and remember that in Fortran, first dimension changes fastest.
    complex(C_DOUBLE_COMPLEX), dimension(nz_trunc_out_rank, nx_trunc_out_rank, ny_trunc_out_rank) :: in_array
    real(C_DOUBLE), dimension(nx_rank, ny_rank, nz_rank) :: out_array
    complex(C_DOUBLE_COMPLEX) :: sum_all
    real(C_DOUBLE) :: max_all
    integer :: ix, iy, iz ! loop indices
    integer :: ix_global, iy_global, iz_global

    ! print 380, nx_global, ny_global, nz_global
380 format ('centered sine x on ', i2, ', squared y on ', i2, ' cos on ', i2)
    ! Set in_array.
    do iy = 1, ny_trunc_out_rank
       iy_global = iy + y_off_trunc_out_rank
       do ix = 1, nx_trunc_out_rank
          ix_global  = ix + x_off_trunc_out_rank
          do iz = 1, nz_trunc_out_rank
             iz_global = iz + z_off_trunc_out_rank
             in_array(iz, ix, iy) = inputRealSymmetric(ix_global, iy_global, iz_global)
          enddo
       enddo
    enddo
    ! Get the sum to zero.
    ! Already sums to zero when it's inputRealSymmetric.
    ! in_array = in_array - sum(in_array) * point_weight
    sum_all = MPISumComplex(sum(in_array))
    max_all = MPIMax(maxval(abs(in_array)))
    if (i_am_mpi_master) then
       ! Sum for IMDPRDFT doesn't make sense, because input domain is truncated.
       ! print 390, sum_all, max_all
390    format ('Dist IMDPRDFT input array sum =', 2es12.4, ' max abs ', es12.4)
    endif

    call tfm%init()

    call tfm%execute(out_array, in_array)

    call tfm%finalize()
  end subroutine distIMDPRDFTTest

#endif

end module transform_tests_mod
