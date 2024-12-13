module convolution_mod
  use data_functions_mod
  implicit none

contains

  integer function wrap1(i, n)
    implicit none
    integer, intent(in) :: i, n

    wrap1 = mod(i + n-1, n) + 1
    if ((wrap1 .lt. 1) .or. (wrap1 .gt. n)) then
       print *, 'Error in wrap1: output ', wrap1
       stop
    endif
  end function wrap1

  real(C_DOUBLE) function symRealPoisson(ix, iy, iz)
    use math_constants_mod, only : PI
    implicit none
    integer, intent(in) :: ix, iy, iz
    real(C_DOUBLE) :: sin2sum

    if ((ix .eq. 1) .and. (iy .eq. 1) .and. (iz .eq. 1)) then
       symRealPoisson = 0._C_DOUBLE
    elseif ((ix .gt. nx_global) .or. (iy .gt. ny_global) .or. (iz .gt. nz_global)) then
       ! exceeds bounds
       symRealPoisson = 0._C_DOUBLE
    else
       sin2sum = &
            sin((ix - 1)*PI / (nx_global * 1._C_DOUBLE))**2 + &
            sin((iy - 1)*PI / (ny_global * 1._C_DOUBLE))**2 + &
            sin((iz - 1)*PI / (nz_global * 1._C_DOUBLE))**2
!       symRealPoisson = -1._C_DOUBLE / ((4._C_DOUBLE * npts) * sin2sum)
       symRealPoisson = -point_weight / (4._C_DOUBLE * sin2sum)
    endif
  end function symRealPoisson

  subroutine laplacian2periodicReal(out_array, in_array)
    implicit none
    real(C_DOUBLE), dimension(:, :, :), intent(out) :: out_array
    real(C_DOUBLE), dimension(:, :, :), intent(in) :: in_array
    integer :: ix, iy, iz ! loop indices

    do iz = 1, nz_global
       do iy = 1, ny_global
          do ix = 1, nx_global
             out_array(ix, iy, iz) = &
                  -6._C_DOUBLE * in_array(ix, iy, iz) + &
                  in_array(wrap1(ix+1, nx_global), iy, iz) + &
                  in_array(wrap1(ix-1, nx_global), iy, iz) + &
                  in_array(ix, wrap1(iy+1, ny_global), iz) + &
                  in_array(ix, wrap1(iy-1, ny_global), iz) + &
                  in_array(ix, iy, wrap1(iz+1, nz_global)) + &
                  in_array(ix, iy, wrap1(iz-1, nz_global))
          enddo
       enddo
    enddo
  end subroutine laplacian2periodicReal
  
  subroutine laplacian2periodicComplex(out_array, in_array)
    ! This works ONLY if mpi_size is 1.
    implicit none
    complex(C_DOUBLE_COMPLEX), dimension(:, :, :), intent(out) :: out_array
    complex(C_DOUBLE_COMPLEX), dimension(:, :, :), intent(in) :: in_array
    integer :: ix, iy, iz ! loop indices

    do iz = 1, nz_global
       do iy = 1, ny_global
          do ix = 1, nx_global
             out_array(ix, iy, iz) = &
                  -6._C_DOUBLE * in_array(ix, iy, iz) + &
                  in_array(wrap1(ix+1, nx_global), iy, iz) + &
                  in_array(wrap1(ix-1, nx_global), iy, iz) + &
                  in_array(ix, wrap1(iy+1, ny_global), iz) + &
                  in_array(ix, wrap1(iy-1, ny_global), iz) + &
                  in_array(ix, iy, wrap1(iz+1, nz_global)) + &
                  in_array(ix, iy, wrap1(iz-1, nz_global))
          enddo
       enddo
    enddo
  end subroutine laplacian2periodicComplex
  
  subroutine doRealConvolution(out_array, in_array, sym_array)
    use fft_mod, only : fftx_3D_mdprdft, fftx_3D_imdprdft
    implicit none
    real(C_DOUBLE), dimension(:, :, :), intent(out) :: out_array
    real(C_DOUBLE), dimension(:, :, :), intent(in) :: in_array
    real(C_DOUBLE), dimension(:, :, :), intent(in) :: sym_array
    complex(C_DOUBLE_COMPLEX) :: inter_array(dimsc_global(1), dimsc_global(2), dimsc_global(3))
    real(C_DOUBLE) :: diff_array(dims_global(1), dims_global(2), dims_global(3))
    type(fftx_3D_mdprdft) :: tfm_mdprdft
    type(fftx_3D_imdprdft) :: tfm_imdprdft
    integer :: ix, iy, iz
    complex(C_DOUBLE_COMPLEX) :: v
    real(C_DOUBLE) :: w
    real(C_DOUBLE) :: tol = 1.e-10

    call tfm_mdprdft%init()
    call tfm_imdprdft%init()

    !    print *, 'calling tfm_mdprdft%execute'
    ! Set inter_array = MDPRDFT(in_array).
    call tfm_mdprdft%execute(inter_array, in_array)
 
    ! Set inter_array *= sym_array.
    inter_array = inter_array * sym_array

    ! Set out_array = IMDPRDFT(inter_array).
    call tfm_imdprdft%execute(out_array, inter_array)
 
    call tfm_mdprdft%finalize()
    call tfm_imdprdft%finalize()
  end subroutine doRealConvolution

  subroutine doComplexConvolution(out_array, in_array, sym_array)
    use fft_mod, only : fftx_3D_mddft, fftx_3D_imddft
    implicit none
    complex(C_DOUBLE_COMPLEX), dimension(:, :, :), intent(out) :: out_array
    complex(C_DOUBLE_COMPLEX), dimension(:, :, :), intent(in) :: in_array
    complex(C_DOUBLE_COMPLEX), dimension(:, :, :), intent(in) :: sym_array
    complex(C_DOUBLE_COMPLEX) :: inter_array(nx_global, ny_global, nz_global)
    type(fftx_3D_mddft) :: tfm_mddft
    type(fftx_3D_imddft) :: tfm_imddft
    integer :: ix, iy, iz ! loop indices
    complex(C_DOUBLE_COMPLEX) :: v

    call tfm_mddft%init()
    call tfm_imddft%init()
    
    ! Set inter_array = MDDFT(in_array).
    call tfm_mddft%execute(inter_array, in_array)

    ! Set inter_array *= sym_array.
    inter_array = inter_array * sym_array

    ! Set out_array = IMDDFT(inter_array).
    call tfm_imddft%execute(out_array, inter_array)

    call tfm_mddft%finalize()
    call tfm_imddft%finalize()
  end subroutine doComplexConvolution
  
  subroutine singleComplexConvolutionTest() ! data_out, data_in, data_symbol
    implicit none
    integer :: ix, iy, iz ! loop indices
    complex(C_DOUBLE_COMPLEX), dimension(nx_global, ny_global, nz_global) :: &
         in_array, out_array, sym_array, lap2out_array, diff_array
    real(C_DOUBLE) :: in_abs_max, lap2out_abs_max, diff_abs_max, rval, ival

    ! Set in_array.
    do iz = 1, nz_global
       do iy = 1, ny_global
          do ix = 1, nx_global
             call random_number(rval)
             call random_number(ival)
             in_array(ix, iy, iz) = dcmplx(rval, ival)
             ! in_array(ix, iy, iz) = dcmplx(inputRealSymmetric(ix, iy, iz), 0._C_DOUBLE)
          enddo
       enddo
    enddo
    in_array = in_array - sum(in_array) * point_weight
    
    print 200, sum(in_array), maxval(abs(in_array)), sum(abs(in_array))
200 format ('Input array sum =', es12.4, es12.4, ' max =', es12.4, ' 1norm =', es12.4)

    ! Set sym_array.
    do iz = 1, nz_global
       do iy = 1, ny_global
          do ix = 1, nx_global
             !             sym_array(ix, iy, iz) = symRealFun(ix, iy, iz)
             sym_array(ix, iy, iz) = symRealPoisson(ix, iy, iz)
          enddo
       enddo
    enddo

    call doComplexConvolution(out_array, in_array, sym_array)

    call laplacian2periodicComplex(lap2out_array, out_array)

    ! Compare lap2out_array with in_array.
    in_abs_max = maxval(abs(in_array))
    lap2out_abs_max = maxval(abs(lap2out_array))
    diff_array = lap2out_array - in_array
    diff_abs_max = maxval(abs(diff_array))

    print 400, in_abs_max, lap2out_abs_max, diff_abs_max, (diff_abs_max/in_abs_max)
400 format ('Single complex conv max abs input', es12.4, ' laplacian2(output)', es12.4, ' ERROR abs', es12.4, ' relative', es12.4)
  end subroutine singleComplexConvolutionTest

  subroutine singleRealConvolutionTest()
    implicit none
    integer :: ix, iy, iz ! loop indices
    real(C_DOUBLE), dimension(nx_global, ny_global, nz_global) :: &
         in_array, out_array, lap2out_array, diff_array
    real(C_DOUBLE), dimension(dimsc_global(1), dimsc_global(2), dimsc_global(3)) :: sym_array
    real(C_DOUBLE) :: in_abs_max, lap2out_abs_max, diff_abs_max

    ! Set in_array.
    do iz = 1, nz_global
       do iy = 1, ny_global
          do ix = 1, nx_global
             call random_number(in_array(ix, iy, iz))
             ! in_array(ix, iy, iz) = inputRealSymmetric(ix, iy, iz)
          enddo
       enddo
    enddo
    in_array = in_array - sum(in_array) * point_weight
    print 200, sum(in_array), maxval(abs(in_array)), sum(abs(in_array))
200 format ('Input array sum =', es12.4, ' max =', es12.4, ' 1norm =', es12.4)

    ! Set sym_array.
    do iz = 1, dimsc_global(3)
       do iy = 1, dimsc_global(2)
          do ix = 1, dimsc_global(1)
             sym_array(ix, iy, iz) = symRealPoisson(ix, iy, iz)
          enddo
       enddo
    enddo

    call doRealConvolution(out_array, in_array, sym_array)

    call laplacian2periodicReal(lap2out_array, out_array)

    ! Compare lap2out_array with in_array.
    in_abs_max = maxval(abs(in_array))
    lap2out_abs_max = maxval(abs(lap2out_array))
    diff_array = lap2out_array - in_array
    diff_abs_max = maxval(abs(diff_array))

    print 400, in_abs_max, lap2out_abs_max, diff_abs_max, (diff_abs_max/in_abs_max)
400 format ('Single real conv max abs input', es12.4, ' laplacian2(output)', es12.4, ' ERROR abs', es12.4, ' relative', es12.4)
  end subroutine singleRealConvolutionTest

#if defined(FFTX_CUDA) || defined(FFTX_HIP)

  subroutine laplacian2periodicDistReal(out_array, in_array)
    use mpi_utils_mod, only : mpi_rank, MPIExchange3dReal
    implicit none
    real(C_DOUBLE), dimension(:, :, :), intent(out) :: out_array
    real(C_DOUBLE), dimension(:, :, :), intent(in) :: in_array
    integer :: ix, iy, iz ! loop indices
    integer :: minx, maxx, miny, maxy, minz, maxz
    real(C_DOUBLE), dimension(:, :, :), allocatable :: expanded

    minx = lbound(in_array, 1)
    miny = lbound(in_array, 2)
    minz = lbound(in_array, 3)

    maxx = ubound(in_array, 1)
    maxy = ubound(in_array, 2)
    maxz = ubound(in_array, 3)
    allocate(expanded(minx-1:maxx+1, miny-1:maxy+1, minz-1:maxz+1))
    call MPIExchange3dReal(expanded, in_array)
    do iz = 1, nz_rank
       do iy = 1, ny_rank
          do ix = 1, nx_rank
             out_array(ix, iy, iz) = &
                  -6._C_DOUBLE * in_array(ix, iy, iz) + &
                  expanded(ix-1, iy, iz) + expanded(ix+1, iy, iz) + &
                  expanded(ix, iy-1, iz) + expanded(ix, iy+1, iz) + &
                  expanded(ix, iy, iz-1) + expanded(ix, iy, iz+1)
          enddo
       enddo
    enddo
  end subroutine laplacian2periodicDistReal
  
  subroutine laplacian2periodicDistComplex(out_array, in_array)
    use mpi_utils_mod, only : mpi_rank, MPIExchange3dComplex
    implicit none
    complex(C_DOUBLE_COMPLEX), dimension(:, :, :), intent(out) :: out_array
    complex(C_DOUBLE_COMPLEX), dimension(:, :, :), intent(in) :: in_array
    integer :: ix, iy, iz ! loop indices
    integer :: minx, maxx, miny, maxy, minz, maxz
    complex(C_DOUBLE_COMPLEX), dimension(:, :, :), allocatable :: expanded

    ! print *, 'Rank', mpi_rank, 'calling exchange'
    minx = lbound(in_array, 1)
    miny = lbound(in_array, 2)
    minz = lbound(in_array, 3)

    maxx = ubound(in_array, 1)
    maxy = ubound(in_array, 2)
    maxz = ubound(in_array, 3)
    allocate(expanded(minx-1:maxx+1, miny-1:maxy+1, minz-1:maxz+1))
    call MPIExchange3dComplex(expanded, in_array)
    do iz = 1, nz_rank
       do iy = 1, ny_rank
          do ix = 1, nx_rank
             out_array(ix, iy, iz) = &
                  -6._C_DOUBLE * in_array(ix, iy, iz) + &
                  expanded(ix-1, iy, iz) + expanded(ix+1, iy, iz) + &
                  expanded(ix, iy-1, iz) + expanded(ix, iy+1, iz) + &
                  expanded(ix, iy, iz-1) + expanded(ix, iy, iz+1)
          enddo
       enddo
    enddo
  end subroutine laplacian2periodicDistComplex
  
  subroutine doDistComplexConvolution(out_array, in_array, sym_array)
    use fft_mod, only : fftx_3D_mddft_dist, fftx_3D_imddft_dist
    implicit none
    complex(C_DOUBLE_COMPLEX), dimension(:, :, :), intent(out) :: out_array
    complex(C_DOUBLE_COMPLEX), dimension(:, :, :), intent(in) :: in_array
    complex(C_DOUBLE_COMPLEX), dimension(:, :, :), intent(in) :: sym_array
    complex(C_DOUBLE_COMPLEX) :: inter_array(nz_out_rank, nx_out_rank, ny_out_rank), v
    type(fftx_3D_mddft_dist) :: tfm_mddft
    type(fftx_3D_imddft_dist) :: tfm_imddft
    integer :: ix, iy, iz, ind
    integer :: ix_global, iy_global, iz_global

    call tfm_mddft%init()
    call tfm_imddft%init()

    ! fftx_3D_mddft_dist is initialized by
    ! int(nx_fft, kind=4), int(ny_fft, kind=4), int(nz_fft, kind=4)
    ! which gets sent to
    ! fftx_plan_distributed_1d(comm, p, M, N, K, batch, is_embedded, is_complex);
    ! so M = nx_fft, N = ny_fft, K = nz_fft.
    ! X loops over 1:nx, Y loops over 1:ny, Z loops over 1:nz.
    ! Layout of input data is [(Z0), Z1, Y, X] where
    ! Z0 is distributed over the ranks (1:p),
    ! Z1 is slowest local dimension (looping over 1:nz/p),
    ! Y is middle local dimension (looping over 1:ny),
    ! X is fastest local dimension (looping over 1:nx).

    ! In Fortran, the first array dimension is fastest.
    ! So Fortran local input array should have dimensions (nx, ny, nz/p),
    ! that is, (nx_rank, ny_rank, nz_rank).

    ! Set inter_array = MDDFT(in_array).
    ! Layout of output data is [(X0), Y, X1, Z] where
    ! X0 is distributed over the ranks (1:p), and
    ! X1 is local (looping over 1:nx/p).

    ! So Fortran local output array should have dimensions (nz, nx/p, ny),
    ! because Z over 1:nz goes fastest, then X1 over 1:nx/p, then Y over 1:ny.
    ! Hence array dimensions (nz_out_rank, nx_out_rank, ny_out_rank).

    ! complex(C_DOUBLE_COMPLEX) :: inter_array(nz_out_rank, nx_out_rank, ny_out_rank)
    call tfm_mddft%execute(inter_array, in_array)

    ! Set inter_array *= sym_array.
    inter_array = inter_array * sym_array

    ! Set out_array = IMDDFT(inter_array).
    call tfm_imddft%execute(out_array, inter_array)

    call tfm_mddft%finalize()
    call tfm_imddft%finalize()
  end subroutine doDistComplexConvolution

  subroutine distComplexConvolutionTest() ! data_out, data_in, data_symbol
    use mpi_utils_mod, only : i_am_mpi_master, mpi_rank, MPISumComplex, MPIMaxReal
    implicit none
    integer :: ix, iy, iz ! loop indices
    complex(C_DOUBLE_COMPLEX), dimension(nx_rank, ny_rank, nz_rank) :: &
         in_array, out_array, lap2out_array, diff_array
    complex(C_DOUBLE_COMPLEX), dimension(nz_out_rank, nx_out_rank, ny_out_rank) :: &
         sym_array
    complex(C_DOUBLE_COMPLEX) :: sum_all, ptval
    real(C_DOUBLE) :: in_abs_max, lap2out_abs_max, diff_abs_max, max_all, rval, ival
    integer :: ix_global, iy_global, iz_global ! global coordinate values

    ! Might want to write this out if calling inputRealSymmetric.
    !    print 380, nx_global, ny_global, nz_global
    ! 380 format ('centered sine x on ', i2, ', squared y on ', i2, ' cos on ', i2)

    !    print 385, mpi_rank, &
    !         nx_rank, ny_rank, nz_rank, &
    !         x_offset_rank, y_offset_rank, z_offset_rank
385 format ('Dist complex Rank', i2, ' in_array on', 3i3, ' offset by', 3i3)
    ! Set in_array.
    do iz = 1, nz_rank
       iz_global = iz + z_offset_rank
       do iy = 1, ny_rank
          iy_global = iy + y_offset_rank
          do ix = 1, nx_rank
             ix_global  = ix + x_offset_rank
             call random_number(rval)
             call random_number(ival)
             in_array(ix, iy, iz) = dcmplx(rval, ival)
             ! in_array(ix, iy, iz) = dcmplx(inputRealSymmetric(ix_global, iy_global, iz_global), 0._C_DOUBLE)
          enddo
       enddo
    enddo
    ! Get the sum to zero if you're going to do a Poisson solve.
    ! If input is set to inputRealSymmetric, then already sums to zero.
    sum_all = MPISumComplex(sum(in_array))
    in_array = in_array - sum_all * point_weight
    sum_all = MPISumComplex(sum(in_array))
    max_all = MPIMaxReal(maxval(abs(in_array)))
    if (i_am_mpi_master) then
       print 390, sum_all, max_all
390    format ('Dist complex conv input sum ', 2es12.4, ' max abs ', es12.4)
    endif

    !    print 395, mpi_rank, &
    !         nz_out_rank, nx_out_rank, ny_out_rank, &
    !         z_off_out_rank, x_off_out_rank, y_off_out_rank
395 format ('Dist complex Rank', i2, ' sym_array on', 3i3, ' offset by', 3i3)
    ! Set sym_array.
    do iy = 1, ny_out_rank
       iy_global = iy + y_off_out_rank
       do ix = 1, nx_out_rank
          ix_global  = ix + x_off_out_rank
          do iz = 1, nz_out_rank
             iz_global = iz + z_off_out_rank
             ! Need the input arguments to symRealPoisson to be in the ranges
             ! 1:nx_global, 1:ny_global, 1:nz_global.
             sym_array(iz, ix, iy) = dcmplx(symRealPoisson(ix_global, iy_global, iz_global), 0._C_DOUBLE)
          enddo
       enddo
    enddo

    call doDistComplexConvolution(out_array, in_array, sym_array)

    ! Compare output, or laplacian of output, with input.
    call laplacian2periodicDistComplex(lap2out_array, out_array)
    lap2out_abs_max = MPIMaxReal(maxval(abs(lap2out_array)))
    diff_array = lap2out_array - in_array

    in_abs_max = MPIMaxReal(maxval(abs(in_array)))
    diff_abs_max = MPIMaxReal(maxval(abs(diff_array)))

    do iz = 1, nz_rank
       iz_global = iz + z_offset_rank
       do iy = 1, ny_rank
          iy_global = iy + y_offset_rank
          do ix = 1, nx_rank
             ix_global  = ix + x_offset_rank
             if (abs(diff_array(ix, iy, iz)) .gt. 1.e-10) then
                ! print 500, mpi_rank, ix_global, iy_global, iz_global, &
                ! in_array(ix, iy, iz), out_array(ix, iy, iz), abs(diff_array(ix, iy, iz))
                ptval = lap2out_array(ix, iy, iz)
                print 500, mpi_rank, ix_global, iy_global, iz_global, &
                     in_array(ix, iy, iz), ptval, abs(diff_array(ix, iy, iz))
             endif
          enddo
       enddo
    enddo
500 format (i2, ' (', i2, ',', i2, ',', i2, ') input', 2es12.4, ' output', 2es12.4, ' abs diff', es12.4)
    
!!    print 400, in_abs_max, lap2out_abs_max
!!400 format ('Max value of input', es12.4, ' laplacian2(output)', es12.4)

    ! max_all = MPIMaxReal(maxval(abs(out_array)))
    if (i_am_mpi_master) then
       print 400, in_abs_max, lap2out_abs_max, diff_abs_max, (diff_abs_max/in_abs_max)
400    format ('Dist complex conv max abs input', es12.4, ' laplacian2(output)', es12.4, ' ERROR abs', es12.4, ' relative', es12.4)
    endif

  end subroutine distComplexConvolutionTest

  subroutine doDistRealConvolution(out_array, in_array, sym_array)
    use fft_mod, only : fftx_3D_mdprdft_dist, fftx_3D_imdprdft_dist
    implicit none
    real(C_DOUBLE), dimension(:, :, :), intent(out) :: out_array
    real(C_DOUBLE), dimension(:, :, :), intent(in) :: in_array
    real(C_DOUBLE), dimension(:, :, :), intent(in) :: sym_array
    ! In C2C distributed, dimensions (nz_out_rank, nx_out_rank, ny_out_rank).
    complex(C_DOUBLE_COMPLEX) :: inter_array(nz_trunc_out_rank, nx_trunc_out_rank, ny_trunc_out_rank), v
    real(C_DOUBLE) :: w
    type(fftx_3D_mdprdft_dist) :: tfm_mdprdft
    type(fftx_3D_imdprdft_dist) :: tfm_imdprdft
    integer :: ix, iy, iz, ind
    integer :: ix_global, iy_global, iz_global

    call tfm_mdprdft%init()
    call tfm_imdprdft%init()

    ! fftx_3D_mdprdft_dist is initialized by
    ! int(nx_fft, kind=4), int(ny_fft, kind=4), int(nz_fft, kind=4)
    ! which gets sent to
    ! fftx_plan_distributed_1d(comm, p, M, N, K, batch, is_embedded, is_complex);
    ! so M = nx_fft, N = ny_fft, K = nz_fft.
    ! X loops over 1:nx, Y loops over 1:ny, Z loops over 1:nz.
    ! Layout of input data is [(Z0), Z1, Y, X] where
    ! Z0 is distributed over the ranks (1:p),
    ! Z1 is slowest local dimension (looping over 1:nz/p),
    ! Y is middle local dimension (looping over 1:ny),
    ! X is fastest local dimension (looping over 1:nx).

    ! In Fortran, the first array dimension is fastest.
    ! So Fortran local input array should have dimensions (nx, ny, nz/p),
    ! that is, (nx_rank, ny_rank, nz_rank).
    ! R2C input is [nx_rank, ny_rank, nz_rank] block-distributed in Z.

    ! Set inter_array = MDPRDFT(in_array).
    ! Layout of output data is [(X0), Y, X1, Z] where
    ! X0 is distributed over the ranks (1:p), and
    ! X1 is local (looping over 1:nx/p).

    ! So Fortran local output array should have dimensions (nz, nx/p, ny),
    ! because Z over 1:nz goes fastest, then X1 over 1:nx/p, then Y over 1:ny.
    ! Hence array dimensions (nz_trunc_out_rank, nx_trunc_out_rank, ny_trunc_out_rank).
    ! R2C output is [nz_rank, nx_rank/2 + 1, ny_rank] block-distributed in Y.

    ! complex(C_DOUBLE_COMPLEX) :: inter_array(nz_trunc_out_rank, nx_trunc_out_rank, ny_trunc_out_rank)
    call tfm_mdprdft%execute(inter_array, in_array)

    ! Set inter_array *= sym_array.
    inter_array = inter_array * sym_array

    ! Set out_array = IMDPRDFT(inter_array).
    call tfm_imdprdft%execute(out_array, inter_array)
    
    call tfm_mdprdft%finalize()
    call tfm_imdprdft%finalize()
  end subroutine doDistRealConvolution
  
  subroutine distRealConvolutionTest() ! data_out, data_in, data_symbol
    use mpi_utils_mod, only : i_am_mpi_master, mpi_rank, MPISumReal, MPIMaxReal
    implicit none
    integer :: ix, iy, iz ! loop indices
    real(C_DOUBLE), dimension(nx_rank, ny_rank, nz_rank) :: &
         in_array, out_array, lap2out_array, diff_array
    real(C_DOUBLE), dimension(nz_trunc_out_rank, nx_trunc_out_rank, ny_trunc_out_rank) :: &
         sym_array
    real(C_DOUBLE) :: sum_all, ptval, in_abs_max, lap2out_abs_max, diff_abs_max, max_all
    integer :: ix_global, iy_global, iz_global ! global coordinate values

    ! Might want to write this out if calling inputRealSymmetric.
    !    print 380, nx_global, ny_global, nz_global
    ! 380 format ('centered sine x on ', i2, ', squared y on ', i2, ' cos on ', i2)

    ! R2C input is [nx_rank, ny_rank, nz_rank] block-distributed in Z.
    !    print 385, mpi_rank, &
    !         nx_rank, ny_rank, nz_rank, &
    !         x_offset_rank, y_offset_rank, z_offset_rank
385 format ('Dist real Rank', i2, ' in_array on', 3i3, ' offset by', 3i3)
    ! Set in_array.
    do iz = 1, nz_rank
       iz_global = iz + z_offset_rank
       do iy = 1, ny_rank
          iy_global = iy + y_offset_rank
          do ix = 1, nx_rank
             ix_global  = ix + x_offset_rank
             ! This check should not be necessary.
             if ((ix_global .le. nx_global) .and. &
                  (iy_global .le. ny_global) .and. &
                  (iz_global .le. nz_global)) then
                ! in_array(ix, iy, iz) = inputRealSymmetric(ix_global, iy_global, iz_global)
                call random_number(in_array(ix, iy, iz))
             else
                print *, 'out of range'
                in_array(ix, iy, iz) = 0._C_DOUBLE
             endif
          enddo
       enddo
    enddo
    ! Get the sum to zero if you're going to do a Poisson solve.
    ! If input is set to inputRealSymmetric, then already sums to zero.
    sum_all = MPISumReal(sum(in_array))
    in_array = in_array - sum_all * point_weight
    sum_all = MPISumReal(sum(in_array))
    max_all = MPIMaxReal(maxval(abs(in_array)))
    if (i_am_mpi_master) then
       print 390, sum_all, max_all
390    format ('Dist real conv input sum ', es12.4, ' max abs ', es12.4)
    endif

    ! R2C output is [nz_rank, nx_rank/2 + 1, ny_rank] block-distributed in X.
    !    print 395, mpi_rank, &
    !         nz_trunc_out_rank, nx_trunc_out_rank, ny_trunc_out_rank, &
    !         z_off_trunc_out_rank, x_off_trunc_out_rank, y_off_trunc_out_rank
395 format ('Dist real Rank', i2, ' sym_array on', 3i3, ' offset by', 3i3)
    ! Set sym_array.
    do iy = 1, ny_trunc_out_rank
       iy_global = iy + y_off_trunc_out_rank
       do ix = 1, nx_trunc_out_rank
          ix_global  = ix + x_off_trunc_out_rank
          do iz = 1, nz_trunc_out_rank
             iz_global = iz + z_off_trunc_out_rank
             ! Need the input arguments to symRealPoisson to be in the ranges
             ! 1:nx_global, 1:ny_global, 1:nz_global.
             sym_array(iz, ix, iy) = symRealPoisson(ix_global, iy_global, iz_global)
          enddo
       enddo
    enddo

    call doDistRealConvolution(out_array, in_array, sym_array)

    ! Compare output, or laplacian of output, with input.
    call laplacian2periodicDistReal(lap2out_array, out_array)
    lap2out_abs_max = MPIMaxReal(maxval(abs(lap2out_array)))
    diff_array = lap2out_array - in_array

    in_abs_max = MPIMaxReal(maxval(abs(in_array)))
    diff_abs_max = MPIMaxReal(maxval(abs(diff_array)))

    do iz = 1, nz_rank
       iz_global = iz + z_offset_rank
       do iy = 1, ny_rank
          iy_global = iy + y_offset_rank
          do ix = 1, nx_rank
             ix_global  = ix + x_offset_rank
             if (abs(diff_array(ix, iy, iz)) .gt. 1.e-10) then
                ! print 500, mpi_rank, ix_global, iy_global, iz_global, &
                ! in_array(ix, iy, iz), out_array(ix, iy, iz), abs(diff_array(ix, iy, iz))
                ptval = lap2out_array(ix, iy, iz)
                print 500, mpi_rank, ix_global, iy_global, iz_global, &
                     in_array(ix, iy, iz), ptval, abs(diff_array(ix, iy, iz))
             endif
          enddo
       enddo
    enddo
500 format (i2, ' (', i2, ',', i2, ',', i2, ') input', es12.4, ' output', es12.4, ' abs diff', es12.4)
    
!!    print 400, in_abs_max, lap2out_abs_max
!!400 format ('Max value of input', es12.4, ' laplacian2(output)', es12.4)

    ! max_all = MPIMaxReal(maxval(abs(out_array)))
    if (i_am_mpi_master) then
       print 400, in_abs_max, lap2out_abs_max, diff_abs_max, (diff_abs_max/in_abs_max)
400    format ('Dist real conv max abs input', es12.4, ' laplacian2(output)', es12.4, ' ERROR abs', es12.4, ' relative', es12.4)
    endif

  end subroutine distRealConvolutionTest

#endif
  
end module convolution_mod
