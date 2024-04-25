module problem_dimensions_mod
  use, intrinsic :: iso_c_binding
  implicit none

  integer, parameter :: trunc_fortran_dim=1
  ! problem dimension global and per rank
  integer ::  &
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
       dimsc_dist_global(3), &
       dims_rank(3), dims_out_rank(3), dimsc_out_rank(3), &
       off_set_rank(3), off_out_rank(3), off_trunc_out_rank(3), & ! rank offsets for all dimensions
       x_offset_rank, y_offset_rank, z_offset_rank, &
       x_off_out_rank, y_off_out_rank, z_off_out_rank, &
       x_off_trunc_out_rank, y_off_trunc_out_rank, z_off_trunc_out_rank, &
       xyz_start_rank(3), xyz_end_rank(3), & ! start and end per rank in all dimension
       x_start_rank, y_start_rank, z_start_rank, &
       x_end_rank, y_end_rank, z_end_rank, &
       nx_rank, ny_rank, nz_rank, n_points_rank, n_points_trunc_rank, &
       nx_out_rank, ny_out_rank, nz_out_rank, &
       nx_trunc_out_rank, ny_trunc_out_rank, nz_trunc_out_rank, &
#endif
       dims_global(3), dimsc_global(3), &
       nx_global, ny_global, nz_global, n_points, n_points_trunc, &
       x_conj_w_fft, y_conj_w_fft, z_conj_w_fft

  real(C_DOUBLE) :: point_weight

contains

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
  !
  ! call lengthsAndOffsets(globlengths, isplit, ranklengths, rankoffsets)
  ! with inputs
  ! globlengths(3) : global length in each dimension
  ! isplit : dimension along which the domain is partitioned by ranks
  ! generates outputs
  ! ranklengths(3) : length in each dimension for this rank
  ! rankoffsets(3) : offset from global index in each dimension for this rank
  !
  subroutine lengthsAndOffsets(globlengths, isplit, ranklengths, rankoffsets)
    use mpi_utils_mod, only : partition_length, partition_offset
    implicit none
    integer, intent(in) :: globlengths(3), isplit
    integer, intent(out) :: ranklengths(3), rankoffsets(3)
    integer :: n_full, n_local, offset

    n_full = globlengths(isplit)
    n_local = partition_length(int(n_full))
    ranklengths = globlengths
    ranklengths(isplit) = int(n_local)
    offset = partition_offset(int(n_full))
    rankoffsets = 0
    rankoffsets(isplit) = int(offset)
  end subroutine lengthsAndOffsets
#endif

  ! Need to call initProblemDimensions with dims_global already set.
  subroutine initProblemDimensions(dims)
    use mpi_utils_mod, only : &
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
         mpi_size, mpi_rank, split_dim, split_out_dim, partition_length, partition_offset, MPIbarrier &
#endif
         i_am_mpi_master
    implicit none
    integer, dimension(3), intent(in) :: dims
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    integer :: offset
    integer :: n_local, n_global
    integer :: off_out, off_trunc_out
#endif

    dims_global = dims
    if (i_am_mpi_master) then
       print 210, dims_global
210    format ('Full global dimensions:', 3i5)
    endif
    
    dimsc_global = dims_global
    ! Truncate in dimension trunc_fortran_dim.
    dimsc_global(trunc_fortran_dim) = (dims_global(trunc_fortran_dim)/2) + 1
    if (i_am_mpi_master) then
       print 220, dimsc_global
220    format ('Truncated dimensions:  ', 3i5)
    endif
    
    nx_global = dims_global(1);
    ny_global = dims_global(2);
    nz_global = dims_global(3);
    n_points = product(dims_global)
    point_weight = 1._C_DOUBLE/real(n_points, C_DOUBLE)
    x_conj_w_fft = nx_global/2+1
    y_conj_w_fft = ny_global/2+1
    z_conj_w_fft = nz_global/2+1
    n_points_trunc = product(dimsc_global)

!    print *, 'dims_global =', dims_global, 'dimsc_global =', dimsc_global

#if defined (FFTX_CUDA) || defined(FFTX_HIP)

    call lengthsAndOffsets(dims_global, split_dim, dims_rank, off_set_rank)
    nx_rank = dims_rank(1);
    ny_rank = dims_rank(2);
    nz_rank = dims_rank(3);
    x_offset_rank = off_set_rank(1)
    y_offset_rank = off_set_rank(2)
    z_offset_rank = off_set_rank(3)

    xyz_start_rank = off_set_rank + 1
    x_start_rank = xyz_start_rank(1)
    y_start_rank = xyz_start_rank(2)
    z_start_rank = xyz_start_rank(3)
    xyz_end_rank = dims_rank + off_set_rank
    x_end_rank = xyz_end_rank(1)
    y_end_rank = xyz_end_rank(2)
    z_end_rank = xyz_end_rank(3)
    n_points_rank = product(dims_rank)

    call lengthsAndOffsets(dims_global, split_out_dim, dims_out_rank, off_out_rank)
    nx_out_rank = dims_out_rank(1);
    ny_out_rank = dims_out_rank(2);
    nz_out_rank = dims_out_rank(3);
    x_off_out_rank = off_out_rank(1)
    y_off_out_rank = off_out_rank(2)
    z_off_out_rank = off_out_rank(3)
    
    ! R2C is
    ! [K, N,       M]
    ! [K, N, M/2 + 1]
    !
    ! C2R is
    ! [K, N, M/2 + 1]
    ! [K, N,       M]

    dimsc_dist_global = dimsc_global
    if (trunc_fortran_dim .eq. split_out_dim) then
       ! Update dimsc_dist_global if necessary so that
       ! dimsc_dist_global(trunc_fortran_dim) is divisible by mpi_size.
       n_global = dimsc_dist_global(trunc_fortran_dim)
       n_local = n_global / mpi_size
       if (mpi_size * n_local .lt. n_global) then
          n_local = n_local + 1
          n_global = mpi_size * n_local
          dimsc_dist_global(trunc_fortran_dim) = n_global
       endif
    endif
    call lengthsAndOffsets(dimsc_dist_global, split_out_dim, dimsc_out_rank, off_trunc_out_rank)
    nx_trunc_out_rank = dimsc_out_rank(1)
    ny_trunc_out_rank = dimsc_out_rank(2)
    nz_trunc_out_rank = dimsc_out_rank(3)
    x_off_trunc_out_rank = off_trunc_out_rank(1)
    y_off_trunc_out_rank = off_trunc_out_rank(2)
    z_off_trunc_out_rank = off_trunc_out_rank(3)
    n_points_trunc_rank = product(dimsc_out_rank)
        
!    print *, 'mpi_rank', mpi_rank, 'dims_rank=', dims_rank, 'dimsc_out_rank=', dimsc_out_rank
!    print 100, mpi_rank, 1, nx_rank, 1, ny_rank, 1, nz_rank, &
!         x_start_rank, x_end_rank, y_start_rank, y_end_rank, z_start_rank, z_end_rank
!100 format ('Rank', i3, ' range local (', i3, ':', i3, ',', i3, ':', i3, ',', i3, ':', i3, ')', &
!         ' global (', i3, ':', i3, ',', i3, ':', i3, ',', i3, ':', i3, ')')
!    print 120, mpi_rank, 1, nx_trunc_out_rank, 1, ny_trunc_out_rank, 1, nz_trunc_out_rank, &
!         x_off_trunc_out_rank + 1, x_off_trunc_out_rank + nx_trunc_out_rank, &
!         y_off_trunc_out_rank + 1, y_off_trunc_out_rank + ny_trunc_out_rank, &
!         z_off_trunc_out_rank + 1, z_off_trunc_out_rank + nz_trunc_out_rank
!120 format ('Rank', i3, ' output range local (', i3, ':', i3, ',', i3, ':', i3, ',', i3, ':', i3, ')', &
!         ' global (', i3, ':', i3, ',', i3, ':', i3, ',', i3, ':', i3, ')')
    call MPIbarrier
#endif
  end subroutine

end module
