!!
!!  Copyright (c) 2018-2025, Carnegie Mellon University
!!  All rights reserved.
!!
!!  See LICENSE file for full information.
!!

module fft_mod
  use, intrinsic :: iso_c_binding
  use problem_dimensions_mod, only : &
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
       nx_rank, ny_rank, nz_rank, n_points_rank, n_points_trunc_rank, &
#endif
       dims_global, n_points, n_points_trunc
  use, intrinsic :: iso_c_binding
  implicit none
  include 'c_test.f03'

  type fftx_3D_mddft
     type(mddft_holder) :: holder
   contains
     procedure :: init => init_fftx_3D_mddft
     procedure :: execute => execute_fftx_3D_mddft
     procedure :: finalize => finalize_fftx_3D_mddft
  end type fftx_3D_mddft

  type fftx_3D_imddft
     type(imddft_holder) :: holder
   contains
     procedure :: init => init_fftx_3D_imddft
     procedure :: execute => execute_fftx_3D_imddft
     procedure :: finalize => finalize_fftx_3D_imddft
  end type fftx_3D_imddft

  type fftx_3D_mdprdft
     type(mdprdft_holder) :: holder
  contains
    !-> allocate the container and its data of size (nx,ny,nz)
    procedure :: init => init_fftx_3D_mdprdft
    procedure :: execute => execute_fftx_3D_mdprdft
    procedure :: finalize => finalize_fftx_3D_mdprdft
 end type fftx_3D_mdprdft

 type fftx_3D_imdprdft
     type(imdprdft_holder) :: holder
  contains
    !-> allocate the container and its data of size (nx,ny,nz)
    procedure :: init => init_fftx_3D_imdprdft
    procedure :: execute => execute_fftx_3D_imdprdft
    procedure :: finalize => finalize_fftx_3D_imdprdft
 end type fftx_3D_imdprdft

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
 
 type fftx_3D_mddft_dist
    type(mddft_dist_holder) :: holder
  contains
    !-> allocate the container and its data of size (nx,ny,nz)
    procedure :: init => init_fftx_3D_mddft_dist
    procedure :: execute => execute_fftx_3D_mddft_dist
    procedure :: finalize => finalize_fftx_3D_mddft_dist
 end type fftx_3D_mddft_dist

 type fftx_3D_imddft_dist
    type(imddft_dist_holder) :: holder
  contains
    !-> allocate the container and its data of size (nx,ny,nz)
    procedure :: init => init_fftx_3D_imddft_dist
    procedure :: execute => execute_fftx_3D_imddft_dist
    procedure :: finalize => finalize_fftx_3D_imddft_dist
 end type fftx_3D_imddft_dist

 type fftx_3D_mdprdft_dist
    type(mdprdft_dist_holder) :: holder
  contains
    !-> allocate the container and its data of size (nx,ny,nz)
    procedure :: init => init_fftx_3D_mdprdft_dist
    procedure :: execute => execute_fftx_3D_mdprdft_dist
    procedure :: finalize => finalize_fftx_3D_mdprdft_dist
 end type fftx_3D_mdprdft_dist

 type fftx_3D_imdprdft_dist
    type(imdprdft_dist_holder) :: holder
  contains
    !-> allocate the container and its data of size (nx,ny,nz)
    procedure :: init => init_fftx_3D_imdprdft_dist
    procedure :: execute => execute_fftx_3D_imdprdft_dist
    procedure :: finalize => finalize_fftx_3D_imdprdft_dist
 end type fftx_3D_imdprdft_dist

#endif
  
contains

  subroutine init_fftx_3D_mddft(this)
    implicit none
    class(fftx_3D_mddft), intent(inout) :: this
    integer :: nx, ny, nz
!    type(C_PTR) :: ptr

    nx = dims_global(1)
    ny = dims_global(2)
    nz = dims_global(3)
    
!    print 100, 'init_fftx_3D_mddft', nx, ny, nz
100 format (a40, ': [', i4, ', ', i4, ', ', i4, ']')

    ! N.B.! reversed the dimensions.
    call fftx_plan_mddft(this%holder, &
         int(nz, kind=4), int(ny, kind=4), int(nx, kind=4) &
         )
  end subroutine init_fftx_3D_mddft

  subroutine init_fftx_3D_imddft(this)
    implicit none
    class(fftx_3D_imddft), intent(inout) :: this
    integer :: nx, ny, nz
!    type(C_PTR) :: ptr

    nx = dims_global(1)
    ny = dims_global(2)
    nz = dims_global(3)
    
!    print 100, 'init_fftx_3D_imddft', nx, ny, nz
100 format (a40, ': [', i4, ', ', i4, ', ', i4, ']')

    ! N.B.! reversed the dimensions.
    call fftx_plan_imddft(this%holder, &
         int(nz, kind=4), int(ny, kind=4), int(nx, kind=4) &
      )
  end subroutine init_fftx_3D_imddft

  subroutine init_fftx_3D_mdprdft(this)
    implicit none
    class(fftx_3D_mdprdft), intent(inout) :: this
    integer :: nx, ny, nz
!    type(C_PTR) :: ptr

    nx = dims_global(1)
    ny = dims_global(2)
    nz = dims_global(3)
    
!    print 100, 'init_fftx_3D_mdprdft', nx, ny, nz, n_points, n_points_trunc
100 format (a40, ': [', i4, ', ', i4, ', ', i4, '] npts=', i6, ' nptsTrunc=', i6)

    ! N.B.! reversed the dimensions.
    call fftx_plan_mdprdft(this%holder, &
         int(nz, kind=4), int(ny, kind=4), int(nx, kind=4), &
         int(n_points, kind=4), int(n_points_trunc, kind=4) &
         )
  end subroutine init_fftx_3D_mdprdft

  subroutine init_fftx_3D_imdprdft(this)
    implicit none
    class(fftx_3D_imdprdft), intent(inout) :: this
    integer :: nx, ny, nz
!    type(C_PTR) :: ptr

    nx = dims_global(1)
    ny = dims_global(2)
    nz = dims_global(3)
    
!    print 100, 'init_fftx_3D_imdprdft', nx, ny, nz, n_points, n_points_trunc
100 format (a40, ': [', i4, ', ', i4, ', ', i4, '] npts=', i6, ' nptsTrunc=', i6)

    ! N.B.! reversed the dimensions.
    call fftx_plan_imdprdft(this%holder, &
         int(nz, kind=4), int(ny, kind=4), int(nx, kind=4), &
         int(n_points, kind=4), int(n_points_trunc, kind=4) &
         )
  end subroutine init_fftx_3D_imdprdft

  subroutine finalize_fftx_3D_mddft(this)
    class(fftx_3D_mddft), intent(inout) :: this

!    print *, 'in call to finalize_fftx_3D_mddft'
    call fftx_plan_destroy_mddft(this%holder)
  end subroutine finalize_fftx_3D_mddft

  subroutine finalize_fftx_3D_imddft(this)
    class(fftx_3D_imddft), intent(inout) :: this

!    print *, 'in call to finalize_fftx_3D_imddft'
    call fftx_plan_destroy_imddft(this%holder)
  end subroutine finalize_fftx_3D_imddft

  subroutine finalize_fftx_3D_mdprdft(this)
    class(fftx_3D_mdprdft), intent(inout) :: this

!    print *, 'in call to finalize_fftx_3D_mdprdft'
    call fftx_plan_destroy_mdprdft(this%holder)
  end subroutine finalize_fftx_3D_mdprdft

  subroutine finalize_fftx_3D_imdprdft(this)
    class(fftx_3D_imdprdft), intent(inout) :: this

!    print *, 'in call to finalize_fftx_3D_imdprdft'
    call fftx_plan_destroy_imdprdft(this%holder)
  end subroutine finalize_fftx_3D_imdprdft

  subroutine execute_fftx_3D_mddft(this, out_data, in_data)
    class(fftx_3D_mddft), intent(in) :: this
    complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(in), target :: in_data
    complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(out), target :: out_data
    type(C_PTR) :: in_cptr, out_cptr
    in_cptr = c_loc(in_data)
    out_cptr = c_loc(out_data)
!    print *, 'in call to execute_fftx_3D_mddft'
    call fftx_execute_mddft(this%holder, out_cptr, in_cptr)
  end subroutine execute_fftx_3D_mddft

  subroutine execute_fftx_3D_imddft(this, out_data, in_data)
    class(fftx_3D_imddft), intent(in) :: this
    complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(in), target :: in_data
    complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(out), target :: out_data
    type(C_PTR) :: in_cptr, out_cptr
    in_cptr = c_loc(in_data)
    out_cptr = c_loc(out_data)
!    print *, 'in call to execute_fftx_3D_imddft'
    call fftx_execute_imddft(this%holder, out_cptr, in_cptr)
  end subroutine execute_fftx_3D_imddft

  subroutine execute_fftx_3D_mdprdft(this, out_data, in_data)
    class(fftx_3D_mdprdft), intent(in) :: this
    real(C_DOUBLE), dimension(:,:,:), intent(in), target :: in_data
    complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(out), target :: out_data
    type(C_PTR) :: in_cptr, out_cptr
    in_cptr = c_loc(in_data)
    out_cptr = c_loc(out_data)
!    print *, 'in call to execute_fftx_3D_mdprdft'
    call fftx_execute_mdprdft(this%holder, out_cptr, in_cptr)
  end subroutine execute_fftx_3D_mdprdft

  subroutine execute_fftx_3D_imdprdft(this, out_data, in_data)
    class(fftx_3D_imdprdft), intent(in) :: this
    complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(in), target :: in_data
    real(C_DOUBLE), dimension(:,:,:), intent(out), target :: out_data
    type(C_PTR) :: in_cptr, out_cptr
    in_cptr = c_loc(in_data)
    out_cptr = c_loc(out_data)
!    print *, 'in call to execute_fftx_3D_imdprdft'
    call fftx_execute_imdprdft(this%holder, out_cptr, in_cptr)
  end subroutine execute_fftx_3D_imdprdft

#if defined (FFTX_CUDA) || defined(FFTX_HIP)

  subroutine init_fftx_3D_mddft_dist(this)
    use mpi_utils_mod, only : mpi_size
    implicit none
    class(fftx_3D_mddft_dist), intent(inout) :: this
    integer :: nx_fft, ny_fft, nz_fft, nx_local_fft
!    type(C_PTR) :: ptr

    nx_fft = dims_global(1)
    ny_fft = dims_global(2)
    nz_fft = dims_global(3)

!    print 100, 'init_fftx_3D_mddft_dist', nx_fft, ny_fft, nz_fft
100 format (a40, ': [', i4, ', ', i4, ', ', i4, ']')

    ! N.B.! reversed the dimensions.  Or not.
    call fftx_plan_mddft_dist(this%holder, &
         int(mpi_size, kind=4), &
         int(nx_fft, kind=4), int(ny_fft, kind=4), int(nz_fft, kind=4), &
         int(n_points_rank, kind=4) &
    )
  end subroutine init_fftx_3D_mddft_dist

  subroutine init_fftx_3D_imddft_dist(this)
    use mpi_utils_mod, only : mpi_size
    implicit none
    class(fftx_3D_imddft_dist), intent(inout) :: this
    integer :: nx_fft, ny_fft, nz_fft, nx_local_fft
!    type(C_PTR) :: ptr

    nx_fft = dims_global(1)
    ny_fft = dims_global(2)
    nz_fft = dims_global(3)

!    print 100, 'init_fftx_3D_imddft_dist', nx_fft, ny_fft, nz_fft
100 format (a40, ': [', i4, ', ', i4, ', ', i4, ']')

    ! N.B.! reversed the dimensions.  Or not.
    call fftx_plan_imddft_dist(this%holder, &
         int(mpi_size, kind=4), &
         int(nx_fft, kind=4), int(ny_fft, kind=4), int(nz_fft, kind=4), &
         int(n_points_rank, kind=4) &
    )
  end subroutine init_fftx_3D_imddft_dist
  
  subroutine init_fftx_3D_mdprdft_dist(this)
    use mpi_utils_mod, only : mpi_size
    implicit none
    class(fftx_3D_mdprdft_dist), intent(inout) :: this
    integer :: nx_fft, ny_fft, nz_fft, nx_local_fft
!    type(C_PTR) :: ptr

    nx_fft = dims_global(1)
    ny_fft = dims_global(2)
    nz_fft = dims_global(3)

!    print 100, 'init_fftx_3D_mdprdft_dist', mpi_rank, nx_fft, ny_fft, nz_fft, n_points_rank, n_points_trunc_rank
100 format (a40, ' at', i2, ': [', i4, ', ', i4, ', ', i4, '] pts', i4, 'pts_trunc', i4)

    ! N.B.! reversed the dimensions.  Or not.
    call fftx_plan_mdprdft_dist(this%holder, &
         int(mpi_size, kind=4), &
         int(nx_fft, kind=4), int(ny_fft, kind=4), int(nz_fft, kind=4), &
         int(n_points_rank, kind=4), int(n_points_trunc_rank, kind=4) &
    )
  end subroutine init_fftx_3D_mdprdft_dist

  subroutine init_fftx_3D_imdprdft_dist(this)
    use mpi_utils_mod, only : mpi_size
    implicit none
    class(fftx_3D_imdprdft_dist), intent(inout) :: this
    integer :: nx_fft, ny_fft, nz_fft, nx_local_fft
!    type(C_PTR) :: ptr

    nx_fft = dims_global(1)
    ny_fft = dims_global(2)
    nz_fft = dims_global(3)

!    print 100, 'init_fftx_3D_imdprdft_dist', mpi_rank, nx_fft, ny_fft, nz_fft, n_points_rank, n_points_trunc_rank
100 format (a40, ' at', i2, ': [', i4, ', ', i4, ', ', i4, '] pts', i4, 'pts_trunc', i4)

    ! N.B.! reversed the dimensions.  Or not.
    call fftx_plan_imdprdft_dist(this%holder, &
         int(mpi_size, kind=4), &
         int(nx_fft, kind=4), int(ny_fft, kind=4), int(nz_fft, kind=4), &
         int(n_points_rank, kind=4), int(n_points_trunc_rank, kind=4) &
    )
  end subroutine init_fftx_3D_imdprdft_dist
  
  subroutine finalize_fftx_3D_mddft_dist(this)
    class(fftx_3D_mddft_dist), intent(inout) :: this

!    print *, 'in call to finalize_fftx_3D_mddft_dist'
    call fftx_plan_destroy_mddft_dist(this%holder)
  end subroutine finalize_fftx_3D_mddft_dist

  subroutine finalize_fftx_3D_imddft_dist(this)
    class(fftx_3D_imddft_dist), intent(inout) :: this

    call fftx_plan_destroy_imddft_dist(this%holder)
  end subroutine finalize_fftx_3D_imddft_dist

  subroutine finalize_fftx_3D_mdprdft_dist(this)
    class(fftx_3D_mdprdft_dist), intent(inout) :: this

!    print *, 'in call to finalize_fftx_3D_mdprdft_dist'
    call fftx_plan_destroy_mdprdft_dist(this%holder)
  end subroutine finalize_fftx_3D_mdprdft_dist

  subroutine finalize_fftx_3D_imdprdft_dist(this)
    class(fftx_3D_imdprdft_dist), intent(inout) :: this

    call fftx_plan_destroy_imdprdft_dist(this%holder)
  end subroutine finalize_fftx_3D_imdprdft_dist

  subroutine execute_fftx_3D_mddft_dist(this, out_data, in_data)
    class(fftx_3D_mddft_dist), intent(in) :: this
    complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(in), target :: in_data
    complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(out), target :: out_data
    type(C_PTR) :: in_cptr, out_cptr
    in_cptr = c_loc(in_data)
    out_cptr = c_loc(out_data)
!    print *, 'in call to execute_fftx_3D_mddft_dist'
    call fftx_execute_mddft_dist(this%holder, out_cptr, in_cptr)
  end subroutine execute_fftx_3D_mddft_dist

  subroutine execute_fftx_3D_imddft_dist(this, out_data, in_data)
    class(fftx_3D_imddft_dist), intent(in) :: this
    complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(in), target :: in_data
    complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(out), target :: out_data
    type(C_PTR) :: in_cptr, out_cptr
    in_cptr = c_loc(in_data)
    out_cptr = c_loc(out_data)
!    print *, 'in call to execute_fftx_3D_mddft_dist'
    call fftx_execute_imddft_dist(this%holder, out_cptr, in_cptr)
  end subroutine execute_fftx_3D_imddft_dist

  subroutine execute_fftx_3D_mdprdft_dist(this, out_data, in_data)
    class(fftx_3D_mdprdft_dist), intent(in) :: this
    real(C_DOUBLE), dimension(:,:,:), intent(in), target :: in_data
    complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(out), target :: out_data
    type(C_PTR) :: in_cptr, out_cptr
    in_cptr = c_loc(in_data)
    out_cptr = c_loc(out_data)
!    print *, 'in call to execute_fftx_3D_mdprdft_dist'
    call fftx_execute_mdprdft_dist(this%holder, out_cptr, in_cptr)
  end subroutine execute_fftx_3D_mdprdft_dist

  subroutine execute_fftx_3D_imdprdft_dist(this, out_data, in_data)
    class(fftx_3D_imdprdft_dist), intent(in) :: this
    complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(in), target :: in_data
    real(C_DOUBLE), dimension(:,:,:), intent(out), target :: out_data
    type(C_PTR) :: in_cptr, out_cptr
    in_cptr = c_loc(in_data)
    out_cptr = c_loc(out_data)
!    print *, 'in call to execute_fftx_3D_mdprdft_dist'
    call fftx_execute_imdprdft_dist(this%holder, out_cptr, in_cptr)
  end subroutine execute_fftx_3D_imdprdft_dist

!  subroutine globalCoorFromLocal1D(x_local, offset, x_global)
!    implicit none
!    integer, intent(in) :: x_local, offset
!    integer, intent(out) :: x_global
!    x_global = x_local + offset
!  end subroutine
#endif

end module fft_mod
