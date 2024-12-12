module mpi_utils_mod
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
  use mpi
#endif
  use, intrinsic :: iso_c_binding
  implicit none

#if defined (FFTX_CUDA) || defined(FFTX_HIP)
  ! MPI VARIABLES
  logical :: i_am_mpi_master
  integer mpi_rank, mpi_size, my_mpi_err
  integer, parameter :: mpi_master_rank = 0
  integer, parameter :: split_dim = 3 ! dimension on which to split arrays over MPI ranks
  integer, parameter :: split_out_dim = 1 ! dimension on which to split output arrays over MPI ranks
  integer, allocatable, dimension(:) :: mpi_local_x_all_proc, & ! the number of slices each process owns
                                        mpi_local_x_offset_all_proc, &
                                        mpi_local_y_all_proc, & ! the number of slices each process owns
                                        mpi_local_y_offset_all_proc, &
                                        mpi_local_z_all_proc, & ! the number of slices each process owns
                                        mpi_local_z_offset_all_proc
#else
  logical, parameter :: i_am_mpi_master = .true.
#endif
  
  contains

    subroutine init_mpi()
      implicit none
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      call MPI_INIT(my_mpi_err)
      if (my_mpi_err .ne. MPI_SUCCESS) then
         print *, 'MPI_INIT returned error code ', my_mpi_err
      endif
      call MPI_COMM_SIZE(MPI_COMM_WORLD, mpi_size, my_mpi_err)
      call MPI_COMM_RANK(MPI_COMM_WORLD, mpi_rank, my_mpi_err)
      allocate(mpi_local_x_all_proc(mpi_size), &
           mpi_local_x_offset_all_proc(mpi_size), &
           mpi_local_y_all_proc(mpi_size), &
           mpi_local_y_offset_all_proc(mpi_size), &
           mpi_local_z_all_proc(mpi_size), &
           mpi_local_z_offset_all_proc(mpi_size))
      i_am_mpi_master = .false.
      if (mpi_rank .eq. mpi_master_rank) i_am_mpi_master=.true.
#endif
    end subroutine init_mpi
    
    subroutine finalize_mpi()
      implicit none
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      call MPI_FINALIZE(my_mpi_err)
#endif
    end subroutine finalize_mpi
    
    real(C_DOUBLE) function MPIMaxReal(scalar)
      implicit none
      real(C_DOUBLE), intent(in) :: scalar
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      real(C_DOUBLE) :: my_mpi_max
      
      call MPI_Reduce(scalar, my_mpi_max, 1, MPI_DOUBLE_PRECISION, &
           MPI_MAX, mpi_master_rank, MPI_COMM_WORLD, my_mpi_err)
      call MPI_Bcast(my_mpi_max, 1, &
           MPI_DOUBLE_PRECISION, mpi_master_rank, MPI_COMM_WORLD, my_mpi_err)
      MPIMaxReal = my_mpi_max
#else
      MPIMaxReal = scalar
#endif
    end function MPIMaxReal
    
    real(C_DOUBLE) function MPISumReal(scalar)
      implicit none
      real(C_DOUBLE), intent(in) :: scalar
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      real(C_DOUBLE) :: my_mpi_sum
      
      call MPI_Reduce(scalar, my_mpi_sum, 1, MPI_DOUBLE_PRECISION, &
           MPI_SUM, mpi_master_rank, MPI_COMM_WORLD, my_mpi_err)
      call MPI_Bcast(my_mpi_sum, 1, &
           MPI_DOUBLE_PRECISION, mpi_master_rank, MPI_COMM_WORLD, my_mpi_err)
      MPISumReal = my_mpi_sum
#else
      MPISumReal = scalar
#endif
    end function MPISumReal
    
    complex(C_DOUBLE_COMPLEX) function MPISumComplex(scalar)
      implicit none
      complex(C_DOUBLE_COMPLEX), intent(in) :: scalar
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      complex(C_DOUBLE_COMPLEX) :: my_mpi_sum
      
      call MPI_Reduce(scalar, my_mpi_sum, 1, MPI_DOUBLE_COMPLEX, &
           MPI_SUM, mpi_master_rank, MPI_COMM_WORLD, my_mpi_err)
      call MPI_Bcast(my_mpi_sum, 1, &
           MPI_DOUBLE_COMPLEX, mpi_master_rank, MPI_COMM_WORLD, my_mpi_err)
      MPISumComplex = my_mpi_sum
#else
      MPISumComplex = scalar
#endif
    end function MPISumComplex
    
    integer function wraparound(i, mini, maxi)
      implicit none
      integer, intent(in) :: i, mini, maxi
      
      if ((i .ge. mini) .and. (i .le. maxi)) then
         wraparound = i
      elseif (i .eq. mini - 1) then
         wraparound = maxi
      elseif (i .eq. maxi + 1) then
         wraparound = mini
      else
         print 100, i, mini-1, maxi+1
100      format('Error in wraparound:', i4, ' not in range ', i4, ' to ', i4)
         stop
      endif
    end function wraparound
    
    
    subroutine MPIbarrier
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      call MPI_Barrier(MPI_COMM_WORLD, my_mpi_err)
#endif
    end subroutine MPIbarrier
    
    real(C_DOUBLE) function MPIMax(scalar)
      implicit none
      real(C_DOUBLE), intent(in)  :: scalar
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
      integer :: ierr
      
      call MPI_Allreduce(scalar, MPIMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, ierr)
#else
      MPIMax = scalar
#endif
    end function MPIMax
    
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
    subroutine MPIExchange3dReal(expanded, arr)
      implicit none
      ! If an array is sent to a subroutine with a lower bound that's not 1,
      ! the lower bound gets changed to 1!
      real(C_DOUBLE), dimension(:,:,:), intent(inout) :: expanded
      real(C_DOUBLE), dimension(:,:,:), intent(in) :: arr
      real(C_DOUBLE), dimension(:,:), allocatable :: slabShiftLeft, slabShiftRight
      integer :: minx, miny, minz, maxx, maxy, maxz, lx, ly, lz
      integer :: ix, iy, iz, jx, jy, jz, npts, inext, iprev
      integer :: tagShiftLeft, tagShiftRight
      integer, dimension(mpi_size) :: mpistatus
      
      minx = lbound(arr, 1)
      miny = lbound(arr, 2)
      minz = lbound(arr, 3)
      
      maxx = ubound(arr, 1)
      maxy = ubound(arr, 2)
      maxz = ubound(arr, 3)
      
      lx = size(expanded, 1)
      ly = size(expanded, 2)
      lz = size(expanded, 3)
      
      ! allocate(expanded(minx-1:maxx+1, miny-1:maxy+1, minz-1:maxz+1))
      ! First set expanded = arr with local wraparound.
      ! Will need to correct the first and last planes in the split dimension.
      do iz = minz-1, maxz+1
         jz = wraparound(iz, minz, maxz)
         do iy = miny-1, maxy+1
            jy = wraparound(iy, miny, maxy)
            do ix = minx-1, maxx+1
               jx = wraparound(ix, minx, maxx)
               expanded(ix+1, iy+1, iz+1) = arr(jx, jy, jz)
            enddo
         enddo
      enddo

      if (mpi_size .gt. 1) then
         ! Assume that array is distributed along split_dim = 3.
         ! allocate(slab(minx-1:maxx+1, miny-1:maxy+1))
         npts = lx * ly
         inext = MODULO(mpi_rank - 1, mpi_size)
         iprev = MODULO(mpi_rank + 1, mpi_size)
      
         ! Send leftmost slab with original data from this rank to previous rank.
         allocate(slabShiftLeft(lx, ly))
         slabShiftLeft = expanded(:, :, 2)
         ! Send rightmost slab with original data from this rank to next rank.
         allocate(slabShiftRight(lx, ly))
         slabShiftRight = expanded(:, :, lz - 1)

         tagShiftLeft = 0
         call MPI_SendRecv_replace(slabShiftLeft, npts, MPI_DOUBLE, &
              inext, tagShiftLeft, iprev, tagShiftLeft, &
              MPI_COMM_WORLD, mpistatus, my_mpi_err)

         tagShiftRight = 1
         call MPI_SendRecv_replace(slabShiftRight, npts, MPI_DOUBLE, &
              iprev, tagShiftRight, inext, tagShiftRight, &
              MPI_COMM_WORLD, mpistatus, my_mpi_err)

         expanded(:, :, lz) = slabShiftLeft
         expanded(:, :, 1) = slabShiftRight
      endif
    end subroutine MPIExchange3dReal
    
    subroutine MPIExchange3dComplex(expanded, arr)
      implicit none
      ! complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(out), allocatable :: expanded
      ! If an array is sent to a subroutine with a lower bound that's not 1,
      ! the lower bound gets changed to 1!
      complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(inout) :: expanded
      complex(C_DOUBLE_COMPLEX), dimension(:,:,:), intent(in) :: arr
      complex(C_DOUBLE_COMPLEX), dimension(:,:), allocatable :: slabShiftLeft, slabShiftRight
      integer :: minx, miny, minz, maxx, maxy, maxz, lx, ly, lz
      integer :: ix, iy, iz, jx, jy, jz, npts, inext, iprev
      integer :: tagShiftLeft, tagShiftRight
      integer, dimension(mpi_size) :: mpistatus
      
      minx = lbound(arr, 1)
      miny = lbound(arr, 2)
      minz = lbound(arr, 3)
      
      maxx = ubound(arr, 1)
      maxy = ubound(arr, 2)
      maxz = ubound(arr, 3)
      
      lx = size(expanded, 1)
      ly = size(expanded, 2)
      lz = size(expanded, 3)
      
      ! allocate(expanded(minx-1:maxx+1, miny-1:maxy+1, minz-1:maxz+1))
      ! First set expanded = arr with local wraparound.
      ! Will need to correct the first and last planes in the split dimension.
      do iz = minz-1, maxz+1
         jz = wraparound(iz, minz, maxz)
         do iy = miny-1, maxy+1
            jy = wraparound(iy, miny, maxy)
            do ix = minx-1, maxx+1
               jx = wraparound(ix, minx, maxx)
               expanded(ix+1, iy+1, iz+1) = arr(jx, jy, jz)
            enddo
         enddo
      enddo
      
      if (mpi_size .gt. 1) then
         ! Assume that array is distributed along split_dim = 3.
         ! allocate(slab(minx-1:maxx+1, miny-1:maxy+1))
         npts = lx * ly
         inext = MODULO(mpi_rank - 1, mpi_size)
         iprev = MODULO(mpi_rank + 1, mpi_size)

         ! Send leftmost slab with original data from this rank to previous rank.
         allocate(slabShiftLeft(lx, ly))
         slabShiftLeft = expanded(:, :, 2)
         ! Send rightmost slab with original data from this rank to next rank.
         allocate(slabShiftRight(lx, ly))
         slabShiftRight = expanded(:, :, lz - 1)
         
         tagShiftLeft = 0
         call MPI_SendRecv_replace(slabShiftLeft, npts, MPI_DOUBLE_COMPLEX, &
              inext, tagShiftLeft, iprev, tagShiftLeft, &
              MPI_COMM_WORLD, mpistatus, my_mpi_err)

         tagShiftRight = 1
         call MPI_SendRecv_replace(slabShiftRight, npts, MPI_DOUBLE_COMPLEX, &
              iprev, tagShiftRight, inext, tagShiftRight, &
              MPI_COMM_WORLD, mpistatus, my_mpi_err)

         expanded(:, :, lz) = slabShiftLeft
         expanded(:, :, 1) = slabShiftRight
      endif
    end subroutine MPIExchange3dComplex
    
    integer function partition_length(n)
      implicit none
      integer, intent(in) :: n
      integer :: remainder

      partition_length = n / mpi_size
      remainder = mod(n, mpi_size)
      if (remainder .gt. 0) then
         ! the condition should really be (mpi_rank .lt. remainder)
         ! so that the sum over all ranks is equal to n,
         ! but we follow what fftx_plan_distributed_1d_spiral does,
         ! where sum over all ranks exceeds n,
         ! because it does the copying between buffers.
         partition_length = partition_length + 1
      endif
    end function partition_length
    
    integer function partition_offset(n)
      implicit none
      integer, intent(in) :: n
      integer :: remainder, length
      
      length = partition_length(n)
      partition_offset = mpi_rank * length
      !! if we wanted the sum over all ranks equal to n, 
      !! then we'd make the modification below, but that's not what
      !! fftx_plan_distributed_1d_spiral does.
      ! remainder = mod(n, mpi_size)
      ! if (mpi_rank .ge. remainder) then
      !    partition_offset = partition_offset + remainder
      ! endif
    end function partition_offset
    
#endif

end module mpi_utils_mod
