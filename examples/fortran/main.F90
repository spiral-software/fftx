PROGRAM FFTX_CONVOLUTION
  use mpi_utils_mod
  use problem_dimensions_mod, only : initProblemDimensions
  use data_functions_mod
  use transform_tests_mod
  use convolution_mod
  implicit none

  DOUBLE PRECISION :: starttime, endtime, time, time_max
  character(len=32) :: inputarg
  integer :: nargs, iostat, i
  integer, dimension(3) :: dims

  ! Initialize MPI, if using.
  call init_mpi()

  ! Get input size.
  
  nargs = command_argument_count()
  if (nargs .lt. 3) then
     if (i_am_mpi_master) then
        print *, 'Error: command-line arguments must include 3 dimensions, nx ny nz'
     endif
     stop
  endif
    
  do i = 1, 3
     call get_command_argument(i, inputarg)
     read (inputarg, *, iostat=iostat) dims(i)
     if (iostat .gt. 0) then
        if (i_am_mpi_master) then
           print *, 'Error: command-line arguments must include 3 dimensions, nx ny nz'
        endif
        stop
     endif
  enddo

  call initProblemDimensions(dims)

  if (i_am_mpi_master) then
     ! Run these single-node tests on the MPI master rank only.
     
     print *, '================= single MDDFT test ==========='
     call CPU_TIME(starttime)
     call singleMDDFTTest()
     call CPU_TIME(endtime)
     time = endtime - starttime
     write(*,*) 'single MDDFT test time:', time
  
     print *, '================= single IMDDFT test ==========='
     call CPU_TIME(starttime)
     call singleIMDDFTTest()
     call CPU_TIME(endtime)
     time = endtime - starttime
     write(*,*) 'single IMDDFT test time:', time
  
     print *, '================= single MDPRDFT test ==========='
     call CPU_TIME(starttime)
     call singleMDPRDFTTest()
     call CPU_TIME(endtime)
     time = endtime - starttime
     write(*,*) 'single MDPRDFT test time:', time
  
     print *, '================= single IMDPRDFT test ==========='
     call CPU_TIME(starttime)
     call singleIMDPRDFTTest()
     call CPU_TIME(endtime)
     time = endtime - starttime
     write(*,*) 'single IMDPRDFT test time:', time

     print *, '================= single COMPLEX convolution test ==========='
     call CPU_TIME(starttime)
     call singleComplexConvolutionTest()
     call CPU_TIME(endtime)
     time = endtime - starttime
     write(*,*) 'single complex convolution test time:', time
  
     print *, '================= single REAL convolution test =============='
     call CPU_TIME(starttime)
     call singleRealConvolutionTest()
     call CPU_TIME(endtime)
     time = endtime - starttime
     write(*,*) 'single real convolution test time:', time
  endif

#if defined (FFTX_CUDA) || defined(FFTX_HIP)

  if (i_am_mpi_master) then
     print *, '================= distributed MDDFT test ==========='
  endif
  starttime = MPI_Wtime()
  call distMDDFTTest()
  endtime   = MPI_Wtime()
  time = endtime - starttime
  time_max = MPIMaxReal(time)
  if (i_am_mpi_master) then
     write(*,*) 'distributed MDDFT test time:', time_max
  endif
  
  if (i_am_mpi_master) then
     print *, '================= distributed IMDDFT test ==========='
  endif
  starttime = MPI_Wtime()
  call distIMDDFTTest()
  endtime   = MPI_Wtime()
  time = endtime - starttime
  time_max = MPIMaxReal(time)
  if (i_am_mpi_master) then
     write(*,*) 'distributed IMDDFT test time:', time_max
  endif
  
  if (i_am_mpi_master) then
     print *, '================= distributed MDPRDFT test ==========='
  endif
  starttime = MPI_Wtime()
  call distMDPRDFTTest()
  endtime   = MPI_Wtime()
  time = endtime - starttime
  time_max = MPIMaxReal(time)
  if (i_am_mpi_master) then
     write(*,*) 'distributed MDPRDFT test time:', time_max
  endif
  
  if (i_am_mpi_master) then
     print *, '================= distributed IMDPRDFT test ==========='
  endif
  starttime = MPI_Wtime()
  call distIMDPRDFTTest()
  endtime   = MPI_Wtime()
  time = endtime - starttime
  time_max = MPIMaxReal(time)
  if (i_am_mpi_master) then
     write(*,*) 'distributed IMDPRDFT test time:', time_max
  endif
  
  if (i_am_mpi_master) then
     print *, '================= distributed COMPLEX convolution test ==========='
  endif
  starttime = MPI_Wtime()
  call distComplexConvolutionTest()
  endtime   = MPI_Wtime()
  time = endtime - starttime
  time_max = MPIMaxReal(time)
  if (i_am_mpi_master) then
     write(*,*) 'distributed complex convolution test time:', time_max
  endif

  if (i_am_mpi_master) then
     print *, '================= distributed REAL convolution test ==========='
  endif
  starttime = MPI_Wtime()
  call distRealConvolutionTest()
  endtime   = MPI_Wtime()
  time = endtime - starttime
  time_max = MPIMaxReal(time)
  if (i_am_mpi_master) then
     write(*,*) 'distributed real convolution test time:', time_max
  endif
  
  ! terminate program: finalize MPI
  call finalize_mpi()

#endif

END PROGRAM FFTX_CONVOLUTION
