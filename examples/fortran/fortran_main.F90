PROGRAM FFTX_CONVOLUTION
  use mpi_utils_mod
!  use problem_dimensions_mod, only : initProblemDimensions
!  use data_functions_mod
  use transform_tests_mod
  use convolution_mod, only : singleRealConvolutionTest, singleComplexConvolutionTest
#if defined (FFTX_CUDA) || defined(FFTX_HIP)
  use convolution_mod, only : distRealConvolutionTest, distComplexConvolutionTest
#endif
  
  implicit none

  DOUBLE PRECISION :: starttime, endtime, time, time_max
  character(len=32) :: inputarg, progname
  integer :: nargs, iostat, i, ier
  integer :: status, itns, itn
  integer, dimension(3) :: dims
  logical :: gotdims

  status = 0

  ! Initialize MPI, if using.
  call init_mpi()

  ! Set default number of iterations.
  itns = 1
  
  call get_command_argument(0, progname)
  nargs = command_argument_count()
  gotdims = .false.
  i = 1
  do while (i .le. nargs)
     call get_command_argument(i, inputarg)
     i = i + 1
     if (inputarg(1:1) .eq. '-') then
        if (inputarg(2:2) .eq. 's') then
           call get_command_argument(i, inputarg)
           i = i + 1
           call getDimsFromString(inputarg, dims, ier)
           if (ier .eq. 0) then
              gotdims = .true.
           else
              status = ier
           endif
        elseif (inputarg(2:2) .eq. 'i') then
           call get_command_argument(i, inputarg)
           i = i + 1
           read (inputarg, *, iostat=iostat) itns
           if (iostat .ne. 0) then
              status = iostat
           endif
        elseif (inputarg(2:2) .eq. 'h') then
           if (i_am_mpi_master) then
              print *, 'Usage: ', trim(progname), ' [ -i iterations ] [ -s MMxNNxKK ] [ -h (print help message) ]'
           endif
           stop status
        else
           if (i_am_mpi_master) then
              print *, trim(progname), ': ignoring unknown argument ', inputarg
           endif
        endif
     endif
  enddo
  if (status .eq. 0 .and. .not. gotdims) then
     status = -1
  endif

  if (status .ne. 0) then
     if (i_am_mpi_master) then
        print *, 'Usage: ', trim(progname), ' [ -i iterations ] [ -s MMxNNxKK ] [ -h (print help message) ]'
     endif
     stop status
  endif
 
  call initProblemDimensions(dims)

100 format (1x, '================= ', a, ',', i3, ' iterations ===========')

  if (i_am_mpi_master) then
     ! Run these single-node tests on the MPI master rank only.

     print 100, 'single MDDFT test', itns
     do itn = 1, itns
        call CPU_TIME(starttime)
        call singleMDDFTTest()
        call CPU_TIME(endtime)
        time = endtime - starttime
        write(*,*) 'single MDDFT test time:', time
     enddo
  
     print 100, 'single IMDDFT test', itns
     do itn = 1, itns
        call CPU_TIME(starttime)
        call singleIMDDFTTest()
        call CPU_TIME(endtime)
        time = endtime - starttime
        write(*,*) 'single IMDDFT test time:', time
     enddo

     print 100, 'single MDPRDFT test', itns
     do itn = 1, itns
        call CPU_TIME(starttime)
        call singleMDPRDFTTest()
        call CPU_TIME(endtime)
        time = endtime - starttime
        write(*,*) 'single MDPRDFT test time:', time
     enddo

     print 100, 'single IMDPRDFT test', itns
     do itn = 1, itns
        call CPU_TIME(starttime)
        call singleIMDPRDFTTest()
        call CPU_TIME(endtime)
        time = endtime - starttime
        write(*,*) 'single IMDPRDFT test time:', time
     enddo

     print 100, 'single COMPLEX convolution test', itns
     do itn = 1, itns
        call CPU_TIME(starttime)
        status = status + singleComplexConvolutionTest()
        call CPU_TIME(endtime)
        time = endtime - starttime
        write(*,*) 'single complex convolution test time:', time
     enddo

     print 100, 'single REAL convolution test', itns
     do itn = 1, itns
        call CPU_TIME(starttime)
        status = status + singleRealConvolutionTest()
        call CPU_TIME(endtime)
        time = endtime - starttime
        write(*,*) 'single real convolution test time:', time
     enddo
  endif

#if defined (FFTX_CUDA) || defined(FFTX_HIP)

  if (i_am_mpi_master) then
     print 100, 'distributed MDDFT test', itns
  endif
  do itn = 1, itns
     starttime = MPI_Wtime()
     call distMDDFTTest()
     endtime   = MPI_Wtime()
     time = endtime - starttime
     time_max = MPIMaxReal(time)
     if (i_am_mpi_master) then
        write(*,*) 'distributed MDDFT test time:', time_max
     endif
  enddo
  
  if (i_am_mpi_master) then
     print 100, 'distributed IMDDFT test', itns
  endif
  do itn = 1, itns
     starttime = MPI_Wtime()
     call distIMDDFTTest()
     endtime   = MPI_Wtime()
     time = endtime - starttime
     time_max = MPIMaxReal(time)
     if (i_am_mpi_master) then
        write(*,*) 'distributed IMDDFT test time:', time_max
     endif
  enddo
  
  if (i_am_mpi_master) then
     print 100, 'distributed MDPRDFT test', itns
  endif
  do itn = 1, itns
     starttime = MPI_Wtime()
     call distMDPRDFTTest()
     endtime   = MPI_Wtime()
     time = endtime - starttime
     time_max = MPIMaxReal(time)
     if (i_am_mpi_master) then
        write(*,*) 'distributed MDPRDFT test time:', time_max
     endif
  enddo
  
  if (i_am_mpi_master) then
     print 100, 'distributed IMDPRDFT test', itns
  endif
  do itn = 1, itns
     starttime = MPI_Wtime()
     call distIMDPRDFTTest()
     endtime   = MPI_Wtime()
     time = endtime - starttime
     time_max = MPIMaxReal(time)
     if (i_am_mpi_master) then
        write(*,*) 'distributed IMDPRDFT test time:', time_max
     endif
  enddo
  
  if (i_am_mpi_master) then
     print 100, 'distributed COMPLEX convolution test', itns
  endif
  do itn = 1, itns
     starttime = MPI_Wtime()
     status = status + distComplexConvolutionTest()
     endtime   = MPI_Wtime()
     time = endtime - starttime
     time_max = MPIMaxReal(time)
     if (i_am_mpi_master) then
        write(*,*) 'distributed complex convolution test time:', time_max
     endif
  enddo

  if (i_am_mpi_master) then
     print 100, 'distributed REAL convolution test', itns
  endif
  do itn = 1, itns
     starttime = MPI_Wtime()
     status = status + distRealConvolutionTest()
     endtime   = MPI_Wtime()
     time = endtime - starttime
     time_max = MPIMaxReal(time)
     if (i_am_mpi_master) then
        write(*,*) 'distributed real convolution test time:', time_max
     endif
  enddo
  
  ! terminate program: finalize MPI
  call finalize_mpi()

#endif

  stop status

END PROGRAM FFTX_CONVOLUTION
