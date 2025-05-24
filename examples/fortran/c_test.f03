!!
!!  Copyright (c) 2018-2025, Carnegie Mellon University
!!  All rights reserved.
!!
!!  See LICENSE file for full information.
!!

  type, bind(C) :: mddft_holder
     type(C_PTR) :: dev_out, dev_in, dev_sym, problem
  end type mddft_holder

  type, bind(C) :: imddft_holder
     type(C_PTR) :: dev_out, dev_in, dev_sym, problem
  end type imddft_holder

  type, bind(C) :: mdprdft_holder
     integer(C_INT) :: npts, nptsTrunc
     type(C_PTR) :: dev_out, dev_in, dev_sym, problem
  end type mdprdft_holder

  type, bind(C) :: imdprdft_holder
     integer(C_INT) :: npts, nptsTrunc
     type(C_PTR) :: dev_out, dev_in, dev_sym, problem
  end type imdprdft_holder

  type, bind(C) :: mddft_dist_holder
     type(C_PTR) :: dev_out, dev_in, dev_sym, plan
  end type mddft_dist_holder

  type, bind(C) :: imddft_dist_holder
     type(C_PTR) :: dev_out, dev_in, dev_sym, plan
  end type imddft_dist_holder

  type, bind(C) :: mdprdft_dist_holder
     integer(C_INT) :: npts, nptsTrunc
     type(C_PTR) :: dev_out, dev_in, dev_sym, plan
  end type mdprdft_dist_holder

  type, bind(C) :: imdprdft_dist_holder
     integer(C_INT) :: npts, nptsTrunc
     type(C_PTR) :: dev_out, dev_in, dev_sym, plan
  end type imdprdft_dist_holder

interface

   subroutine fftx_plan_mddft(holder, M, N, K) &
        bind(C, name='fftx_plan_mddft_shim')
     use iso_c_binding, only: C_INT, C_PTR
     import :: mddft_holder
     type(mddft_holder) :: holder
     integer(C_INT), value :: M, N, K
   end subroutine fftx_plan_mddft

   subroutine fftx_plan_imddft(holder, M, N, K) &
        bind(C, name='fftx_plan_imddft_shim')
     use iso_c_binding, only: C_INT, C_PTR
     import :: imddft_holder
     type(imddft_holder) :: holder
     integer(C_INT), value :: M, N, K
   end subroutine fftx_plan_imddft

   subroutine fftx_plan_mdprdft(holder, M, N, K, npts, nptsTrunc) &
        bind(C, name='fftx_plan_mdprdft_shim')
     use iso_c_binding, only: C_INT, C_PTR
     import :: mdprdft_holder
     type(mdprdft_holder) :: holder
     integer(C_INT), value :: M, N, K, npts, nptsTrunc
   end subroutine fftx_plan_mdprdft

   subroutine fftx_plan_imdprdft(holder, M, N, K, npts, nptsTrunc) &
        bind(C, name='fftx_plan_imdprdft_shim')
     use iso_c_binding, only: C_INT, C_PTR
     import :: imdprdft_holder
     type(imdprdft_holder) :: holder
     integer(C_INT), value :: M, N, K, npts, nptsTrunc
   end subroutine fftx_plan_imdprdft


   subroutine fftx_execute_mddft(holder, out_buffer, in_buffer) &
        bind(C, name='fftx_execute_mddft_shim')
     use iso_c_binding, only: C_PTR
     import :: mddft_holder
     type(mddft_holder) :: holder
     type(C_PTR), value :: out_buffer, in_buffer
   end subroutine fftx_execute_mddft

   subroutine fftx_execute_imddft(holder, out_buffer, in_buffer) &
        bind(C, name='fftx_execute_imddft_shim')
     use iso_c_binding, only: C_PTR
     import :: imddft_holder
     type(imddft_holder) :: holder
     type(C_PTR), value :: out_buffer, in_buffer
   end subroutine fftx_execute_imddft

   subroutine fftx_execute_mdprdft(holder, out_buffer, in_buffer) &
        bind(C, name='fftx_execute_mdprdft_shim')
     use iso_c_binding, only: C_PTR
     import :: mdprdft_holder
     type(mdprdft_holder) :: holder
     type(C_PTR), value :: out_buffer, in_buffer
   end subroutine fftx_execute_mdprdft

   subroutine fftx_execute_imdprdft(holder, out_buffer, in_buffer) &
        bind(C, name='fftx_execute_imdprdft_shim')
     use iso_c_binding, only: C_PTR
     import :: imdprdft_holder
     type(imdprdft_holder) :: holder
     type(C_PTR), value :: out_buffer, in_buffer
   end subroutine fftx_execute_imdprdft


   subroutine fftx_plan_destroy_mddft(holder) &
        bind(C, name='fftx_plan_destroy_mddft_shim')
     use iso_c_binding, only: C_PTR
     import :: mddft_holder
     type(mddft_holder) :: holder
   end subroutine fftx_plan_destroy_mddft

   subroutine fftx_plan_destroy_imddft(holder) &
        bind(C, name='fftx_plan_destroy_imddft_shim')
     use iso_c_binding, only: C_PTR
     import :: imddft_holder
     type(imddft_holder) :: holder
   end subroutine fftx_plan_destroy_imddft

   subroutine fftx_plan_destroy_mdprdft(holder) &
        bind(C, name='fftx_plan_destroy_mdprdft_shim')
     use iso_c_binding, only: C_PTR
     import :: mdprdft_holder
     type(mdprdft_holder) :: holder
   end subroutine fftx_plan_destroy_mdprdft

   subroutine fftx_plan_destroy_imdprdft(holder) &
        bind(C, name='fftx_plan_destroy_imdprdft_shim')
     use iso_c_binding, only: C_PTR
     import :: imdprdft_holder
     type(imdprdft_holder) :: holder
   end subroutine fftx_plan_destroy_imdprdft


   type(C_PTR) function fftx_plan_distributed(p, M, N, K, batch, is_embedded, is_complex) &
        bind(C, name='fftx_plan_distributed_shim')
     use iso_c_binding, only: C_INT, C_BOOL, C_PTR
     integer(C_INT), value :: p, M, N, K, batch
     logical(C_BOOL), value:: is_embedded, is_complex
   end function fftx_plan_distributed

   subroutine fftx_plan_mddft_dist(holder, p, M, N, K, npts) &
        bind(C, name='fftx_plan_mddft_dist_shim')
     use iso_c_binding, only: C_INT, C_PTR
     import :: mddft_dist_holder
     type(mddft_dist_holder) :: holder
     integer(C_INT), value :: p, M, N, K, npts
   end subroutine fftx_plan_mddft_dist

   subroutine fftx_plan_imddft_dist(holder, p, M, N, K, npts) &
        bind(C, name='fftx_plan_imddft_dist_shim')
     use iso_c_binding, only: C_INT, C_PTR
     import :: imddft_dist_holder
     type(imddft_dist_holder) :: holder
     integer(C_INT), value :: p, M, N, K, npts
   end subroutine fftx_plan_imddft_dist

   subroutine fftx_plan_mdprdft_dist(holder, p, M, N, K, npts, nptsTrunc) &
        bind(C, name='fftx_plan_mdprdft_dist_shim')
     use iso_c_binding, only: C_INT, C_PTR
     import :: mdprdft_dist_holder
     type(mdprdft_dist_holder) :: holder
     integer(C_INT), value :: p, M, N, K, npts, nptsTrunc
   end subroutine fftx_plan_mdprdft_dist

   subroutine fftx_plan_imdprdft_dist(holder, p, M, N, K, npts, nptsTrunc) &
        bind(C, name='fftx_plan_imdprdft_dist_shim')
     use iso_c_binding, only: C_INT, C_PTR
     import :: imdprdft_dist_holder
     type(imdprdft_dist_holder) :: holder
     integer(C_INT), value :: p, M, N, K, npts, nptsTrunc
   end subroutine fftx_plan_imdprdft_dist

   
   subroutine fftx_execute_mddft_dist(holder, out_buffer, in_buffer) &
        bind(C, name='fftx_execute_mddft_dist_shim')
     use iso_c_binding, only: C_PTR
     import :: mddft_dist_holder
     type(mddft_dist_holder) :: holder
     type(C_PTR), value :: out_buffer, in_buffer
   end subroutine fftx_execute_mddft_dist

   subroutine fftx_execute_imddft_dist(holder, out_buffer, in_buffer) &
        bind(C, name='fftx_execute_imddft_dist_shim')
     use iso_c_binding, only: C_PTR
     import :: imddft_dist_holder
     type(imddft_dist_holder) :: holder
     type(C_PTR), value :: out_buffer, in_buffer
   end subroutine fftx_execute_imddft_dist

   subroutine fftx_execute_mdprdft_dist(holder, out_buffer, in_buffer) &
        bind(C, name='fftx_execute_mdprdft_dist_shim')
     use iso_c_binding, only: C_PTR
     import :: mdprdft_dist_holder
     type(mdprdft_dist_holder) :: holder
     type(C_PTR), value :: out_buffer, in_buffer
   end subroutine fftx_execute_mdprdft_dist

   subroutine fftx_execute_imdprdft_dist(holder, out_buffer, in_buffer) &
        bind(C, name='fftx_execute_imdprdft_dist_shim')
     use iso_c_binding, only: C_PTR
     import :: imdprdft_dist_holder
     type(imdprdft_dist_holder) :: holder
     type(C_PTR), value :: out_buffer, in_buffer
   end subroutine fftx_execute_imdprdft_dist

   
   subroutine fftx_plan_destroy_mddft_dist(holder) &
        bind(C, name='fftx_plan_destroy_mddft_dist_shim')
     use iso_c_binding, only: C_PTR
     import :: mddft_dist_holder
     type(mddft_dist_holder) :: holder
   end subroutine fftx_plan_destroy_mddft_dist

   subroutine fftx_plan_destroy_imddft_dist(holder) &
        bind(C, name='fftx_plan_destroy_imddft_dist_shim')
     use iso_c_binding, only: C_PTR
     import :: imddft_dist_holder
     type(imddft_dist_holder) :: holder
   end subroutine fftx_plan_destroy_imddft_dist

   subroutine fftx_plan_destroy_mdprdft_dist(holder) &
        bind(C, name='fftx_plan_destroy_mdprdft_dist_shim')
     use iso_c_binding, only: C_PTR
     import :: mdprdft_dist_holder
     type(mdprdft_dist_holder) :: holder
   end subroutine fftx_plan_destroy_mdprdft_dist

   subroutine fftx_plan_destroy_imdprdft_dist(holder) &
        bind(C, name='fftx_plan_destroy_imdprdft_dist_shim')
     use iso_c_binding, only: C_PTR
     import :: imdprdft_dist_holder
     type(imdprdft_dist_holder) :: holder
   end subroutine fftx_plan_destroy_imdprdft_dist

end interface
