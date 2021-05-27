#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <iostream>
#include <algorithm>

#include "DFT_80.fftx.codegen.hpp"
#include "IDFT_80.fftx.codegen.hpp"
#include "DFT_100.fftx.codegen.hpp"
#include "IDFT_100.fftx.codegen.hpp"
#include "DFT_224_224_100.fftx.codegen.hpp"
#include "IDFT_224_224_100.fftx.codegen.hpp"

#define THREADS 128
#define THREAD_BLOCKS 320
#define C_SPEED 1
#define EP0 1

// pack the data
__global__ void pack_data(int l,
			  int m,
			  int n,
			  double *input,
			  int l_is,
			  int m_is,
			  int n_is, 
			  double *output,
			  int l_os,
			  int m_os,
			  int n_os) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  for(int iter = id; iter < (l * m * n); iter += blockDim.x * gridDim.x) {
    int i = (iter % l);
    int j = (iter / l) % m;
    int k = (iter / (l * m)) % n;

    *(output + i + l_os * j + l_os * m_os * k) = *(input + i + l_is * j + l_is * m_is * k);
  }
}

// shift the data
__global__ void shift_data(int l,
			   int m,
			   int n,
			   cufftDoubleComplex *io,
			   int do_shift_i,
			   cufftDoubleComplex *shift_i,
			   int do_shift_j,
			   cufftDoubleComplex *shift_j,
			   int do_shift_k,
			   cufftDoubleComplex *shift_k) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  cufftDoubleComplex Z;
  Z.x = 1.0;
  Z.y = 0.0;
  
  for(int iter = id; iter < (l * m * n); iter += blockDim.x * gridDim.x) {
    int i = (iter % l);
    int j = (iter / l) % m;
    int k = (iter / (l * m)) % n;

    cufftDoubleComplex v_shift_i = (do_shift_i == 0) ? Z : *(shift_i + i);
    cufftDoubleComplex v_shift_j = (do_shift_j == 0) ? Z : *(shift_j + j);
    cufftDoubleComplex v_shift_k = (do_shift_k == 0) ? Z : *(shift_k + k);
    
    cufftDoubleComplex value = *(io + iter);
    cufftDoubleComplex result;

    result.x = value.x * v_shift_i.x - value.y * v_shift_i.y;
    result.y = value.x * v_shift_i.y + value.y * v_shift_i.x;

    result.x = result.x * v_shift_j.x - result.y * v_shift_j.y;
    result.y = result.x * v_shift_j.y + result.y * v_shift_j.x;

    result.x = result.x * v_shift_k.x - result.y * v_shift_k.y;
    result.y = result.x * v_shift_k.y + result.y * v_shift_k.x;
    
    *(io + iter) = result;
  }
}

// compute contraction
__global__ void compute_contraction(int l,
				    int m,
				    int n,
				    cufftDoubleComplex *io,
				    double *modified_ki_arr,
				    double *modified_kj_arr,
				    double *modified_kk_arr,
				    double *C_arr,
				    double *S_arr,
				    double *X1_arr,
				    double *X2_arr,
				    double *X3_arr) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  double c2 = C_SPEED * C_SPEED;
  double inv_ep0 = 1.0 / EP0;
  
  for(int iter = id; iter < (l * m * n); iter += blockDim.x * gridDim.x) {
    int i = (iter % l);
    int j = (iter / l) % m;
    int k = (iter / (l * m)) % n;

    // E and B fields
    cufftDoubleComplex Ex = *(io + 0 * l * m * n + iter);
    cufftDoubleComplex Ey = *(io + 1 * l * m * n + iter);
    cufftDoubleComplex Ez = *(io + 2 * l * m * n + iter);
    cufftDoubleComplex Bx = *(io + 3 * l * m * n + iter);
    cufftDoubleComplex By = *(io + 4 * l * m * n + iter);
    cufftDoubleComplex Bz = *(io + 5 * l * m * n + iter);

    // Shortcut for the values of J and rho
    cufftDoubleComplex Jx = *(io + 6 * l * m * n + iter);
    cufftDoubleComplex Jy = *(io + 7 * l * m * n + iter);
    cufftDoubleComplex Jz = *(io + 8 * l * m * n + iter);
    cufftDoubleComplex rho_old = *(io + 9 * l * m * n + iter);
    cufftDoubleComplex rho_new = *(io + 10 * l * m * n + iter);
    
    // k vector values, and coefficients
    double kx = *(modified_ki_arr + i);
    double ky = *(modified_kj_arr + j);
    double kz = *(modified_kk_arr + k);
    
    double C = *(C_arr + iter);
    double S_ck = *(S_arr + iter);
    double X1 = *(X1_arr + iter);
    double X2 = *(X2_arr + iter);
    double X3 = *(X3_arr + iter);

    cufftDoubleComplex ex, ey, ez, bx, by, bz;

    ex.x = C * Ex.x + S_ck * (-1.0 * c2 * (ky * Bz.y - kz * By.y) - inv_ep0 * Jx.x) + (X2 * rho_new.y - X3 * rho_old.y) * kx;
    ex.y = C * Ex.y + S_ck * (       c2 * (ky * Bz.x - kz * By.x) - inv_ep0 * Jx.y) - (X2 * rho_new.x - X3 * rho_old.x) * kx; 

    ey.x = C * Ey.x + S_ck * (-1.0 * c2 * (kz * Bx.y - kx * Bz.y) - inv_ep0 * Jy.x) + (X2 * rho_new.y - X3 * rho_old.y) * ky;
    ey.y = C * Ey.y + S_ck * (       c2 * (kz * Bx.x - kx * Bz.x) - inv_ep0 * Jy.y) - (X2 * rho_new.x - X3 * rho_old.x) * ky;

    ez.x = C * Ez.x + S_ck * (-1.0 * c2 * (kx * By.y - ky * Bx.y) - inv_ep0 * Jz.x) + (X2 * rho_new.y - X3 * rho_old.y) * kz;
    ez.y = C * Ez.y + S_ck * (       c2 * (kx * By.x - ky * Bx.x) - inv_ep0 * Jz.y) - (X2 * rho_new.x - X3 * rho_old.x) * kz;

    bx.x = C * Bx.x + S_ck * (ky * Ez.y - kz * Ey.y) - X1 * (ky * Jz.y - kz * Jy.y);
    bx.y = C * Bx.y - S_ck * (ky * Ez.x - kz * Ey.x) + X1 * (ky * Jz.x - kz * Jy.x);

    by.x = C * By.x + S_ck * (kz * Ex.y - kx * Ez.y) - X1 * (kz * Jx.y - kx * Jz.y);
    by.y = C * By.y - S_ck * (kz * Ex.x - kx * Ez.x) + X1 * (kz * Jx.x - kx * Jz.x);

    bz.x = C * Bz.x + S_ck * (kx * Ey.y - ky * Ex.y) - X1 * (kx * Jy.y - ky * Jx.y);
    bz.y = C * Bz.y - S_ck * (kx * Ey.x - ky * Ex.x) + X1 * (kx * Jy.x - ky * Jx.x);
    
    // Update E 
    *(io + 0 * l * m * n + iter) = ex;
    *(io + 1 * l * m * n + iter) = ey;
    *(io + 2 * l * m * n + iter) = ez;
    
    // Update B 
    *(io + 3 * l * m * n + iter) = bx;
    *(io + 4 * l * m * n + iter) = by;
    *(io + 5 * l * m * n + iter) = bz;
  }
}


// compute forward and inverse Fourier transforms
inline void __attribute__((always_inline)) compute_warp_forward_dft(cufftHandle plan,
								    int l,
								    int m,
								    int n,
								    double *input,
								    int l_is,
								    int m_is,
								    int n_is,
								    double *temp,
								    cufftDoubleComplex *output,
                                                                    cufftDoubleComplex *output_fftx,
								    int do_shift_i,
								    cufftDoubleComplex *shift_i,
								    int do_shift_j,
								    cufftDoubleComplex *shift_j,
								    int do_shift_k,
								    cufftDoubleComplex *shift_k) {
  // cufft implementation
  pack_data<<<THREAD_BLOCKS, THREADS>>>(l, m, n,
					input,
					l_is, m_is, n_is,
					temp,
					l, m, n);
  cufftExecD2Z(plan, temp, output);
  shift_data<<<THREAD_BLOCKS, THREADS>>>(l / 2 + 1, m, n,
					 output,
					 do_shift_i,
					 shift_i,
					 do_shift_j,
					 shift_j,
					 do_shift_k,
					 shift_k);

  // fftx implementation
  pack_data<<<THREAD_BLOCKS, THREADS>>>(l, m, n,
					input,
					l_is, m_is, n_is,
					temp,
					l, m, n);
  
  fftx::box_t<3> inputBox({{1,1,1}},{{l,m,n}});
  fftx::box_t<3> outputBox({{1,1,1}},{{l/2+1,m,n}});
  
  fftx::array_t<3,double> in(fftx::global_ptr<double>(temp), inputBox);
  fftx::array_t<3,std::complex<double>> out(fftx::global_ptr<std::complex<double>>((std::complex<double>*)output_fftx),outputBox);

  if(l==80 && m==80 && n==80)
    {
      DFT_80::transform(in, out, in);
    }
  else if(l==100 && m==100 && n==100)
    {
      DFT_100::transform(in, out, in);
    }
  else if(l==224 && m==224 && n==100)
    {
      DFT_224_224_100::transform(in, out, in);
    }
  else
    {
      std::cout<<"transform not found for FFTX "<<l<<" "<<m<<" "<<n<<"\n";
    }
  shift_data<<<THREAD_BLOCKS, THREADS>>>(l / 2 + 1, m, n,
					 output_fftx,
					 do_shift_i,
					 shift_i,
					 do_shift_j,
					 shift_j,
					 do_shift_k,
					 shift_k);
}

inline void __attribute__((always_inline)) compute_warp_inverse_dft(cufftHandle plan,
								    int l,
								    int m,
								    int n,
								    cufftDoubleComplex *input,
                                                                    cufftDoubleComplex *input_fftx,
								    double *temp,
								    double *output,
								    double *output_fftx,
								    int l_os,
								    int m_os,
								    int n_os,
								    int do_shift_i,
								    cufftDoubleComplex *shift_i,
								    int do_shift_j,
								    cufftDoubleComplex *shift_j,
								    int do_shift_k,
								    cufftDoubleComplex *shift_k) {
  // cufft implementation
  shift_data<<<THREAD_BLOCKS, THREADS>>>(l / 2 + 1, m, n,
					 input,
					 do_shift_i,
					 shift_i,
					 do_shift_j,
					 shift_j,
					 do_shift_k,
					 shift_k);
  cufftExecZ2D(plan, input, temp);
  pack_data<<<THREAD_BLOCKS, THREADS>>>(l, m, n,
					temp,
					l, m, n,
					output,
					l_os, m_os, n_os);

  // fftx implementation
  shift_data<<<THREAD_BLOCKS, THREADS>>>(l / 2 + 1, m, n,
					 input_fftx,
					 do_shift_i,
					 shift_i,
					 do_shift_j,
					 shift_j,
					 do_shift_k,
					 shift_k);

  fftx::box_t<3> inputBox({{1,1,1}},{{l/2+1,m,n}});
  fftx::box_t<3> outputBox({{1,1,1}},{{l,m,n}});

  fftx::array_t<3,std::complex<double>> in(fftx::global_ptr<std::complex<double>>((std::complex<double>*)input_fftx), inputBox);
  fftx::array_t<3,double> out(fftx::global_ptr<double>(temp), outputBox);
  
  if(l==80  && m==80 && n==80)
    {
      IDFT_80::transform(in, out, out);
    }
  else if(l==100 && m==100 && n==100)
    {
      IDFT_100::transform(in, out, out);
    }
  else if(l==224 && m==224 && n==100)
    {
      IDFT_224_224_100::transform(in, out, out);
    }
  else
    {
      std::cout<<"inverse transform not found for FFTX "<<l<<" "<<m<<" "<<n<<"\n";
    }
      
  pack_data<<<THREAD_BLOCKS, THREADS>>>(l, m, n,
					temp,
					l, m, n,
					output_fftx,
					l_os, m_os, n_os); 
}

// compute Spectral Solve
inline void __attribute__((always_inline)) compute_spectral_solve(int l,
								  int m,
								  int n,
								  cufftHandle plan_forward,
								  cufftHandle plan_inverse,
								  double *Ex_in,
								  double *Ey_in,
								  double *Ez_in,
								  double *Bx_in,
								  double *By_in,
								  double *Bz_in,
								  double *Jx,
								  double *Jy,
								  double *Jz,
								  double *rho_0,
								  double *rho_1,
								  cufftDoubleComplex *fshift_i,
								  cufftDoubleComplex *fshift_j,
								  cufftDoubleComplex *fshift_k,
								  double *temp0,
								  cufftDoubleComplex *temp1,
                                                                  cufftDoubleComplex *temp1_fftx,
								  double *modified_ki_arr,
								  double *modified_kj_arr,
								  double *modified_kk_arr,
								  double *C_arr,
								  double *S_arr,
								  double *X1_arr,
								  double *X2_arr,
								  double *X3_arr,
								  double *Ex_out,
								  double *Ey_out,
								  double *Ez_out,
								  double *Bx_out,
								  double *By_out,
								  double *Bz_out,
                                                                  double *Ex_out_fftx,
								  double *Ey_out_fftx,
								  double *Ez_out_fftx,
								  double *Bx_out_fftx,
								  double *By_out_fftx,
								  double *Bz_out_fftx,
								  cufftDoubleComplex *ishift_i,
								  cufftDoubleComplex *ishift_j,
								  cufftDoubleComplex *ishift_k) {
  // Ex, Ey, Ez fields
  compute_warp_forward_dft(plan_forward,
			   l,
			   m,
			   n,
			   Ex_in,
			   l, (m + 1), (n + 1),
			   (temp0 + 0),
			   (temp1 + 0 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 0 * (l / 2 + 1) * m * n),
			   0,
			   fshift_i,
			   1,
			   fshift_j,
			   1,
			   fshift_k);
  compute_warp_forward_dft(plan_forward,
			   l,
			   m,
			   n,
			   Ey_in,
			   (l + 1), m, (n + 1),
			   (temp0 + 0),
			   (temp1 + 1 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 1 * (l / 2 + 1) * m * n),
			   1,
			   fshift_i,
			   0,
			   fshift_j,
			   1,
			   fshift_k);
  compute_warp_forward_dft(plan_forward,
			   l,
			   m,
			   n,
			   Ez_in,
			   (l + 1), (m + 1), n,
			   (temp0 + 0),
			   (temp1 + 2 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 2 * (l / 2 + 1) * m * n),
			   1,
			   fshift_i,
			   1,
			   fshift_j,
			   0,
			   fshift_k);
  // Bx, By, Bz fields
  compute_warp_forward_dft(plan_forward,
			   l,
			   m,
			   n,
			   Bx_in,
			   (l + 1), m, n,
			   (temp0 + 0),
			   (temp1 + 3 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 3 * (l / 2 + 1) * m * n),
			   1,
			   fshift_i,
			   0,
			   fshift_j,
			   0,
			   fshift_k);
  compute_warp_forward_dft(plan_forward,
			   l,
			   m,
			   n,
			   By_in,
			   l, (m + 1), n,
			   (temp0 + 0),
			   (temp1 + 4 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 4 * (l / 2 + 1) * m * n),
			   0,
			   fshift_i,
			   1,
			   fshift_j,
			   0,
			   fshift_k);
  compute_warp_forward_dft(plan_forward,
			   l,
			   m,
			   n,
			   Bz_in,
			   l, m, (n + 1),
			   (temp0 + 0),
			   (temp1 + 5 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 5 * (l / 2 + 1) * m * n),
			   0,
			   fshift_i,
			   0,
			   fshift_j,
			   1,
			   fshift_k);
  // Jx, Jy, Jz fields
  compute_warp_forward_dft(plan_forward,
			   l,
			   m,
			   n,
			   Jx,
			   l, (m + 1), (n + 1),
			   (temp0 + 0),
			   (temp1 + 6 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 6 * (l / 2 + 1) * m * n),
			   0,
			   fshift_i,
			   1,
			   fshift_j,
			   1,
			   fshift_k);
  compute_warp_forward_dft(plan_forward,
			   l,
			   m,
			   n,
			   Jy,
			   (l + 1), m, (n + 1),
			   (temp0 + 0),
			   (temp1 + 7 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 7 * (l / 2 + 1) * m * n),
			   1,
			   fshift_i,
			   0,
			   fshift_j,
			   1,
			   fshift_k);
  compute_warp_forward_dft(plan_forward,
			   l,
			   m,
			   n,
			   Jz,
			   (l + 1), (m + 1), n,
			   (temp0 + 0),
			   (temp1 + 8 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 8 * (l / 2 + 1) * m * n),
			   1,
			   fshift_i,
			   1,
			   fshift_j,
			   0,
			   fshift_k);
  // rho_0, rho_1 fields
  compute_warp_forward_dft(plan_forward,
			   l,
			   m,
			   n,
			   rho_0,
			   l, m, n,
			   (temp0 + 0),
			   (temp1 + 9 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 9 * (l / 2 + 1) * m * n),
			   0,
			   fshift_i,
			   0,
			   fshift_j,
			   0,
			   fshift_k);
  compute_warp_forward_dft(plan_forward,
			   l,
			   m,
			   n,
			   rho_1,
			   l, m, n,
			   (temp0 + 0),
			   (temp1 + 10 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 10 * (l / 2 + 1) * m * n),
			   0,
			   fshift_i,
			   0,
			   fshift_j,
			   0,
			   fshift_k);
  // contraction
  compute_contraction<<<THREAD_BLOCKS, THREADS>>>((l / 2 + 1),
						  m,
						  n,
						  temp1,
						  modified_ki_arr,
						  modified_kj_arr,
						  modified_kk_arr,
						  C_arr,
						  S_arr,
						  X1_arr,
						  X2_arr,
						  X3_arr);
  compute_contraction<<<THREAD_BLOCKS, THREADS>>>((l / 2 + 1),
						  m,
						  n,
						  temp1_fftx,
						  modified_ki_arr,
						  modified_kj_arr,
						  modified_kk_arr,
						  C_arr,
						  S_arr,
						  X1_arr,
						  X2_arr,
						  X3_arr);
  
  // Ex, Ey, Ez fields
  compute_warp_inverse_dft(plan_inverse,
			   l,
			   m,
			   n,
			   (temp1 + 0 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 0 * (l / 2 + 1) * m * n),
			   temp0,
			   Ex_out,
			   Ex_out_fftx,
			   l, (m + 1), (n + 1),
			   0,
			   ishift_i,
			   1,
			   ishift_j,
			   1,
			   ishift_k);
  compute_warp_inverse_dft(plan_inverse,
			   l,
			   m,
			   n,
			   (temp1 + 1 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 1 * (l / 2 + 1) * m * n),
			   temp0,
			   Ey_out,
			   Ey_out_fftx,
			   (l + 1), m, (n + 1),
			   1,
			   ishift_i,
			   0,
			   ishift_j,
			   1,
			   ishift_k);
  compute_warp_inverse_dft(plan_inverse,
			   l,
			   m,
			   n,
			   (temp1 + 2 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 2 * (l / 2 + 1) * m * n),
			   temp0,
			   Ez_out,
			   Ez_out_fftx,
			   (l + 1), (m + 1), n,
			   1,
			   ishift_i,
			   1,
			   ishift_j,
			   0,
			   ishift_k);
  // Bx, By, Bz fields
  compute_warp_inverse_dft(plan_inverse,
			   l,
			   m,
			   n,
			   (temp1 + 3 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 3 * (l / 2 + 1) * m * n),
			   temp0,
			   Bx_out,
			   Bx_out_fftx,
			   (l + 1), m, n,
			   1,
			   ishift_i,
			   0,
			   ishift_j,
			   0,
			   ishift_k);
  compute_warp_inverse_dft(plan_inverse,
			   l,
			   m,
			   n,
			   (temp1 + 4 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 4 * (l / 2 + 1) * m * n),
			   temp0,
			   By_out,
			   By_out_fftx,
			   l, (m + 1), n,
			   0,
			   ishift_i,
			   1,
			   ishift_j,
			   0,
			   ishift_k);
  compute_warp_inverse_dft(plan_inverse,
			   l,
			   m,
			   n,
			   (temp1 + 5 * (l / 2 + 1) * m * n),
                           (temp1_fftx + 5 * (l / 2 + 1) * m * n),
			   temp0,
			   Bz_out,
			   Bz_out_fftx,
			   l, m, (n + 1),
			   0,
			   ishift_i,
			   0,
			   ishift_j,
			   1,
			   ishift_k);
}

void reportDifferences(const char* name, double* cufft_out, double* fftx_out, int ll, int mm, int nn) {
  double diff=0, cufft_max=0, fftx_max=0;
  int imax=-1, jmax=-1, kmax=-1;
  
  for(int k = 0; k < nn; k++)
    for(int j = 0; j < mm; j++)
      for(int i = 0; i < ll; i++) { 
	int idx = i + j * ll + k * (ll * mm);
	double c = cufft_out[idx];
	double f = fftx_out[idx];
	double d = std::abs(c-f);
	  

	if(d> diff)
          {
            diff=d;
	  //std::cout << i << ", " << j << ", " << k << std::endl;
            imax=i;
            jmax=j;
            kmax=k;
          }
	
	if(std::abs(c)>cufft_max) cufft_max=c;
	if(std::abs(f)>fftx_max) fftx_max=f;   
      }

  std::cout<<"max norm diff for "<<name<<" is "<<diff<<" at ["<<imax<<","<<jmax<<","<<kmax<<"]  cufft_max="<<cufft_max<<"  fftx_max="<<fftx_max<<"\n";
}

float execute_code(int l,
		   int m,
		   int n,
		   double **fields_in,
		   cufftDoubleComplex **shift_in,
		   double **contractions,
		   double **fields_out, double **fields_out_fftx,
		   cufftDoubleComplex **shift_out) {
  // the fields
  double *dev_Ex_in, *dev_Ey_in, *dev_Ez_in, *dev_Bx_in, *dev_By_in, *dev_Bz_in, *dev_Jx, *dev_Jy, *dev_Jz, *dev_rho_0, *dev_rho_1;
  double *dev_Ex_out, *dev_Ey_out, *dev_Ez_out, *dev_Bx_out, *dev_By_out, *dev_Bz_out;
  double *dev_Ex_out_fftx, *dev_Ey_out_fftx, *dev_Ez_out_fftx, *dev_Bx_out_fftx, *dev_By_out_fftx, *dev_Bz_out_fftx;

  // the shifts
  cufftDoubleComplex *dev_fshift_i, *dev_fshift_j, *dev_fshift_k, *dev_ishift_i, *dev_ishift_j, *dev_ishift_k;

  // the temporaries
  double *dev_temp0;
  cufftDoubleComplex *dev_temp1, *dev_temp1_fftx;

  // the contraction arrays;
  double *dev_modified_ki_arr, *dev_modified_kj_arr, *dev_modified_kk_arr;
  double *dev_C_arr, *dev_S_arr, *dev_X1_arr, *dev_X2_arr, *dev_X3_arr;

  int cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    exit(-1);
  }

  // device memory allocation
  // allocate Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho_0, rho_1
  cudaStatus = cudaMalloc((void**)&dev_Ex_in, l * (m + 1) * (n + 1) * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ex_in cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Ey_in, (l + 1) * m * (n + 1) * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ey_in cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Ez_in, (l + 1) * (m + 1) * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ey_in cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Bx_in, (l + 1) * m * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Bx_in cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_By_in, l * (m + 1) * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_By_in cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Bz_in, l * m * (n + 1) * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_By_in cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Jx, l * (m + 1) * (n + 1) * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Jx cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Jy, (l + 1) * m * (n + 1) * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Jy cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Jz, (l + 1) * (m + 1) * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Jz cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_rho_0, l * m * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_rho_0 cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_rho_1, l * m * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_rho_1 cudaMalloc failed!");
    exit(-1);
  }

  // allocate Ex, Ey, Ez, Bx, By, Bz
  cudaStatus = cudaMalloc((void**)&dev_Ex_out, l * (m + 1) * (n + 1) * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ex_out cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Ey_out, (l + 1) * m * (n + 1) * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ey_out cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Ez_out, (l + 1) * (m + 1) * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ey_out cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Bx_out, (l + 1) * m * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Bx_out cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_By_out, l * (m + 1) * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_By_out cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Bz_out, l * m * (n + 1) * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_By_out cudaMalloc failed!");
    exit(-1);
  }

    // allocate Ex, Ey, Ez, Bx, By, Bz for the FFTX version of the algorithm.
  
  cudaStatus = cudaMalloc((void**)&dev_Ex_out_fftx, l * (m + 1) * (n + 1) * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ex_out cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Ey_out_fftx, (l + 1) * m * (n + 1) * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ey_out cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Ez_out_fftx, (l + 1) * (m + 1) * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ey_out cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Bx_out_fftx, (l + 1) * m * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Bx_out cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_By_out_fftx, l * (m + 1) * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_By_out cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_Bz_out_fftx, l * m * (n + 1) * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_By_out cudaMalloc failed!");
    exit(-1);
  }
  
  // allocate the shifts
  cudaStatus = cudaMalloc((void**)&dev_fshift_i, (l / 2 + 1) * sizeof(cufftDoubleComplex));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_fshift_i cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_fshift_j, m * sizeof(cufftDoubleComplex));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_fshift_j cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_fshift_k, n * sizeof(cufftDoubleComplex));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_fshift_n cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_ishift_i, (l / 2 + 1) * sizeof(cufftDoubleComplex));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_ishift_i cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_ishift_j, m * sizeof(cufftDoubleComplex));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_ishift_j cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_ishift_k, n * sizeof(cufftDoubleComplex));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_ishift_n cudaMalloc failed!");
    exit(-1);
  }
  
  // allocate temporary arrays
  cudaStatus = cudaMalloc((void**)&dev_temp0, l * m * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_temp0 cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_temp1, 11 * (l / 2 + 1) * m * n * sizeof(cufftDoubleComplex));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_temp1 cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_temp1_fftx, 11 * (l / 2 + 1) * m * n * sizeof(cufftDoubleComplex));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_temp1 cudaMalloc failed!");
    exit(-1);
  }
  
  // allocate the contraction arrays
  cudaStatus = cudaMalloc((void**)&dev_modified_ki_arr, (l / 2 + 1) * m * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_modified_ki_arr cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_modified_kj_arr, (l / 2 + 1) * m * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_modified_kj_arr cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_modified_kk_arr, (l / 2 + 1) * m * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_modified_kk_arr cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_C_arr, (l / 2 + 1) * m * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_C_arr cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_S_arr, (l / 2 + 1) * m * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_S_arr cudaMalloc failed!");
    exit(-1);
  }
  
  cudaStatus = cudaMalloc((void**)&dev_X1_arr, (l / 2 + 1) * m * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_X1_arr cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_X2_arr, (l / 2 + 1) * m * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_X2_arr cudaMalloc failed!");
    exit(-1);
  }

  cudaStatus = cudaMalloc((void**)&dev_X3_arr, (l / 2 + 1) * m * n * sizeof(double));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_X3_arr cudaMalloc failed!");
    exit(-1);
  }

  // copy the data to the device
  // copy Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho_0, rho_1
  cudaStatus = cudaMemcpy((void*) dev_Ex_in, fields_in[0], l * (m + 1) * (n + 1) * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ex_in cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_Ey_in, fields_in[1], (l + 1) * m * (n + 1) * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ey_in cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_Ez_in, fields_in[2], (l + 1) * (m + 1) * n * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ez_in cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_Bx_in, fields_in[3], (l + 1) * m * n * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Bx_in cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_By_in, fields_in[4], l * (m + 1) * n * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_By_in cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_Bz_in, fields_in[5], l * m * (n + 1) * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Bz_in cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_Jx, fields_in[6], l * (m + 1) * (n + 1) * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Jx cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_Jy, fields_in[7], (l + 1) * m * (n + 1) * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Jy cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_Jz, fields_in[8], (l + 1) * (m + 1) * n * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Jz cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_rho_0, fields_in[9], l * m * n * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_rho_0 cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_rho_1, fields_in[10], l * m * n * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_rho_1 cudaMemcpy failed!");
    exit(-1);
  }

  // copy the shift arrays
  cudaStatus = cudaMemcpy((void*) dev_fshift_i, shift_in[0], (l / 2 + 1) * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_fshift_i cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_fshift_j, shift_in[1], m * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_fshift_j cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_fshift_k, shift_in[2], n * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_fshift_k cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_ishift_i, shift_out[0], (l / 2 + 1) * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_ishift_i cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_ishift_j, shift_out[1], m * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_ishift_j cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_ishift_k, shift_out[2], n * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_ishift_k cudaMemcpy failed!");
    exit(-1);
  }

  // copy the contraction arrays
  cudaStatus = cudaMemcpy((void*) dev_modified_ki_arr, contractions[0], (l / 2 + 1) * m * n * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_modified_ki_arr cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_modified_kj_arr, contractions[1], (l / 2 + 1) * m * n * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_modified_kj_arr cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_modified_kk_arr, contractions[2], (l / 2 + 1) * m * n * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_modified_kk_arr cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_C_arr, contractions[3], (l / 2 + 1) * m * n * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_C_arr cudaMemcpy failed!");
    exit(-1);
  }
  
  cudaStatus = cudaMemcpy((void*) dev_S_arr, contractions[4], (l / 2 + 1) * m * n * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_S_arr cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_X1_arr, contractions[5], (l / 2 + 1) * m * n * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_X1_arr cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_X2_arr, contractions[6], (l / 2 + 1) * m * n * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_X2_arr cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) dev_X3_arr, contractions[7], (l / 2 + 1) * m * n * sizeof(double), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_X3_arr cudaMemcpy failed!");
    exit(-1);
  }

  // create Fourier plans
  cufftHandle plan_forward, plan_inverse;
  cufftPlan3d(&plan_forward, l, m, n, CUFFT_D2Z);
  cufftPlan3d(&plan_inverse, l, m, n, CUFFT_Z2D);
  if(l==m==n==80) {
    DFT_80::init();
    IDFT_80::init();
  }
  
  if(l==m==n==100) {
    DFT_100::init();
    IDFT_100::init();
  }
  
  if(l==m==224 && n==100) {
    DFT_224_224_100::init();
    IDFT_224_224_100::init();
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  compute_spectral_solve(l,
			 m,
			 n,
			 plan_forward,
			 plan_inverse,
			 dev_Ex_in,
			 dev_Ey_in,
			 dev_Ez_in,
			 dev_Bx_in,
			 dev_By_in,
			 dev_Bz_in,
			 dev_Jx,
			 dev_Jy,
			 dev_Jz,
			 dev_rho_0,
			 dev_rho_1,
			 dev_fshift_i,
			 dev_fshift_j,
			 dev_fshift_k,
			 dev_temp0,
			 dev_temp1,
                         dev_temp1_fftx,
			 dev_modified_ki_arr,
			 dev_modified_kj_arr,
			 dev_modified_kk_arr,
			 dev_C_arr,
			 dev_S_arr,
			 dev_X1_arr,
			 dev_X2_arr,
			 dev_X3_arr,
			 dev_Ex_out,
			 dev_Ey_out,
			 dev_Ez_out,
			 dev_Bx_out,
			 dev_By_out,
			 dev_Bz_out,
                         dev_Ex_out_fftx,
			 dev_Ey_out_fftx,
			 dev_Ez_out_fftx,
			 dev_Bx_out_fftx,
			 dev_By_out_fftx,
			 dev_Bz_out_fftx,
			 dev_ishift_i,
			 dev_ishift_j,
			 dev_ishift_k);
  cudaEventRecord(stop);
  
  // synchronize the device
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    exit(-1);
  }

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  // copy the data from the device
  cudaStatus = cudaMemcpy((void*) fields_out[0], dev_Ex_out, l * (m + 1) * (n + 1) * sizeof(double), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ex_out cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) fields_out[1], dev_Ey_out, (l + 1) * m * (n + 1) * sizeof(double), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ey_out cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) fields_out[2], dev_Ez_out, (l + 1) * (m + 1) * n * sizeof(double), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ez_out cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) fields_out[3], dev_Bx_out, (l + 1) * m * n * sizeof(double), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Bx_out cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) fields_out[4], dev_By_out, l * (m + 1) * n * sizeof(double), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_By_out cudaMemcpy failed!");
    exit(-1);
  }
  

  cudaStatus = cudaMemcpy((void*) fields_out[5], dev_Bz_out, l * m * (n + 1) * sizeof(double), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Bz_out cudaMemcpy failed!");
    exit(-1);
  }


    cudaStatus = cudaMemcpy((void*) fields_out_fftx[0], dev_Ex_out_fftx, l * (m + 1) * (n + 1) * sizeof(double), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ex_out_fftx cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) fields_out_fftx[1], dev_Ey_out_fftx, (l + 1) * m * (n + 1) * sizeof(double), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ey_out_fftx cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) fields_out_fftx[2], dev_Ez_out_fftx, (l + 1) * (m + 1) * n * sizeof(double), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Ez_out_fftx cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) fields_out_fftx[3], dev_Bx_out_fftx, (l + 1) * m * n * sizeof(double), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Bx_out_fftx cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) fields_out_fftx[4], dev_By_out_fftx, l * (m + 1) * n * sizeof(double), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_By_out_fftx cudaMemcpy failed!");
    exit(-1);
  }

  cudaStatus = cudaMemcpy((void*) fields_out_fftx[5], dev_Bz_out_fftx, l * m * (n + 1) * sizeof(double), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "dev_Bz_out_fftx cudaMemcpy failed!");
    exit(-1);
  }

  // compare answers between cufft and fftx
  reportDifferences("Ex",fields_out[0],fields_out_fftx[0],l, m+1, n+1);
  reportDifferences("Ey",fields_out[1],fields_out_fftx[1],l+1, m, n+1);
  reportDifferences("Ez",fields_out[2],fields_out_fftx[2],l+1, m+1, n);
  reportDifferences("Bx",fields_out[3],fields_out_fftx[3],l+1, m, n);
  reportDifferences("By",fields_out[4],fields_out_fftx[4],l, m+1, n);
  reportDifferences("Bz",fields_out[5],fields_out_fftx[5],l, m, n+1);
 
  // destroy cufftPlans
  cufftDestroy(plan_forward);
  cufftDestroy(plan_inverse);
  
  if(l==m==n==80) {
    DFT_80::destroy();
    IDFT_80::destroy();
  }
  
  if(l==m==n==100) {
    DFT_100::destroy();
    IDFT_100::destroy();
  }
  
  if(l==m==224 && n==100) {
    DFT_224_224_100::destroy();
    IDFT_224_224_100::destroy();
  }
  
  // deallocate device memory
  cudaFree(dev_Ex_in);
  cudaFree(dev_Ey_in);
  cudaFree(dev_Ez_in);
  cudaFree(dev_Bx_in);
  cudaFree(dev_By_in);
  cudaFree(dev_Bz_in);
  cudaFree(dev_Jx);
  cudaFree(dev_Jy);
  cudaFree(dev_Jz);
  cudaFree(dev_rho_0);
  cudaFree(dev_rho_1);

  cudaFree(dev_fshift_i);
  cudaFree(dev_fshift_j);
  cudaFree(dev_fshift_k);
  cudaFree(dev_ishift_i);
  cudaFree(dev_ishift_j);
  cudaFree(dev_ishift_k);

  cudaFree(dev_temp0);
  cudaFree(dev_temp1);
  cudaFree(dev_temp1_fftx);
  
  cudaFree(dev_modified_ki_arr);
  cudaFree(dev_modified_kj_arr);
  cudaFree(dev_modified_kk_arr);
  
  cudaFree(dev_S_arr);
  cudaFree(dev_C_arr);

  cudaFree(dev_X1_arr);
  cudaFree(dev_X2_arr);
  cudaFree(dev_X3_arr);
  
  cudaFree(dev_Ex_out);
  cudaFree(dev_Ey_out);
  cudaFree(dev_Ez_out);
  cudaFree(dev_Bx_out);
  cudaFree(dev_By_out);
  cudaFree(dev_Bz_out);

  cudaFree(dev_Ex_out_fftx);
  cudaFree(dev_Ey_out_fftx);
  cudaFree(dev_Ez_out_fftx);
  cudaFree(dev_Bx_out_fftx);
  cudaFree(dev_By_out_fftx);
  cudaFree(dev_Bz_out_fftx);

  return milliseconds;
}

int main(int argc, char **argv) {
  int l = atoi(argv[1]);
  int m = atoi(argv[2]);
  int n = atoi(argv[3]);

  // input and output fields
  double **fields_in, **fields_out;
  double **fields_out_fftx;

  // shifting arrays
  cufftDoubleComplex **shift_in, **shift_out;

  // contraction arrays
  double **contractions;

  // allocate memory for the fields
  fields_in = (double**) malloc(11 * sizeof(double*));
  fields_out = (double**) malloc(6 * sizeof(double*));
  fields_out_fftx = (double**) malloc(6 * sizeof(double*));

  fields_in[0] = (double*) malloc(l * (m + 1) * (n + 1) * sizeof(double));
  fields_in[1] = (double*) malloc((l + 1) * m * (n + 1) * sizeof(double));
  fields_in[2] = (double*) malloc((l + 1) * (m + 1) * n * sizeof(double));

  fields_in[3] = (double*) malloc((l + 1) * m * n * sizeof(double));
  fields_in[4] = (double*) malloc(l * (m + 1) * n * sizeof(double));
  fields_in[5] = (double*) malloc(l * m * (n + 1) * sizeof(double));

  fields_in[6] = (double*) malloc(l * (m + 1) * (n + 1) * sizeof(double));
  fields_in[7] = (double*) malloc((l + 1) * m * (n + 1) * sizeof(double));
  fields_in[8] = (double*) malloc((l + 1) * (m + 1) * n * sizeof(double));

  fields_in[9] = (double*) malloc(l * m * n * sizeof(double));
  fields_in[10] = (double*) malloc(l * m * n * sizeof(double));

  fields_out[0] = (double*) malloc(l * (m + 1) * (n + 1) * sizeof(double));
  fields_out[1] = (double*) malloc((l + 1) * m * (n + 1) * sizeof(double));
  fields_out[2] = (double*) malloc((l + 1) * (m + 1) * n * sizeof(double));

  fields_out[3] = (double*) malloc((l + 1) * m * n * sizeof(double));
  fields_out[4] = (double*) malloc(l * (m + 1) * n * sizeof(double));
  fields_out[5] = (double*) malloc(l * m * (n + 1) * sizeof(double));
  
  fields_out_fftx[0] = (double*) malloc(l * (m + 1) * (n + 1) * sizeof(double));
  fields_out_fftx[1] = (double*) malloc((l + 1) * m * (n + 1) * sizeof(double));
  fields_out_fftx[2] = (double*) malloc((l + 1) * (m + 1) * n * sizeof(double));

  fields_out_fftx[3] = (double*) malloc((l + 1) * m * n * sizeof(double));
  fields_out_fftx[4] = (double*) malloc(l * (m + 1) * n * sizeof(double));
  fields_out_fftx[5] = (double*) malloc(l * m * (n + 1) * sizeof(double));

  for(int i = 0; i < l * (m + 1) * (n + 1); ++i) {
    fields_in[0][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_in[6][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_out[0][i] = 0.0;
    fields_out_fftx[0][i] = 0.0;
  }

  for(int i = 0; i < (l + 1) * m * (n + 1); ++i) {
    fields_in[1][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_in[7][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_out[1][i] = 0.0;
    fields_out_fftx[1][i] = 0.0;
  }

  for(int i = 0; i < (l + 1) * (m + 1) * n; ++i) {
    fields_in[2][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_in[8][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_out[2][i] = 0.0;
    fields_out_fftx[2][i] = 0.0;
  }

  for(int i = 0; i < (l + 1) * m * n; ++i) {
    fields_in[3][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_out[3][i] = 0.0;
    fields_out_fftx[3][i] = 0.0;
  }

  for(int i = 0; i < l * (m + 1) * n; ++i) {
    fields_in[4][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_out[4][i] = 0.0;
    fields_out_fftx[4][i] = 0.0;
  }

  for(int i = 0; i < l * m * (n + 1); ++i) {
    fields_in[5][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_out[5][i] = 0.0;
    fields_out_fftx[5][i] = 0.0;
  }

  for(int i = 0; i < l * m * n; ++i) {
    fields_in[9][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_in[10][i] = rand() / ((double) (INT_MAX * 1.0));
  }

  // allocate the shifting arrays
  shift_in = (cufftDoubleComplex**) malloc(3 * sizeof(cufftDoubleComplex*));
  shift_out = (cufftDoubleComplex**) malloc(3 * sizeof(cufftDoubleComplex*));

  shift_in[0] = (cufftDoubleComplex*) malloc((l / 2 + 1) * sizeof(cufftDoubleComplex));
  shift_in[1] = (cufftDoubleComplex*) malloc(m * sizeof(cufftDoubleComplex));
  shift_in[2] = (cufftDoubleComplex*) malloc(n * sizeof(cufftDoubleComplex));

  shift_out[0] = (cufftDoubleComplex*) malloc((l / 2 + 1) * sizeof(cufftDoubleComplex));
  shift_out[1] = (cufftDoubleComplex*) malloc(m * sizeof(cufftDoubleComplex));
  shift_out[2] = (cufftDoubleComplex*) malloc(n * sizeof(cufftDoubleComplex));

  for(int i = 0; i < (l / 2 + 1); ++i) {
    double re = rand() / ((double) (INT_MAX * 1.0));
    double im = rand() / ((double) (INT_MAX * 1.0));

    shift_in[0][i].x = re;
    shift_in[0][i].y = im;
    shift_out[0][i].x = re;
    shift_out[0][i].x = -1.0 * im;
  }

  for(int i = 0; i < m; ++i) {
    double re = rand() / ((double) (INT_MAX * 1.0));
    double im = rand() / ((double) (INT_MAX * 1.0));

    shift_in[1][i].x = re;
    shift_in[1][i].y = im;
    shift_out[1][i].x = re;
    shift_out[1][i].x = -1.0 * im;
  }

  for(int i = 0; i < n; ++i) {
    double re = rand() / ((double) (INT_MAX * 1.0));
    double im = rand() / ((double) (INT_MAX * 1.0));

    shift_in[2][i].x = re;
    shift_in[2][i].y = im;
    shift_out[2][i].x = re;
    shift_out[2][i].x = -1.0 * im;
  }

  // allocate the contraction arrays
  contractions = (double**) malloc(8 * sizeof(double*));

  contractions[0] = (double*) malloc((l / 2 + 1) * m * n * sizeof(double));
  contractions[1] = (double*) malloc((l / 2 + 1) * m * n * sizeof(double));
  contractions[2] = (double*) malloc((l / 2 + 1) * m * n * sizeof(double));
  contractions[3] = (double*) malloc((l / 2 + 1) * m * n * sizeof(double));
  contractions[4] = (double*) malloc((l / 2 + 1) * m * n * sizeof(double));
  contractions[5] = (double*) malloc((l / 2 + 1) * m * n * sizeof(double));
  contractions[6] = (double*) malloc((l / 2 + 1) * m * n * sizeof(double));
  contractions[7] = (double*) malloc((l / 2 + 1) * m * n * sizeof(double));

  for(int i = 0; i < (l / 2 + 1) * m * n; ++i) {
    contractions[0][i] = rand() / ((double) (INT_MAX * 1.0));
    contractions[1][i] = rand() / ((double) (INT_MAX * 1.0));
    contractions[2][i] = rand() / ((double) (INT_MAX * 1.0));
    contractions[3][i] = rand() / ((double) (INT_MAX * 1.0));
    contractions[4][i] = rand() / ((double) (INT_MAX * 1.0));
    contractions[5][i] = rand() / ((double) (INT_MAX * 1.0));
    contractions[6][i] = rand() / ((double) (INT_MAX * 1.0));
    contractions[7][i] = rand() / ((double) (INT_MAX * 1.0));
  }

  // gpu execution
  float milliseconds = execute_code(l,
				    m,
				    n,
				    fields_in,
				    shift_in,
				    contractions,
				    fields_out,
				    fields_out_fftx,
				    shift_out);

  printf("Execution time:\t%f\n", milliseconds);
  
  free(fields_in[0]);
  free(fields_in[1]);
  free(fields_in[2]);
  free(fields_in[3]);
  free(fields_in[4]);
  free(fields_in[5]);
  free(fields_in[6]);
  free(fields_in[7]);
  free(fields_in[8]);
  free(fields_in[9]);
  free(fields_in[10]);

  free(fields_out[0]);
  free(fields_out[1]);
  free(fields_out[2]);
  free(fields_out[3]);
  free(fields_out[4]);
  free(fields_out[5]);
  free(fields_out_fftx[0]);
  free(fields_out_fftx[1]);
  free(fields_out_fftx[2]);
  free(fields_out_fftx[3]);
  free(fields_out_fftx[4]);
  free(fields_out_fftx[5]);

  free(fields_in);
  free(fields_out);
  free(fields_out_fftx);

  free(shift_in[0]);
  free(shift_in[1]);
  free(shift_in[2]);

  free(shift_out[0]);
  free(shift_out[1]);
  free(shift_out[2]);

  free(shift_in);
  free(shift_out);

  free(contractions[0]);
  free(contractions[1]);
  free(contractions[2]);
  free(contractions[3]);
  free(contractions[4]);
  free(contractions[5]);
  free(contractions[6]);
  free(contractions[7]);

  free(contractions);
  
  return 0;
}