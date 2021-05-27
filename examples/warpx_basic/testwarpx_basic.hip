#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <hip/hcc_detail/host_defines.h>
#include <hip/hcc_detail/math_functions.h>
#include <hip/hcc_detail/device_functions.h>
#include <rocfft.h>

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
  int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

  for(int iter = id; iter < (l * m * n); iter += hipBlockDim_x * hipGridDim_x) {
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
			   double2 *io,
			   int do_shift_i,
			   double2 *shift_i,
			   int do_shift_j,
			   double2 *shift_j,
			   int do_shift_k,
			   double2 *shift_k) {
  int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

  double2 Z;
  Z.x = 1.0;
  Z.y = 0.0;
  
  for(int iter = id; iter < (l * m * n); iter += hipBlockDim_x * hipGridDim_x) {
    int i = (iter % l);
    int j = (iter / l) % m;
    int k = (iter / (l * m)) % n;

    double2 v_shift_i = (do_shift_i == 0) ? Z : *(shift_i + i);
    double2 v_shift_j = (do_shift_j == 0) ? Z : *(shift_j + j);
    double2 v_shift_k = (do_shift_k == 0) ? Z : *(shift_k + k);
    
    double2 value = *(io + iter);
    double2 result;

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
				    double2 *io,
				    double *modified_ki_arr,
				    double *modified_kj_arr,
				    double *modified_kk_arr,
				    double *C_arr,
				    double *S_arr,
				    double *X1_arr,
				    double *X2_arr,
				    double *X3_arr) {
  int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

  double c2 = C_SPEED * C_SPEED;
  double inv_ep0 = 1.0 / EP0;
  
  for(int iter = id; iter < (l * m * n); iter += hipBlockDim_x * hipGridDim_x) {
    int i = (iter % l);
    int j = (iter / l) % m;
    int k = (iter / (l * m)) % n;

    // E and B fields
    double2 Ex = *(io + 0 * l * m * n + iter);
    double2 Ey = *(io + 1 * l * m * n + iter);
    double2 Ez = *(io + 2 * l * m * n + iter);
    double2 Bx = *(io + 3 * l * m * n + iter);
    double2 By = *(io + 4 * l * m * n + iter);
    double2 Bz = *(io + 5 * l * m * n + iter);

    // Shortcut for the values of J and rho
    double2 Jx = *(io + 6 * l * m * n + iter);
    double2 Jy = *(io + 7 * l * m * n + iter);
    double2 Jz = *(io + 8 * l * m * n + iter);
    double2 rho_old = *(io + 9 * l * m * n + iter);
    double2 rho_new = *(io + 10 * l * m * n + iter);
    
    // k vector values, and coefficients
    double kx = *(modified_ki_arr + i);
    double ky = *(modified_kj_arr + j);
    double kz = *(modified_kk_arr + k);
    
    double C = *(C_arr + iter);
    double S_ck = *(S_arr + iter);
    double X1 = *(X1_arr + iter);
    double X2 = *(X2_arr + iter);
    double X3 = *(X3_arr + iter);

    double2 ex, ey, ez, bx, by, bz;

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
inline void __attribute__((always_inline)) compute_warp_forward_dft(rocfft_plan plan,
								    int l,
								    int m,
								    int n,
								    double *input,
								    int l_is,
								    int m_is,
								    int n_is,
								    double *temp,
								    double2 *output,
								    int do_shift_i,
								    double2 *shift_i,
								    int do_shift_j,
								    double2 *shift_j,
								    int do_shift_k,
								    double2 *shift_k) {
  dim3 shapeB(THREAD_BLOCKS, 1);
  dim3 shapeT(THREADS, 1);
  
  hipLaunchKernelGGL(pack_data, shapeB, shapeT, 0, 0, l, m, n, input, l_is, m_is, n_is, temp, l, m, n);
  rocfft_execute(plan, (void**) &input, (void**) &temp, NULL);
  hipLaunchKernelGGL(shift_data, shapeB, shapeT, 0, 0, l / 2 + 1, m, n, output, do_shift_i, shift_i, do_shift_j, shift_j, do_shift_k, shift_k);
}

inline void __attribute__((always_inline)) compute_warp_inverse_dft(rocfft_plan plan,
								    int l,
								    int m,
								    int n,
								    double2 *input,
								    double *temp,
								    double *output,
								    int l_os,
								    int m_os,
								    int n_os,
								    int do_shift_i,
								    double2 *shift_i,
								    int do_shift_j,
								    double2 *shift_j,
								    int do_shift_k,
								    double2 *shift_k) {
  dim3 shapeB(THREAD_BLOCKS, 1);
  dim3 shapeT(THREADS, 1);

  hipLaunchKernelGGL(shift_data, shapeB, shapeT, 0, 0, l / 2 + 1, m, n, input, do_shift_i, shift_i, do_shift_j, shift_j, do_shift_k, shift_k);
  rocfft_execute(plan, (void**) &input, (void**) &temp, NULL);
  hipLaunchKernelGGL(pack_data, shapeB, shapeT, 0, 0, l, m, n, temp, l, m, n, output, l_os, m_os, n_os);
}

// compute Spectral Solve
inline void __attribute__((always_inline)) compute_spectral_solve(int l,
								  int m,
								  int n,
								  rocfft_plan plan_forward,
								  rocfft_plan plan_inverse,
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
								  double2 *fshift_i,
								  double2 *fshift_j,
								  double2 *fshift_k,
								  double *temp0,
								  double2 *temp1,
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
								  double2 *ishift_i,
								  double2 *ishift_j,
								  double2 *ishift_k) {
  // Ex, Ey, Ez fields
  compute_warp_forward_dft(plan_forward,
			   l,
			   m,
			   n,
			   Ex_in,
			   l, (m + 1), (n + 1),
			   (temp0 + 0),
			   (temp1 + 0 * (l / 2 + 1) * m * n),
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
			   0,
			   fshift_i,
			   0,
			   fshift_j,
			   0,
			   fshift_k);
  // contraction
  dim3 shapeB(THREAD_BLOCKS, 1);
  dim3 shapeT(THREADS, 1);

  hipLaunchKernelGGL(compute_contraction, shapeB, shapeT, 0, 0, (l / 2 + 1), m, n, temp1, modified_ki_arr, modified_kj_arr, modified_kk_arr, C_arr, S_arr, X1_arr, X2_arr, X3_arr);
  
  // Ex, Ey, Ez fields
  compute_warp_inverse_dft(plan_inverse,
			   l,
			   m,
			   n,
			   (temp1 + 0 * (l / 2 + 1) * m * n),
			   temp0,
			   Ex_out,
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
			   temp0,
			   Ey_out,
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
			   temp0,
			   Ez_out,
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
			   temp0,
			   Bx_out,
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
			   temp0,
			   By_out,
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
			   temp0,
			   Bz_out,
			   l, m, (n + 1),
			   0,
			   ishift_i,
			   0,
			   ishift_j,
			   1,
			   ishift_k);
}

float execute_code(int l,
		   int m,
		   int n,
		   double **fields_in,
		   double2 **shift_in,
		   double **contraction,
		   double **fields_out,
		   double2 **shift_out) {
  // the fields
  double *dev_Ex_in, *dev_Ey_in, *dev_Ez_in, *dev_Bx_in, *dev_By_in, *dev_Bz_in, *dev_Jx, *dev_Jy, *dev_Jz, *dev_rho_0, *dev_rho_1, *dev_Ex_out, *dev_Ey_out, *dev_Ez_out, *dev_Bx_out, *dev_By_out, *dev_Bz_out;

  // the shifts
  double2 *dev_fshift_i, *dev_fshift_j, *dev_fshift_k, *dev_ishift_i, *dev_ishift_j, *dev_ishift_k;

  // the temporaries
  double *dev_temp0;
  double2 *dev_temp1;

  // the contraction arrays;
  double *dev_modified_ki_arr, *dev_modified_kj_arr, *dev_modified_kk_arr;
  double *dev_C_arr, *dev_S_arr, *dev_X1_arr, *dev_X2_arr, *dev_X3_arr;

  // device memory allocation
  // allocate Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho_0, rho_1
  hipMalloc((void**)&dev_Ex_in, l * (m + 1) * (n + 1) * sizeof(double));
  hipMalloc((void**)&dev_Ey_in, (l + 1) * m * (n + 1) * sizeof(double));
  hipMalloc((void**)&dev_Ez_in, (l + 1) * (m + 1) * n * sizeof(double));
  hipMalloc((void**)&dev_Bx_in, (l + 1) * m * n * sizeof(double));
  hipMalloc((void**)&dev_By_in, l * (m + 1) * n * sizeof(double));
  hipMalloc((void**)&dev_Bz_in, l * m * (n + 1) * sizeof(double));
  hipMalloc((void**)&dev_Jx, l * (m + 1) * (n + 1) * sizeof(double));
  hipMalloc((void**)&dev_Jy, (l + 1) * m * (n + 1) * sizeof(double));
  hipMalloc((void**)&dev_Jz, (l + 1) * (m + 1) * n * sizeof(double));
  hipMalloc((void**)&dev_rho_0, l * m * n * sizeof(double));
  hipMalloc((void**)&dev_rho_1, l * m * n * sizeof(double));

  // allocate Ex, Ey, Ez, Bx, By, Bz
  hipMalloc((void**)&dev_Ex_out, l * (m + 1) * (n + 1) * sizeof(double));
  hipMalloc((void**)&dev_Ey_out, (l + 1) * m * (n + 1) * sizeof(double));
  hipMalloc((void**)&dev_Ez_out, (l + 1) * (m + 1) * n * sizeof(double));
  hipMalloc((void**)&dev_Bx_out, (l + 1) * m * n * sizeof(double));
  hipMalloc((void**)&dev_By_out, l * (m + 1) * n * sizeof(double));
  hipMalloc((void**)&dev_Bz_out, l * m * (n + 1) * sizeof(double));

  // allocate the shifts
  hipMalloc((void**)&dev_fshift_i, (l / 2 + 1) * sizeof(double2));
  hipMalloc((void**)&dev_fshift_j, m * sizeof(double2));
  hipMalloc((void**)&dev_fshift_k, n * sizeof(double2));
  hipMalloc((void**)&dev_ishift_i, (l / 2 + 1) * sizeof(double2));
  hipMalloc((void**)&dev_ishift_j, m * sizeof(double2));
  hipMalloc((void**)&dev_ishift_k, n * sizeof(double2));

  // allocate temporary arrays
  hipMalloc((void**)&dev_temp0, l * m * n * sizeof(double));
  hipMalloc((void**)&dev_temp1, 11 * (l / 2 + 1) * m * n * sizeof(double2));
  
  // allocate the contraction arrays
  hipMalloc((void**)&dev_modified_ki_arr, (l / 2 + 1) * m * n * sizeof(double));
  hipMalloc((void**)&dev_modified_kj_arr, (l / 2 + 1) * m * n * sizeof(double));
  hipMalloc((void**)&dev_modified_kk_arr, (l / 2 + 1) * m * n * sizeof(double));
  hipMalloc((void**)&dev_C_arr, (l / 2 + 1) * m * n * sizeof(double));
  hipMalloc((void**)&dev_S_arr, (l / 2 + 1) * m * n * sizeof(double));
  hipMalloc((void**)&dev_X1_arr, (l / 2 + 1) * m * n * sizeof(double));
  hipMalloc((void**)&dev_X2_arr, (l / 2 + 1) * m * n * sizeof(double));
  hipMalloc((void**)&dev_X3_arr, (l / 2 + 1) * m * n * sizeof(double));

  // copy the data to the device
  // copy Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho_0, rho_1
  hipMemcpy((void*) dev_Ex_in, fields_in[0], l * (m + 1) * (n + 1) * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_Ey_in, fields_in[1], (l + 1) * m * (n + 1) * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_Ez_in, fields_in[2], (l + 1) * (m + 1) * n * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_Bx_in, fields_in[3], (l + 1) * m * n * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_By_in, fields_in[4], l * (m + 1) * n * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_Bz_in, fields_in[5], l * m * (n + 1) * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_Jx, fields_in[6], l * (m + 1) * (n + 1) * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_Jy, fields_in[7], (l + 1) * m * (n + 1) * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_Jz, fields_in[8], (l + 1) * (m + 1) * n * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_rho_0, fields_in[9], l * m * n * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_rho_1, fields_in[10], l * m * n * sizeof(double), hipMemcpyHostToDevice);

  // copy the shift arrays
  hipMemcpy((void*) dev_fshift_i, shift_in[0], (l / 2 + 1) * sizeof(double2), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_fshift_j, shift_in[1], m * sizeof(double2), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_fshift_k, shift_in[2], n * sizeof(double2), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_ishift_i, shift_out[0], (l / 2 + 1) * sizeof(double2), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_ishift_j, shift_out[1], m * sizeof(double2), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_ishift_k, shift_out[2], n * sizeof(double2), hipMemcpyHostToDevice);

  // copy the contraction arrays
  hipMemcpy((void*) dev_modified_ki_arr, contraction[0], (l / 2 + 1) * m * n * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_modified_kj_arr, contraction[1], (l / 2 + 1) * m * n * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_modified_kk_arr, contraction[2], (l / 2 + 1) * m * n * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_C_arr, contraction[3], (l / 2 + 1) * m * n * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_S_arr, contraction[4], (l / 2 + 1) * m * n * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_X1_arr, contraction[5], (l / 2 + 1) * m * n * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_X2_arr, contraction[6], (l / 2 + 1) * m * n * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy((void*) dev_X3_arr, contraction[7], (l / 2 + 1) * m * n * sizeof(double), hipMemcpyHostToDevice);


  rocfft_plan plan_forward = nullptr, plan_inverse = nullptr;
  size_t length[3];
  length[0] = l;
  length[1] = m;
  length[2] = n;
  
  rocfft_plan_create(&plan_forward,
		     rocfft_placement_outplace,
		     rocfft_transform_type_real_forward,
		     rocfft_precision_double,
		     3,
		     length,
		     1,
		     nullptr);

  rocfft_plan_create(&plan_inverse,
		     rocfft_placement_outplace,
		     rocfft_transform_type_real_inverse,
		     rocfft_precision_double,
		     3,
		     length,
		     1,
		     nullptr);
  
  // Check if the plan requires a work buffer
  size_t work_buf_forward_size = 0;
  size_t work_buf_inverse_size = 0;
  rocfft_plan_get_work_buffer_size(plan_forward, &work_buf_forward_size);
  rocfft_plan_get_work_buffer_size(plan_inverse, &work_buf_inverse_size);
  
  void* work_buf_forward = nullptr;
  void* work_buf_inverse = nullptr;
  
  rocfft_execution_info info_forward = nullptr;
  rocfft_execution_info info_inverse = nullptr;
  
  if(work_buf_forward_size) {
    rocfft_execution_info_create(&info_forward);
    hipMalloc(&work_buf_forward, work_buf_forward_size);
    rocfft_execution_info_set_work_buffer(info_forward, work_buf_forward, work_buf_forward_size);
  }

  if(work_buf_inverse_size) {
    rocfft_execution_info_create(&info_inverse);
    hipMalloc(&work_buf_inverse, work_buf_inverse_size);
    rocfft_execution_info_set_work_buffer(info_inverse, work_buf_inverse, work_buf_inverse_size);
  }
  
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipEventRecord(start);
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
			 dev_ishift_i,
			 dev_ishift_j,
			 dev_ishift_k);
  hipEventRecord(stop);
  
  // synchronize the device
  hipDeviceSynchronize();

  float milliseconds = 0;
  hipEventElapsedTime(&milliseconds, start, stop);
  
  // copy the data from the device
  hipMemcpy((void*) fields_out[0], dev_Ex_out, l * (m + 1) * (n + 1) * sizeof(double), hipMemcpyDeviceToHost);
  hipMemcpy((void*) fields_out[1], dev_Ey_out, (l + 1) * m * (n + 1) * sizeof(double), hipMemcpyDeviceToHost);
  hipMemcpy((void*) fields_out[2], dev_Ez_out, (l + 1) * (m + 1) * n * sizeof(double), hipMemcpyDeviceToHost);
  hipMemcpy((void*) fields_out[3], dev_Bx_out, (l + 1) * m * n * sizeof(double), hipMemcpyDeviceToHost);
  hipMemcpy((void*) fields_out[4], dev_By_out, l * (m + 1) * n * sizeof(double), hipMemcpyDeviceToHost);
  hipMemcpy((void*) fields_out[5], dev_Bz_out, l * m * (n + 1) * sizeof(double), hipMemcpyDeviceToHost);

  // destroy cufftPlans
  rocfft_plan_destroy(plan_forward);
  rocfft_plan_destroy(plan_inverse);
  
  // deallocate device memory
  hipFree(dev_Ex_in);
  hipFree(dev_Ey_in);
  hipFree(dev_Ez_in);
  hipFree(dev_Bx_in);
  hipFree(dev_By_in);
  hipFree(dev_Bz_in);
  hipFree(dev_Jx);
  hipFree(dev_Jy);
  hipFree(dev_Jz);
  hipFree(dev_rho_0);
  hipFree(dev_rho_1);

  hipFree(dev_fshift_i);
  hipFree(dev_fshift_j);
  hipFree(dev_fshift_k);
  hipFree(dev_ishift_i);
  hipFree(dev_ishift_j);
  hipFree(dev_ishift_k);

  hipFree(dev_temp0);
  hipFree(dev_temp1);

  hipFree(dev_modified_ki_arr);
  hipFree(dev_modified_kj_arr);
  hipFree(dev_modified_kk_arr);
  
  hipFree(dev_S_arr);
  hipFree(dev_C_arr);

  hipFree(dev_X1_arr);
  hipFree(dev_X2_arr);
  hipFree(dev_X3_arr);
  
  hipFree(dev_Ex_out);
  hipFree(dev_Ey_out);
  hipFree(dev_Ez_out);
  hipFree(dev_Bx_out);
  hipFree(dev_By_out);
  hipFree(dev_Bz_out);

  return milliseconds;
}

int main(int argc, char **argv) {
  int l = atoi(argv[1]);
  int m = atoi(argv[2]);
  int n = atoi(argv[3]);

  // input and output fields
  double **fields_in, **fields_out;

  // shifting arrays
  double2 **shift_in, **shift_out;

  // contraction arrays
  double **contractions;

  // allocate memory for the fields
  fields_in = (double**) malloc(11 * sizeof(double*));
  fields_out = (double**) malloc(6 * sizeof(double*));

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

  for(int i = 0; i < l * (m + 1) * (n + 1); ++i) {
    fields_in[0][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_in[6][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_out[0][i] = 0.0;
  }

  for(int i = 0; i < (l + 1) * m * (n + 1); ++i) {
    fields_in[1][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_in[7][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_out[1][i] = 0.0;
  }

  for(int i = 0; i < (l + 1) * (m + 1) * n; ++i) {
    fields_in[2][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_in[8][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_out[2][i] = 0.0;
  }

  for(int i = 0; i < (l + 1) * m * n; ++i) {
    fields_in[3][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_out[3][i] = 0.0;
  }

  for(int i = 0; i < l * (m + 1) * n; ++i) {
    fields_in[4][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_out[4][i] = 0.0;
  }

  for(int i = 0; i < l * m * (n + 1); ++i) {
    fields_in[5][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_out[5][i] = 0.0;
  }

  for(int i = 0; i < l * m * n; ++i) {
    fields_in[9][i] = rand() / ((double) (INT_MAX * 1.0));
    fields_in[10][i] = rand() / ((double) (INT_MAX * 1.0));
  }

  // allocate the shifting arrays
  shift_in = (double2**) malloc(3 * sizeof(double2*));
  shift_out = (double2**) malloc(3 * sizeof(double2*));

  shift_in[0] = (double2*) malloc((l / 2 + 1) * sizeof(double2));
  shift_in[1] = (double2*) malloc(m * sizeof(double2));
  shift_in[2] = (double2*) malloc(n * sizeof(double2));

  shift_out[0] = (double2*) malloc((l / 2 + 1) * sizeof(double2));
  shift_out[1] = (double2*) malloc(m * sizeof(double2));
  shift_out[2] = (double2*) malloc(n * sizeof(double2));

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

  free(fields_in);
  free(fields_out);

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

