//
//  Copyright (c) 2018-2025, Carnegie Mellon University
//  All rights reserved.
//
//  See LICENSE file for full information.
//

#include <mpi.h>
#include <complex>
#include <iostream>
#include <cstring> // memcpy
#include <stdlib.h>     /* srand, rand */
#include "fftxdevice_macros.h"

#include "fftx_mpi.hpp"

// using namespace std;

#define FFTX_DEBUG 0
#define FFTX_DEBUG_OUTPUT 1
#define FFTX_PRETTY_PRINT 1
#define FFTX_TOLERANCE 1e-8

inline size_t ceil_div(size_t a, size_t b)
{
  return (a + b - 1) / b;
}

// When running with p ranks, length of LAST dimension must be divisible by p.

using cx = std::complex<double>;

// Set arr to zero at n positions.
void zeroDoubles(double* arr, size_t n)
{
  for (size_t ind = 0; ind < n; ind++)
    {
      arr[ind] = 0.;
    }
  return;
}

// Set arr to random numbers at n positions.
void randDoubles(double* arr, size_t n)
{
  for (size_t ind = 0; ind < n; ind++)
    {
      arr[ind] = 2.0 * rand() / RAND_MAX - 1.0;
    }
}

// Set dst to src at n positions.
void copyDoubles(double* dst, double* src, size_t n)
{
  for (size_t ind = 0; ind < n; ind++)
    {
      dst[ind] = src[ind];
    }
}
                 
// return fftx::box_t starting at 0 and having given extents
template<int DIM>
inline fftx::box_t<DIM> box0size(int* x)
{
  fftx::point_t<DIM> pt;
  for (int d = 0; d < DIM; d++)
    {
      pt[d] = x[d] - 1;
    }
  return fftx::box_t<DIM>(fftx::point_t<DIM>::Zero(), pt);
}
                                 
// In the C2R (!is_complex && !is_forward) case only,
// the input must be Hermitian-symmetric
// in order for the output to be real.

// A Hermitian-symmetric vector of length n, with k in range 0:n-1,
// must be of the form testfun(k, n), where:

// cx testfun(size_t k, size_t n)
// {
//   if (k == 0)
//     return cx(AN ARBITRARY REAL NUMBER, 0.0);
//   else if (2*k == n)
//     return cx(ANOTHER ARBITRARY REAL NUMBER, 0.0);
//   else if (2*k < n)
//     return cx(AN ARBITRARY FUNCTION OF k, ANOTHER ARBITRARY FUNCTION OF k);
//   else if (2*k > n)
//     return conj(testfun(n - k, n)); // RECURSIVE CALL TO THE FUNCTION
//   else
//     return cx(0., 0.);
// }

// We will take a product of these functions, one for each of the 3 dimensions.
// That will give us a function with 3D Hermitian symmetry.

cx testfun0hermitian(size_t k, size_t n)
{
  if (k == 0)
    return cx(1.5, 0.0);
  else if (2*k == n)
    return cx(2.2, 0.0);
  else if (2*k < n)
    return cx(sin(k * 1.), log((1 + k)*1.));
  else if (2*k > n)
    return conj(testfun0hermitian(n - k, n));
  else
    return cx(0., 0.);
}

cx testfun1hermitian(size_t k, size_t n)
{
  if (k == 0)
    return cx(-0.9, 0.0);
  else if (2*k == n)
    return cx(1.3, 0.0);
  else if (2*k < n)
    return cx(cos(k * 1.), tan(1. + (k*1.)/(n*2.)));
  else if (2*k > n)
    return conj(testfun1hermitian(n - k, n));
  else
    return cx(0., 0.);
}

cx testfun2hermitian(size_t k, size_t n)
{
  if (k == 0)
    return cx(1.1, 0.0);
  else if (2*k == n)
    return cx(-0.7, 0.0);
  else if (2*k < n)
    return cx(exp(-abs(k * 1.)), atan(1. + (k*1.)/(n*1.)));
  else if (2*k > n)
    return conj(testfun2hermitian(n - k, n));
  else
    return cx(0., 0.);
}

cx testfunhermitian(size_t k0, size_t k1, size_t k2,
                    size_t n0, size_t n1, size_t n2)
{
  return testfun0hermitian(k0, n0) * testfun1hermitian(k1, n1) * testfun2hermitian(k2, n2);
}
  

double inputRealSymmetric(int i, int j, int l, int K, int N, int M)
{
  double center = ((K + 1)*1.)/2.;
  double y = sin(i*1. - center) * ((j*j)*3. + 5. * cos(l*1.));
  return y;
}

int main(int argc, char* argv[])
{
  char *prog = argv[0];
  int status = 0;
  
  MPI_Init(&argc, &argv);

  int rank;
  int p;

  // root MPI rank
  size_t root = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (argc != 10)
    {
      if (rank == root)
        {
          fftx::OutStream() << "usage: " << argv[0]
                            << " <M> <N> <K> <batch> <embedded> <forward> <complex> <trials> <check>"
                            << std::endl;
        }
      MPI_Finalize();
      exit(-1);
    }
  // X dim is size M,
  // Y dim is size N,
  // Z dim is size K.
  // Sizes are given in real-space. i.e. M is size of input for R2C, output for C2R.
  // This allows the same plan to be used for both forward and inverse transforms.
  size_t M = atoi(argv[1]);
  size_t N = atoi(argv[2]);
  size_t K = atoi(argv[3]);

  size_t batch     = atoi(argv[4]);
  bool is_embedded = 0 < atoi(argv[5]);
  bool is_forward  = 0 < atoi(argv[6]);
  bool is_complex  = 0 < atoi(argv[7]);
  int trials       = atoi(argv[8]);
  int check        = atoi(argv[9]);

  if (trials <= 0)
    {
      if (rank == root)
        {
          fftx::OutStream() << "Error: trials must be greater than 0" << std::endl;
        }
      MPI_Finalize();
      exit(-1);
    }


  // check == 1: basic test (first element),
  // check == 2: full test (local comparison of all elements)

  // (slowest to fastest)
  // R2C input is [K,       N, M]         doubles, block distributed Z.
  // C2R input is [N, M/2 + 1, K] complex doubles, block distributed X.
  // C2C input is [K,       N, M] complex doubles, block distributed Z.
  // TODO: check this
  // C2C inv   is [N,       M, K] complex doubles, block distributed Z.
  bool R2C = !is_complex &&  is_forward;
  bool C2R = !is_complex && !is_forward;
  bool C2C =  is_complex;
  size_t expansion = is_embedded ? 2 : 1;

  size_t C_in = (R2C) ? 1 : 2; // 1 if input real, 2 if input complex.
  size_t C_out = (C2R) ? 1 : 2; // 1 if output real, 2 if output complex.

  /*
    Set the input:  host_in on host, and dev_in on device.

    If forward transform, set input to random numbers.

    If inverse transform, 
    in case C2C, set to random numbers;
    in case C2R, set to testfunhermitian function.
   */
  size_t Mfull_in = C2R ? (M*expansion/2 + 1) : (M*expansion);
  size_t Mfull_out = R2C ? (M*expansion/2 + 1) : (M*expansion);
  size_t K_in_local, K_in_global, K_in, N_in, M_in_global, M_in_local, M_in;
  size_t K_out_global, K_out_local, K_out, N_out, M_out;
  size_t M_out_global = Mfull_out; // == R2C ? (M*expansion/2 + 1) : (M*expansion);
  size_t M_out_local = ceil_div(M_out_global, p);

  size_t in_pts, in_doubles, in_bytes, out_pts;
  double *host_in; // input of FFTX transform

  fftx::box_t<3> in_domain, out_domain;
  fftx::box_t<3> in_subdomain, out_subdomain;
  if (is_forward)
    {
      /*
        [(pz), ceil(K/pz), N, M]
        [(pz), ceil(K/pz), N, M*expansion] (embed)
        [(pz), Mfull_out, ceil(K/pz), N] (stage 1, permute) (Mfull_out depends on C2C or R2C, and embedded)
        [(pz), px, ceil(Mfull_out/px), ceil(K/pz), N] (reshape)
        [(px), pz, ceil(Mfull_out/px), ceil(K/pz), N] (a2a)
        [(px), ceil(Mfull_out/px), pz, ceil(K/pz), N] (permute)
        [(px), ceil(Mfull_out/px), pz*ceil(K/pz), N] (reshape)
        [(px), ceil(Mfull_out/px), pz*ceil(K/pz), N*expansion] (embed)
        [(px), N*expansion, ceil(Mfull_out/px), pz*ceil(K/pz)] (stage 2, permute)
        [(px), N*expansion, ceil(Mfull_out/px), pz*ceil(K/pz)*expansion] (embed) --> TODO: embed could go into a smaller space?
        [(px), N*expansion, ceil(Mfull_out/px), pz*ceil(K/pz)*expansion] (stage 3)
        [(px), N*expansion, ceil(Mfull_out/px), K*expansion] (stage 3, if embedded in smaller space)
      */
      // TODO: what about when K % p != 0?

      // input as [(pz), ceil(K/pz), N, M*expansion] (embed)
      K_in_global = K;
      K_in_local = ceil_div(K_in_global, p);
      N_in = N;
      M_in = Mfull_in; // == M*expansion;

      int in_extents[] = {(int) K_in_local, (int) N_in, (int) M_in};
      in_domain = box0size<3>(in_extents);
      in_pts = in_domain.size();
      in_doubles = in_pts * batch*C_in;
      in_bytes = in_doubles * sizeof(double);
      host_in = (double *) malloc(in_bytes);

      // Now set host_in.
      // subdomain of points to fill with random data; set to zero elsewhere
      in_subdomain = in_domain;
      // need rank*K_in_local + kk < K_in_global
      // so kk < K_in_global - rank * K_in_local
      // Makes a difference only when K not divisible by p, and rank == p-1.
      in_subdomain.hi[0] = min(K_in_local, K_in_global - rank*K_in_local) - 1;
      if (is_embedded)
        {
          in_subdomain.lo[2] = M/2;
          in_subdomain.hi[2] = M/2 + M-1;
        }
      for (size_t ind_pt = 0; ind_pt < in_pts; ind_pt++)
        {
          fftx::point_t<3> pt = pointFromPositionBox(ind_pt, in_domain);
          // index within host_in of first double at this point
          size_t ind_double = ind_pt * batch*C_in;
          if (isInBox(pt, in_subdomain))
            {
              randDoubles(host_in + ind_double, batch*C_in);
            }
          else
            {
              zeroDoubles(host_in + ind_double, batch*C_in);
            }
        }

      // global output spatial lengths, in order:
      // output as [(px), N*expansion, ceil(Mfull/px), pz*ceil(K/pz)*expansion] (stage 3)
      N_out = N*expansion;
      K_out_global = K;
      K_out = K * expansion;
 
      int out_extents[] = {(int) N_out, (int) M_out_local, (int) K_out};
      out_domain = box0size<3>(out_extents);
      out_pts = out_domain.size();
    }
  else
    { // inverse
      /* Assumes inverse embedded keeps full doubled-embedded space.
         [(px), N*expansion, ceil(Mfull_out/px), pz*ceil(K/pz)*expansion] (output of fwd)
         [(px), N*expansion, ceil(Mfull_out/px), K*expansion] (output of fwd, if embedded puts into smaller space)
         [(px), N*expansion, ceil(Mfull_out/px), K] (currently written) --> is actually this.
         NOTE: FOR NOW, assuming K is divisble by number of ranks.
         Looks like library code may be needed to changed to support otherwise.
         [(px), N*expansion, ceil(Mfull_out/px), K*expansion] (stage 1)
         [(px), ceil(Mfull_out/px), K*expansion, N*expansion] (stage 2, permute)
         [(px), ceil(Mfull_out/px), pz, ceil(K*expansion/pz), N*expansion] (reshape)
         [(px), pz, ceil(Mfull_out/px), ceil(K*expansion/pz), N*expansion] (permute)
         [(pz), px, ceil(Mfull_out/px), ceil(K*expansion/pz), N*expansion] (a2a)
         [(pz), px*ceil(Mfull_out/px), ceil(K*expansion/pz), N*expansion] (reshape)
         [(pz), ceil(K*expansion/pz), N*expansion, px*ceil(Mfull_out/px)] (stage3, permute)
         [(pz), ceil(K*expansion/pz), N*expansion, Mfull_out] (stage3, permute, embed?)
      */
      
      //Inverse transform has input ``XPar, [Y, Xlocal, Z, b]`` (lengths p, N, M/p, K, batch),
      // output ``Zpar, [Zlocal, Y, X, b]`` (lengths p, K/p, N, M, batch).
      // global input spatial lengths, in order:
      // N expanded, M expanded (divided, truncated if R2C), K expanded.

      // input as  [(px), N*expansion, ceil(Mfull_out/px), K*expansion] (output of fwd, if embedded puts into smaller space)
      N_in = N*expansion;
      M_in_global = Mfull_in; // == C2R ? (M*expansion/2 + 1) : (M*expansion);
      M_in_local = ceil_div(M_in_global, p);
      K_in = K*expansion;

      int in_extents[] = {(int) N_in, (int) M_in_local, (int) K_in};
      in_domain = box0size<3>(in_extents);
      in_pts = in_domain.size();
      in_doubles = in_pts * batch*C_in;
      in_bytes = in_doubles * sizeof(double);
      host_in = (double *) malloc(in_bytes);

      // Now set host_in.
      // subdomain of points to fill with nonzero data; set to zero elsewhere
      in_subdomain = in_domain;
      // need rank*M_in_local + mm < M_in_global
      // so mm < M_in_global - rank * M_in_local
      // Makes a difference only when M not divisible by p, and rank == p-1.
      in_subdomain.hi[1] = max(M_in_local, M_in_global - rank*M_in_local) - 1;
      for (size_t ind_pt = 0; ind_pt < in_pts; ind_pt++)
        {
          fftx::point_t<3> pt = pointFromPositionBox(ind_pt, in_domain);
          // index within host_in of first double at this point
          size_t ind_double = ind_pt * batch*C_in;
          if (isInBox(pt, in_subdomain))
            {
              if (C2R)
                { // need to be have a complex-valued input with
                  // the proper symmetry so that it transforms
                  // to a real-valued output
                  int nn = pt[0];
                  int mm = pt[1];
                  int kk = pt[2];
                  size_t mm_global = rank * M_in_local + mm;
                  for (size_t b = 0; b < batch; b++)
                    {
                      cx v = testfunhermitian(nn, mm_global, kk,
                                              N*expansion, M*expansion, K*expansion);
                      host_in[ind_double] = real(v);
                      ind_double++;
                      host_in[ind_double] = imag(v);
                      ind_double++;
                    }
                }
              else // this is C2C inverse
                {
                  randDoubles(host_in + ind_double, batch*C_in);
                }
            }
          else
            {
              zeroDoubles(host_in + ind_double, batch*C_in);
            }
        }

      // global output spatial lengths, in order:
      // output as [(pz), ceil(K*expansion/pz), N*expansion, px*ceil(Mfull_out/px)] (stage3, permute)
      K_out_global = K*expansion;
      K_out_local = ceil_div(K_out_global, p);
      N_out = N*expansion;
      M_out = p * M_out_local;
      
      int out_extents[] = {(int) K_out_local, (int) N_out, (int) M_out};
      out_domain = box0size<3>(out_extents);
      out_pts = out_domain.size();

      // Makes a difference only when K_out_global not divisible by p, and rank == p-1.
      out_subdomain = out_domain;
      out_subdomain.hi[0] = min(K_out_local, K_out_global - rank*K_out_local) - 1;
    } // end forward/inverse

  size_t out_doubles = out_pts * batch*C_out;
  size_t out_bytes = out_doubles * sizeof(double);
  double *host_out = (double *) malloc(out_bytes);
  double *dev_in, *dev_out;
  FFTX_DEVICE_MALLOC(&dev_in , in_bytes);
  FFTX_DEVICE_MALLOC(&dev_out, out_bytes);

  FFTX_DEVICE_MEM_COPY(dev_in, host_in, in_bytes,
                       FFTX_MEM_COPY_HOST_TO_DEVICE);

  if (FFTX_PRETTY_PRINT)
    {
      if (rank == root)
        {
          fftx::OutStream() << "Problem size: " << M << " x " << N << " x " << K << std::endl;
          fftx::OutStream() << "Batch size  : " << batch << std::endl;
          fftx::OutStream() << "Embedded    : " << (is_embedded ? "Yes" : "No") << std::endl;
          fftx::OutStream() << "Direction   : " << (is_forward ? "Forward" : "Inverse") << std::endl;
          fftx::OutStream() << "Complex     : " << (is_complex ? "Yes" : "No") << std::endl;
          fftx::OutStream() << "MPI Ranks   : " << p << std::endl;
          fftx::OutStream() << "Times       : " << std::endl;
        }
    }

  /*
    Get plan for the distributed FFTX transform.
  */
  fftx_plan plan = fftx_plan_distributed_1d(MPI_COMM_WORLD, p, M, N, K, batch,
                                            is_embedded, is_complex);
  MPI_Barrier(MPI_COMM_WORLD);

  /*
    Run trials of the distributed FFTX transform, and report timings.
  */
  FFTX_DEVICE_EVENT_T custart, custop;
  FFTX_DEVICE_EVENT_CREATE ( &custart );
  FFTX_DEVICE_EVENT_CREATE ( &custop );
  for (int t = 1; t <= trials; t++)
    {
      FFTX_DEVICE_EVENT_RECORD ( custart );

      fftx_execute_1d(plan,
                      (double*)dev_out,
                      (double*)dev_in,
                      (is_forward ? FFTX_DEVICE_FFT_FORWARD :
                       FFTX_DEVICE_FFT_INVERSE));

      FFTX_DEVICE_EVENT_RECORD ( custop );
      FFTX_DEVICE_EVENT_SYNCHRONIZE ( custop );
      float millisec;
      FFTX_DEVICE_EVENT_ELAPSED_TIME ( &millisec, custart, custop );
      float max_time;
      MPI_Reduce(&millisec, &max_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

      if (rank == root)
        {
          if (FFTX_PRETTY_PRINT)
            {
              fftx::OutStream() << "\tTrial " << t << ": "
                                << max_time << " millisec" << std::endl;
            }
          else
            {
              fftx::OutStream()
                << M << "," << N << "," << K << ","
                << batch << ","
                << p << ","
                << (is_embedded ? "embedded" : "") << ","
                << (is_forward ? "fwd" : "inv") << ","
                << (is_complex ? "complex" : "real") << ","
                << (check == 1 ? "first_elem" : "local") << ","
                << max_time;
              if (t < trials-1)
                { // only check last iter, will write its own end line.
                  fftx::OutStream() << std::endl;
                }
            }
        }
    }

  MPI_Barrier(MPI_COMM_WORLD);

  if (check == 1)
    { // simple check that first element of output array matches sum over input array, or vice versa.
      bool correct = true;
      FFTX_DEVICE_MEM_COPY(host_out, dev_out,
                           out_bytes,
                           FFTX_MEM_COPY_DEVICE_TO_HOST);

      if (is_forward)
        { // Forward C2C or R2C:  we have full M*N*K input array.
          // Allocate space for sum of each array in batch input.
          // C2C:  each input array has M*H*K complex elements.
          // R2C:  each input array has M*N*K real elements.
          // input as  [(pz), ceil(K/pz), N, M*expansion] (embed)
          // so K_in_local (note K_in_global), N_in, M_in.
          
          double *sum_in_real = new double[batch];
          zeroDoubles(sum_in_real, batch);
          double *sum_in_imag = new double[batch];
          zeroDoubles(sum_in_imag, batch);

          for (size_t ind_pt = 0; ind_pt < in_pts; ind_pt++)
            {
              fftx::point_t<3> pt = pointFromPositionBox(ind_pt, in_domain);
              if (isInBox(pt, in_subdomain))
                { // Input was set to zero outside in_subdomain.
                  // index within host_in of first double at this point
                  size_t ind_double = ind_pt * batch*C_in;
                  for (size_t b = 0; b < batch; b++)
                    {
                      sum_in_real[b] += host_in[ind_double];
                      ind_double++;
                      if (C_in == 2) // C2C forward (not R2C)
                        {
                          sum_in_imag[b] += host_in[ind_double];
                          ind_double++;
                        }
                    }
                }
            }

          for (size_t b = 0; b < batch; b++)
            {
              MPI_Allreduce(MPI_IN_PLACE, sum_in_real + b, 1,
                            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
              MPI_Allreduce(MPI_IN_PLACE, sum_in_imag + b, 1,
                            MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }
          
          if (rank == 0)
            { // First element of output is on rank 0.
              for (size_t b = 0; b < batch; b++)
                {
                  double first_out_real = host_out[b*C_out];
                  if  ( abs(first_out_real - sum_in_real[b]) >
                        FFTX_TOLERANCE )
                    {
                      correct = false;
                    }
                  if (C_out == 2) // C2C forward (not R2C)
                    {
                      double first_out_imag = host_out[b*C_out + 1];
                      if  ( abs(first_out_imag - sum_in_imag[b]) >
                            FFTX_TOLERANCE )
                        {
                          correct = false;
                        }
                    }
              }
          }
        
        delete[] sum_in_real;
        delete[] sum_in_imag;
      }
    else
      { // Inverse C2C or C2R:  we have full M*N*K output array.
        // In the C2R case, we cannot sum up the whole M*N*K input array,
        // because the input array is truncated, so instead we sum up
        // the output array and compare that sum (scaled by array size)
        // with the first element of the input array.
        // output as [(pz), ceil(K*expansion/pz), N*expansion, px*ceil(Mfull_out/px)] (stage3, permute)
        // so K_out_local (note K_out_global), N_out, M_out.
      
        // C2C:  each output array has M*N*K complex elements.
        // C2R:  each output array has M*N*K real elements.

        // inv input
        // [(px), N*expansion, ceil(Mfull_out/px), pz*ceil(K/pz)*expansion] (output of fwd)
        // [(px), N*expansion, ceil(Mfull_out/px), K*expansion] (output of fwd, if embedded puts into smaller space)
        // inv output
        // [(pz), ceil(K*expansion/pz), N*expansion, Mfull_out] (stage3, permute, embed?)

        double *sum_out_real = new double[batch];
        zeroDoubles(sum_out_real, batch);
        double *sum_out_imag = new double[batch];
        zeroDoubles(sum_out_imag, batch);

        for (size_t ind_pt = 0; ind_pt < out_pts; ind_pt++)
          {
            fftx::point_t<3> pt = pointFromPositionBox(ind_pt, out_domain);
            if (isInBox(pt, out_subdomain))
              { // Can ignore output outside out_subdomain.
                // index within host_out of first double at this point
                size_t ind_double = ind_pt * batch*C_out;
                for (size_t b = 0; b < batch; b++)
                  {
                    sum_out_real[b] += host_out[ind_double];
                    ind_double++;
                    if (C_out == 2)
                      {
                        sum_out_imag[b] += host_out[ind_double];
                        ind_double++;
                      }
                  }
              }
          }
        for (size_t b = 0; b < batch; b++)
          {
            MPI_Allreduce(MPI_IN_PLACE, sum_out_real + b, 1,
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, sum_out_imag + b, 1,
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          }

        double scaling_output =
          (double) 1.0 / ((double) (M*expansion * N*expansion * K*expansion));

        if (rank == 0)
          { // First element of input is on rank 0.
            for (size_t b = 0; b < batch; b++)
              {
                double first_in_real = host_in[b*C_in];
                if ( abs(first_in_real - sum_out_real[b]*scaling_output) >
                     FFTX_TOLERANCE )
                  {
                    correct = false;
                  }
                if (C_out == 2)
                  {
                    double first_in_imag = (C_in == 2) ?
                      host_in[b*C_in + 1] : 0.;
                    if ( abs(first_in_imag - sum_out_imag[b]*scaling_output) >
                         FFTX_TOLERANCE )
                      {
                        correct = false;
                      }
                  }
              }
            
          } // end root rank

        delete[] sum_out_real;
        delete[] sum_out_imag;
      }

      if (rank == root)
        {
          if (FFTX_PRETTY_PRINT)
            {
              fftx::OutStream() << "Correct     : "
                                << (correct ? "Yes" : "No")
                                << std::endl;
            }
          else
            {
              if (correct)
                {
                  fftx::OutStream() << ",1";
                }
              else
                {
                  fftx::OutStream() << ",0";
                }
            }
        }
    }
  else if (check == 2)
    {  // local 3D comparison, check all elements.
      if (M > 64 || N > 64 || K > 64 || p > 4)
        {
          // too big
          if (rank == root)
            {
              fftx::OutStream() << ",X" << std::endl;
            }
          goto end;
        }

      bool correct = true;

      double *host_global_in_gath;
      double *host_global_FFTX_out;

      // Number of points/doubles/bytes in gathered input data.
      size_t global_in_gath_pts = p * in_pts;
      size_t global_in_gath_doubles = global_in_gath_pts * batch*C_in;
      size_t global_in_gath_bytes = global_in_gath_doubles * sizeof(double);

      // Number of points/doubles/bytes in gathered FFTX output data.
      size_t global_out_gath_pts = p * out_pts;
      size_t global_out_gath_doubles = global_out_gath_pts * batch*C_out;
      size_t global_out_gath_bytes = global_out_gath_doubles * sizeof(double);

      // Gather input, save in host_global_in_gath.
      // Gather FFTX output, save in host_global_FFTX_out.
      if (rank == root)
        {
          host_global_in_gath = (double *) malloc(global_in_gath_bytes);

          host_global_FFTX_out = (double *) malloc(global_out_gath_bytes);
        }

      MPI_Barrier(MPI_COMM_WORLD);
      {
        int error = MPI_Gather(host_in, in_doubles, MPI_DOUBLE,
                               host_global_in_gath, in_doubles, MPI_DOUBLE,
                               root, MPI_COMM_WORLD);
        if (error != 0)
          {
            fftx::ErrStream() << "Error: MPI_Gather on input returned error code "
                              << error << std::endl;
            status++;
          }
      }
      MPI_Barrier(MPI_COMM_WORLD);

      // Copy local FFTX output dev_out to host as host_out,
      // and then gather host_out on root to host_global_FFTX_out.
      FFTX_DEVICE_MEM_COPY(host_out, dev_out,
                           out_bytes,
                           FFTX_MEM_COPY_DEVICE_TO_HOST);
      MPI_Barrier(MPI_COMM_WORLD);

      {
        int error = MPI_Gather(host_out, out_doubles, MPI_DOUBLE,
                               host_global_FFTX_out, out_doubles, MPI_DOUBLE,
                               root, MPI_COMM_WORLD);
        if (error != 0)
          {
            fftx::ErrStream() << "Error: MPI_Gather on FFTX output returned error code "
                              << error << std::endl;
            status++;
          }
      }
      MPI_Barrier(MPI_COMM_WORLD);

      if (rank == root)
        {
          // local extents of the input that has been gathered
          // in host_global_in_gath
          fftx::point_t<3> in_extents = in_domain.extents();
          int global_in_gath_extents[] = {p, in_extents[0], in_extents[1], in_extents[2]};
          fftx::box_t<4> global_in_gath_domain = box0size<4>(global_in_gath_extents);
      
          // If forward:
          // in_pts = K_in_local * N_in * M_in;
          // == ceil_div(K, p) * N * M*expansion
          // == K/p * N * 2*M if embedded
          // out_pts = N_out * M_out_local * K_out;
          // == N*expansion * ceil_div(Mfull_out, p) * p*ceil_div(K, p)*expansion
          // == 2*N * 2*M/p * 2*K if embedded and complex

          // If inverse:
          // in_pts = N_in * M_in_local * K_in;
          // == N*expansion * ceil_div(Mfull_in, p) * K*expansion
          // == 2*N * 2*M/p, 2*K if embedded and complex
          // out_pts = K_out_local * N_out * M_out;
          // == ceil_div(K*expansion, p) * N*expansion * p*ceil_div(M*expansion, p)
          // == 2*K/p * 2*N * 2*M if embedded

          // Allocate global input array for vendor FFT.
          // (In unembedded forward case, can reuse host_global_in_gath.)
          int global_in_extents[] = {(int) (K*expansion), (int) (N*expansion), (int) Mfull_in};
          fftx::box_t<3> global_in_domain = box0size<3>(global_in_extents);
          size_t global_in_pts = global_in_domain.size();
          size_t global_in_doubles = global_in_pts * batch*C_in;
          size_t global_in_bytes = global_in_doubles * sizeof(double);
          double *host_global_in;
          // Set to true if host_global_in is allocated.
          bool host_global_in_allocated = false;
          if (is_embedded || !is_forward)
            {
              host_global_in = (double *) malloc(global_in_bytes);
              host_global_in_allocated = true;
            }

          // Allocate global output array for vendor FFT.
          int global_out_extents[] = {(int) (K*expansion), (int) (N*expansion), (int) (M_out_global)};
          fftx::box_t<3> global_out_domain = box0size<3>(global_out_extents);
          size_t global_out_pts = global_out_domain.size();
          size_t global_out_doubles = global_out_pts * batch*C_out;
          size_t global_out_bytes = global_out_doubles * sizeof(double);
          double *host_global_vendor_out;
          host_global_vendor_out = (double *) malloc(global_out_bytes);

          if (is_forward)
            {
              if (is_embedded)
                {
                  // Embed host_global_in_gath into host_global_in,
                  // padding the rest with zeros.

                  // put gathered data into embedded tensor.
                  // layout is [pz, ceil(K/pz), N, M*e]
                  // gathered layout is [Z, Y, 2X, b], pad Y and Z dims.
                  
                  // size_t npts = K/2 * N*2 * Mfull_in;

                  // We will set host_global_in on global_in_domain
                  // to a transpose of host_global_in_gath.
                  // global_in_gath_extents[] == {p, K_in_local, N_in, M_in};

                  // Initialize host_global_in to zero before setting
                  // to host_global_in_gath on global_in_gath_domain.
                  zeroDoubles(host_global_in, global_in_doubles);
                  for (size_t ind_pt = 0; ind_pt < global_in_gath_domain.size(); ind_pt++)
                    {
                      fftx::point_t<4> pt =
                        pointFromPositionBox(ind_pt, global_in_gath_domain);
                      int prank = pt[0];
                      int kk = pt[1];
                      int nn = pt[2];
                      int mm = pt[3];
                      int kk_global = prank * K_in_local + kk;
                      if (kk_global < K)
                        {
			  int kk_mod = kk_global + K/2;
                          int nn_mod = nn + N/2;
                          fftx::point_t<3> pt_mod( { {kk_mod, nn_mod, mm} } );
                          size_t ind_mod_pt =
                            positionInBox(pt_mod, global_in_domain);
                          // index within host_global_in_gath of
                          // first double at this point
                          size_t ind_double = ind_pt * batch*C_in;
                          // index within host_global_in of
                          // first double at this point
                          size_t ind_mod_double = ind_mod_pt * batch*C_in;
                          copyDoubles(host_global_in + ind_mod_double,
                                      host_global_in_gath + ind_double,
                                      batch*C_in);
                        }
                    }
                }
              else
                {
                  // forward, not embedded
                  // Can use host_global_in_gath; don't need to allocate
                  // memory for host_global_in, so just set the pointer.
                  host_global_in = host_global_in_gath;
                }
            }
          else
            { // is inverse (could be embedded or not)
              // We will set host_global_in on global_in_domain
              // to a transpose of host_global_in_gath.
              // global_in_gath_extents[] == {p, N_in, M_in_local, K_in};

              for (size_t ind_pt = 0; ind_pt < global_in_gath_domain.size(); ind_pt++)
                {
                  fftx::point_t<4> pt =
                    pointFromPositionBox(ind_pt, global_in_gath_domain);
                  int prank = pt[0];
                  int nn = pt[1];
                  int mm = pt[2];
                  int kk = pt[3];
                  int mm_global = prank * M_in_local + mm;
                  if (mm_global < M_in_global)
                    {
                      fftx::point_t<3> pt_mod( { {kk, nn, mm_global} } );
                      size_t ind_mod_pt =
                        positionInBox(pt_mod, global_in_domain);
                      // index within host_global_in_gath of
                      // first double at this point
                      size_t ind_double = ind_pt * batch*C_in;
                      // index within host_global_in of
                      // first double at this point
                      size_t ind_mod_double = ind_mod_pt * batch*C_in;
                      copyDoubles(host_global_in + ind_mod_double,
                                  host_global_in_gath + ind_double,
                                  batch*C_in);
                    }
                }
              
            } // end if-else fwd/inv

          double *dev_global_in, *dev_global_out;
          FFTX_DEVICE_MALLOC(&dev_global_in, global_in_bytes);
          FFTX_DEVICE_MALLOC(&dev_global_out, global_out_bytes);

          FFTX_DEVICE_MEM_COPY(dev_global_in,
                               host_global_in,
                               global_in_bytes,
                               FFTX_MEM_COPY_HOST_TO_DEVICE);

          // Define cuFFT/rocFFT plan.
          FFTX_DEVICE_FFT_HANDLE plan_vendor;
 
          // dimensions from slowest to fastest.
          int stride = batch;
          int dist = 1;
          int dims[] = {(int) (K*expansion), (int) (N*expansion), (int) (M*expansion)};

          FFTX_DEVICE_FFT_TYPE tfmtype;
          if (C2C)
            {
              tfmtype = FFTX_DEVICE_FFT_Z2Z;
            }
          else if (R2C)
            {
              tfmtype = FFTX_DEVICE_FFT_D2Z;
            }
          else if (C2R)
            {
              tfmtype = FFTX_DEVICE_FFT_Z2D;
            }
          else
            {
              fftx::ErrStream() << "Error: unknown plan type." << std::endl;
              status++;
              goto end;
            }
          FFTX_DEVICE_FFT_PLAN_MANY(&plan_vendor, 3, dims,
                                    global_in_extents, stride, dist,
                                    global_out_extents, stride, dist,
                                    tfmtype, batch);
          if (C2C)
            {
              // FFTX_DEVICE_FFT_PLAN3D(&plan_vendor,
              // K*expansion, N*expansion, M*expansion,
              // FFTX_DEVICE_FFT_Z2Z);
              FFTX_DEVICE_FFT_EXECZ2Z(plan_vendor,
                                      (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) dev_global_in,
                                      (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) dev_global_out,
                                      is_forward ? FFTX_DEVICE_FFT_FORWARD : FFTX_DEVICE_FFT_INVERSE
                                      );
            }
          else if (R2C)
            {
              // FFTX_DEVICE_FFT_PLAN3D(&plan_vendor,
              // K*expansion, N*expansion, M*expansion,
              // FFTX_DEVICE_FFT_D2Z);
              FFTX_DEVICE_FFT_EXECD2Z(plan_vendor,
                                      (FFTX_DEVICE_FFT_DOUBLEREAL *) dev_global_in,
                                      (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) dev_global_out
                                      );
            }
          else if (C2R)
            {
              // FFTX_DEVICE_FFT_PLAN3D(&plan_vendor,
              // K*expansion, N*expansion, M*expansion,
              // FFTX_DEVICE_FFT_Z2D);
              FFTX_DEVICE_FFT_EXECZ2D(plan_vendor,
                                      (FFTX_DEVICE_FFT_DOUBLECOMPLEX *) dev_global_in,
                                      (FFTX_DEVICE_FFT_DOUBLEREAL *) dev_global_out
                                      );
            }
          else
            {
              fftx::ErrStream() << "Error: unknown plan type." << std::endl;
              status++;
              goto end;
            }

          {
            FFTX_DEVICE_ERROR_T device_status = FFTX_DEVICE_SYNCHRONIZE();
            if (device_status != FFTX_DEVICE_SUCCESS)
              {
                fftx::ErrStream() << "FFTX_DEVICE_SYNCHRONIZE returned error code "
                                  << device_status << " after 3DFFT!"
                                  << std::endl;
                status++;
              }
          }

          FFTX_DEVICE_FFT_DESTROY(plan_vendor);

          FFTX_DEVICE_MEM_COPY(host_global_vendor_out,
                               dev_global_out,
                               global_out_bytes,
                               FFTX_MEM_COPY_DEVICE_TO_HOST);

          {
            FFTX_DEVICE_ERROR_T device_status = FFTX_DEVICE_SYNCHRONIZE();
            if (device_status != FFTX_DEVICE_SUCCESS)
              {
                fftx::ErrStream() << "FFTX_DEVICE_SYNCHRONIZE returned error code "
                                  << device_status << " after 3DFFT!"
                                  << std::endl;
                status++;
              }
          }

          // Check host_global_vendor_out against host_global_FFTX_out.
          if (is_forward) // includes R2C
            {
              // M_out_local == ceil_div(M_out_global, p) == ceil_div(Mfull_out, p)
              fftx::point_t<3> out_extents = out_domain.extents();
              int global_FFTX_extents[] = {p, out_extents[0], out_extents[1], out_extents[2]};
              fftx::box_t<4> global_FFTX_domain = box0size<4>(global_FFTX_extents);

              for (size_t ind_pt = 0; ind_pt < global_FFTX_domain.size(); ind_pt++)
                {
                  fftx::point_t<4> pt =
                    pointFromPositionBox(ind_pt, global_FFTX_domain);
                  int prank = pt[0];
                  int nn = pt[1];
                  int mm = pt[2];
                  int kk = pt[3];
                  int mm_global = prank * M_out_local + mm;
                  if ((kk < K*expansion) && (mm_global < M_out_global))
                    {
                      // index within host_global_FFTX_out of
                      // first double at this point
                      size_t ind_double = ind_pt * batch*C_out;
                      fftx::point_t<3> pt_vendor( { {kk, nn, mm_global} } );
                      size_t ind_vendor_pt =
                        positionInBox(pt_vendor, global_out_domain);
                      // index within host_global_vendor_out of
                      // first double at this point
                      size_t ind_vendor_double = ind_vendor_pt * batch*C_out;
                      for (size_t b = 0; b < batch; b++)
                        for (size_t c = 0; c < C_out; c++)
                          {
                            double val_FFTX = host_global_FFTX_out[ind_double];
                            double val_vendor = host_global_vendor_out[ind_vendor_double];
                            double val_diff = abs(val_FFTX - val_vendor);
                            bool same = (val_diff < FFTX_TOLERANCE);
                            if (!same)
                              {
                                correct = false;
                                if (FFTX_DEBUG_OUTPUT)
                                  {
                                    fftx::OutStream() << "batch=" << b << " "
                                                      << kk << " " << nn << " " << mm_global
                                                      << " part=" << c
                                                      << std::scientific << std::setw(12)
                                                      << std::setprecision(4)
                                                      << " vendor=" << val_vendor
                                                      << std::scientific << std::setw(12)
                                                      << " FFTX=" << val_FFTX
                                                      << std::endl;
                                  }
                              }
                            ind_double++;
                            ind_vendor_double++;
                          }
                    }
                }
            }
          else
            { // inverse C2C or C2R

              for (size_t ind_pt = 0; ind_pt < global_out_domain.size(); ind_pt++)
                {
                  fftx::point_t<3>  pt =
                    pointFromPositionBox(ind_pt, global_out_domain);
                  int kk = pt[0];
                  int nn = pt[1];
                  int mm = pt[2];
                  // index within host_global_FFTX_out and
                  // also within host_global_vendor_out of
                  // first double at this point
                  size_t ind_double = ind_pt * batch*C_out;
                  for (size_t b = 0; b < batch; b++)
                    for (size_t c = 0; c < C_out; c++)
                      {
                        double val_FFTX = host_global_FFTX_out[ind_double];
                        double val_vendor = host_global_vendor_out[ind_double];
                        double val_diff = abs(val_FFTX - val_vendor);
                        bool same = (val_diff < FFTX_TOLERANCE);
                        if (!same)
                          {
                            correct = false;
                            if (FFTX_DEBUG_OUTPUT)
                              {
                                fftx::OutStream() << "batch=" << b << " "
                                                  << mm << " " << nn << " " << kk
                                                  << " part=" << c
                                                  << std::scientific << std::setw(12)
                                                  << std::setprecision(4)
                                                  << " vendor=" << val_vendor
                                                  << std::scientific << std::setw(12)
                                                  << " FFTX=" << val_FFTX
                                                  << std::endl;
                              }
                          }
                        ind_double++;
                      }
                }
            }
          
          if (FFTX_PRETTY_PRINT)
            {
              fftx::OutStream() << "Correct     : "
                                << (correct ? "Yes" : "No")
                                << std::endl;
            }
          else
            {
              if (correct)
                {
                  fftx::OutStream() << ",1";
                }
              else
                {
                  fftx::OutStream() << ",0";
                }
            }

          // Free memory.
          if (host_global_in_allocated)
            {
              free(host_global_in);
            }
          free(host_global_in_gath);
          free(host_global_FFTX_out);
          free(host_global_vendor_out);
          FFTX_DEVICE_FREE(dev_global_in);
          FFTX_DEVICE_FREE(dev_global_out);
        } // end root rank
    } // end check == 2
  else
    {
      // no check
      if (rank == root)
        {
          fftx::OutStream() << ",-";
        }
    }

  if (rank == root)
    {
      fftx::OutStream() << std::endl;
    }

 end:
  fftx_plan_destroy(plan);
  FFTX_DEVICE_FREE(dev_in);
  FFTX_DEVICE_FREE(dev_out);

  free(host_in);
  free(host_out);

  MPI_Finalize();

  if (rank == root)
    {
      fftx::OutStream() << prog << ": All done, exiting with status "
                        << status << std::endl;
      std::flush(fftx::OutStream());
    }
  
  return status;
}
