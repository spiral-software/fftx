#include <complex>
#include <cstdio>
#include <vector>
#include <mpi.h>
#include <iostream>

#include "device_macros.h"
#include "fftx_gpu.h"
#include "fftx_1d_gpu.h"
#include "fftx_util.h"

#include "fftx_mpi.hpp"
#include "fftx_mpi_default.hpp"
#include "fftx_1d_mpi_default.hpp"

using namespace std;

inline size_t ceil_div(size_t a, size_t b) {
  return (a + b - 1) / b;
}

void init_1d_comms(fftx_plan plan, int pp, int M, int N, int K) {
  // can selectively do this for real fwd or inv.
  size_t M0 = ceil_div(M, pp);
  size_t M1 = pp;

  size_t K0 = ceil_div(K, pp);
  size_t K1 = pp;

  size_t max_size = (((size_t)M0)*((size_t)M1)*((size_t)N)*((size_t)K0)*((size_t)K1)*((size_t)(plan->is_embed ? 8 : 1))/(plan->r)) * plan->b;
#if CUDA_AWARE_MPI
  DEVICE_MALLOC(&(plan->send_buffer), max_size * sizeof(complex<double>));
  DEVICE_MALLOC(&(plan->recv_buffer), max_size * sizeof(complex<double>));
#else
  plan->send_buffer = (complex<double> *) malloc(max_size * sizeof(complex<double>));
  plan->recv_buffer = (complex<double> *) malloc(max_size * sizeof(complex<double>));
#endif
}

void destroy_1d_comms(fftx_plan plan) {
  if (plan) {
#if CUDA_AWARE_MPI
    DEVICE_FREE(plan->send_buffer);
    DEVICE_FREE(plan->recv_buffer);
#else
    free(plan->send_buffer);
    free(plan->recv_buffer);
#endif
  }
}

fftx_plan fftx_plan_distributed_1d(
  int p, int M, int N, int K,
  int batch, bool is_embedded, bool is_complex) {
  fftx_plan plan;
#if FORCE_VENDOR_LIB
  {
#else
  if(is_complex || (!is_complex && batch == 1)) {
    plan = fftx_plan_distributed_1d_spiral(p, M, N, K, batch, is_embedded, is_complex);
    plan->use_fftx = true;
  } else {
#endif
    std::cout << "configuration not supported, using vendor backend" << std::endl;
    plan = fftx_plan_distributed_1d_default(p, M, N, K, batch, is_embedded, is_complex);
    plan->use_fftx = false;
  }
  return plan;
}

void fftx_mpi_rcperm_1d(
  fftx_plan plan, double * Y, double *X, int stage, bool is_embedded
) {
  size_t e = is_embedded ? 2 : 1;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  switch (stage) {
    case FFTX_MPI_EMBED_1:
      {
          // TODO: support different MPI CPU/HOST devices.
          // after first 1D FFT on X dim (slowest to fastest)
          // [X'   , Z/p, Y] <= [Z/p, Y, X]
          // [M/2+1, K/p, N] <= [K/p, N, M]
          // or for C2C
          // [M    , K/p, N] <= [K/p, N, M]

          // TODO: check these sizes.
          // TODO: change this to be acceptable for C2C, not just R2C.
          // size_t buffer_size = (plan->M*e/2+1) * plan->shape[4] * plan->shape[2] * plan->b;
          size_t buffer_size = plan->shape[1] * plan->shape[0] * plan->shape[4] * plan->shape[3] * plan->shape[2] * plan->b;
          size_t sendSize    =                  plan->shape[0] * plan->shape[4] * plan->shape[3] * plan->shape[2] * plan->b;
          size_t recvSize    = sendSize;

          DEVICE_MEM_COPY(
            plan->send_buffer, X,
            buffer_size * sizeof(complex<double>),
            MEM_COPY_DEVICE_TO_HOST
          );
          // TODO: make sure buffer is padded out before send?

          // [pz, X'/px, Z/pz, Y] <= [X', Z/pz, Y]
          MPI_Alltoall(
            plan->send_buffer, sendSize,
            MPI_DOUBLE_COMPLEX,
            plan->recv_buffer, recvSize,
            MPI_DOUBLE_COMPLEX,
            MPI_COMM_WORLD
          );
          //      [ceil(X'/px), pz, Z/pz, Y] <= [pz, ceil(X'/px), Z/pz, Y]
          // i.e. [ceil(X'/px),        Z, Y]
          if (is_embedded) {
            DEVICE_MEM_COPY(
              Y, plan->recv_buffer,
              sizeof(complex<double>) * plan->shape[5] * plan->shape[0] * plan->shape[4] * plan->shape[2] * plan->b,
              MEM_COPY_HOST_TO_DEVICE
            );
            pack_embed(
              plan,
              (complex<double> *) X, (complex<double> *) Y,
              plan->shape[4] * plan->N * plan->b,
              plan->shape[0],
              plan->shape[5],
              false
            );
            embed(
              (complex<double> *) Y, (complex<double> *) X,
              plan->shape[2], // faster
              plan->shape[2], // faster padded
              plan->shape[0] * plan->shape[5] * plan->shape[4], // slower
              plan->b // copy size
            );
          } else {
            DEVICE_MEM_COPY(
              X, plan->recv_buffer,
              sizeof(complex<double>) * plan->shape[5] * plan->shape[0] * plan->shape[4] * plan->shape[2] * plan->b,
              MEM_COPY_HOST_TO_DEVICE
            );
            pack_embed(
              plan,
              (complex<double> *) Y, (complex<double> *) X,
              plan->shape[4] * plan->shape[2] * plan->b,
              plan->shape[0],
              plan->shape[5],
              false
            );
          }
      } // end stage 1 case.
      break;

    case FFTX_MPI_EMBED_2:
      {
        if (is_embedded) {
          // [Y, ceil(X'/px), 2Z] <= [Y, ceil(X'/px), pz, ceil(Z/pz)]
          size_t K0 = ceil_div(plan->K, plan->r);
          size_t K1 = plan->r;
          // embed(
          //   (complex<double> *) Y, (complex<double> *) X,
          //   plan->K * plan->b, // fastest dim, to be doubled and embedded
          //   K1 * K0 * plan->b, // faster padded dim, which K is embedded in.
          //   plan->N*e * plan->shape[0] // slower dim
          // );

          embed(
            (complex<double> *) Y, (complex<double> *) X,
            plan->K, // fastest dim, to be doubled and embedded
            K1 * K0, // faster padded dim, which K is embedded in.
            plan->N*e * plan->shape[0], // slower dim
            plan->b
          );


        } else {
          // NOTE: this case handled outside of this function.
          // swap pointers.
          // void *tmp = (void *) X;
          // X = Y;
          // Y = (complex<double> *) tmp;
        }
      }
      break;

    case FFTX_MPI_EMBED_3:
      {
        // TODO: resume here.

      }
      break;

    case FFTX_MPI_EMBED_4:
      {
        // if inverse,
        // permute such that
        // [X'/px, pz, Z/pz, Y] <= [X'/px,        Z, Y] (reshape)
        // [pz, X'/px, Z/pz, Y] <= [X'/px, pz, Z/pz, Y] (permute)
        if (is_embedded) {
          // [ceil(X'/px),          pz, Z/pz, Y] <= [ceil(X'/px),                 Z, Y] (reshape)
          // [         pz, ceil(X'/px), Z/pz, Y] <= [ceil(X'/px),          pz, Z/pz, Y] (permute)
          size_t K0 = ceil_div(plan->K*e, plan->r);
          size_t K1 = plan->r;
          // arg size isn't supposed to be padded in the dim that it's going to be padded in.
          {
            pack(
              (complex<double> *) Y, (complex<double> *) X,
              plan->shape[0],
              plan->K*e * plan->N*e * plan->b, // istride
              K0 * plan->N*e * plan->b, // ostride
              K1,
              K0 * plan->N*e * plan->b,
              plan->shape[0] * K0 * plan->N*e * plan->b,
              K0 * plan->N*e * plan->b
            );

          }

          // size_t sendSize = plan->shape[0] * plan->shape[4]*e * plan->N*e * plan->b;
          size_t sendSize = plan->shape[0] * K0 * plan->N*e * plan->b;
          size_t recvSize = sendSize;
          DEVICE_MEM_COPY(
            plan->send_buffer, Y,
            sizeof(complex<double>) * K1 * sendSize,
            MEM_COPY_DEVICE_TO_HOST
          );

          // [px, ceil(X'/px), Z/pz, Y] <= [pz, ceil(X'/px), Z/pz, Y] (all2all)
          // [px*ceil(X'/px), Z/pz, Y] <=                      (reshape)
          // kind of automatically strip the excess since X is slowest dim.
          // [X', Z/pz, Y] <=                      (reshape)
          MPI_Alltoall(
            plan->send_buffer,  sendSize,
            MPI_DOUBLE_COMPLEX,
            plan->recv_buffer, recvSize,
            MPI_DOUBLE_COMPLEX,
            MPI_COMM_WORLD
          );

          DEVICE_MEM_COPY(
            Y, plan->recv_buffer,
            sizeof(complex<double>) * plan->shape[1] * recvSize,
            MEM_COPY_HOST_TO_DEVICE
          );
        } else {
          {
            size_t a = plan->shape[4] * plan->shape[3] * plan->shape[2] * plan->b;
            size_t b = plan->shape[5];
            size_t c = plan->shape[0];
            pack(
              (complex<double> *) Y, (complex<double> *) X,
              b,   a, c*a,
              c, b*a,   a,
              a
            );
          }

          size_t sendSize = plan->shape[0] * plan->shape[4] * plan->shape[3] * plan->shape[2] * plan->b;
          size_t recvSize = sendSize;
          DEVICE_MEM_COPY(
            plan->send_buffer, Y,
            sizeof(complex<double>) * plan->shape[5] * sendSize,
            MEM_COPY_DEVICE_TO_HOST
          );

          // [px, X'/px, Z/pz, Y] <= [pz, X'/px, Z/pz, Y] (all2all)
          // [       X', Z/pz, Y] <=                      (reshape)
          MPI_Alltoall(
            plan->send_buffer,  sendSize,
            MPI_DOUBLE_COMPLEX,
            plan->recv_buffer, recvSize,
            MPI_DOUBLE_COMPLEX,
            MPI_COMM_WORLD
          );

          DEVICE_MEM_COPY(
            Y, plan->recv_buffer,
            sizeof(complex<double>) * plan->shape[5] * recvSize,
            MEM_COPY_HOST_TO_DEVICE
          );
        }
      } // end FFTX_MPI_EMBED_4
      break;
    } // end switch/case.
}

void fftx_execute_1d(
  fftx_plan plan,
  double * out_buffer, double * in_buffer,
  int direction
) {
#if FORCE_VENDOR_LIB
  {
#else
  if (plan->use_fftx) {
    fftx_execute_1d_spiral(plan, out_buffer, in_buffer, direction);
  } else {
#endif
    fftx_execute_1d_default(plan, out_buffer, in_buffer, direction);
  }
}
