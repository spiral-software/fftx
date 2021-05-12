#include "mddft3d.h"

#define PPCAT_NX(A, B) A ## B
#define PPCAT(A, B) PPCAT_NX(A, B)

#define INIT_FN_NAME PPCAT(init_, __FILEROOT__)
#define DESTROY_FN_NAME PPCAT(destroy_, __FILEROOT__)

#define FFTX_MPI_3D_CUFFT_STAGE2 2
#define FFTX_MPI_3D_CUFFT_STAGE1 1

void INIT_FN_NAME();
void __FILEROOT__(double *Y, double *X);
void DESTROY_FN_NAME();
