#ifndef transformer_PRECOMPILE_H
#define transformer_PRECOMPILE_H

#include "fftx3.hpp"

/*
 Real 3D convolution class for precompiled transforms

 null contruction should fail if a transform of size [NX,NY,NZ] is not available
*/

namespace fftx {
  
  template <int DIM>
  class transformer
  {
  public:
    transformer(const point_t<DIM>& a_size)
    {
      m_size = a_size;
      m_cubesize = getSize(a_size);
      printf("Defining transformer on [%d,%d,%d]\n",
             m_cubesize.dimx, m_cubesize.dimy, m_cubesize.dimz);
      // transformTuple_t* tupl = fftx_transformer_Tuple ( m_cubesize );
      // setInit(tuple);
    }
    
    ~transformer()
    {
      if (destroy_spiral != nullptr) destroy_spiral();
    }

#ifdef __CUDACC__
    cudaEvent_t m_start, m_stop;
    void kernelStart() {cudaEventRecord(m_start);}
    void kernelStop()
    {
      cudaEventRecord(m_stop);
      cudaDeviceSynchronize();
      cudaEventSynchronize(m_stop);
      cudaEventElapsedTime(&m_GPU_milliseconds, m_start, m_stop);
    }
#else
    void kernelStart(){ }
    void kernelStop(){ }
#endif

    double CPU_milliseconds() { return m_CPU_milliseconds; }
    double GPU_milliseconds() { return m_GPU_milliseconds; }
 
  protected:

    cubesize_t m_cubesize;

    point_t<DIM> m_size;

    double m_CPU_milliseconds = 0.;
    float  m_GPU_milliseconds = 0.;

    cubesize_t getSize(const point_t<DIM> a_size)
    {
      cubesize_t returnsize;
      returnsize.dimx = a_size[0];
      returnsize.dimy = a_size[1];
      returnsize.dimz = a_size[2];
      return returnsize;
    }

    void setInit(transformTuple_t* a_tupl)
    {
      // look up this transform size in the database.
      // I would prefer if this was a constexpr kind of thing where we fail at compile time
      // a_tupl = fftx_transformer_Tuple ( m_cubesize );
      if (a_tupl == NULL)
        {
          printf("transformer: this size is not in the library.\n");
        }
      else
        {
          printf("transformer: this size is in the library. Initializing.\n");
          // Still need to set transform_spiral in the derived class.
          init_spiral = *a_tupl->initfp;
          destroy_spiral = *a_tupl->destroyfp;
        }

      if (init_spiral != nullptr)
        {
          printf("calling init_spiral()\n");
          init_spiral();
          printf("called init_spiral()\n");
#ifdef __CUDACC__
          cudaEventCreate(&m_start);
          cudaEventCreate(&m_stop);
#endif
        }
    }

    // private:
    void (*init_spiral)() = nullptr;
    // void (*transform_spiral)(double*, double*, double*) = nullptr;
    void (*destroy_spiral)() = nullptr;
  };
};

#endif  
