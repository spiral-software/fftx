#ifndef rconv_PRECOMPILE_H
#define rconv_PRECOMPILE_H

#include "fftx3.hpp"
#include "transformer.fftx.precompile.hpp"

/*
 Real convolution class for precompiled transforms

 Construction should fail if a transform of given size is not available.
*/

namespace fftx {
  
  template <int DIM>
  class rconv : public transformer<DIM>
  {
  public:

    rconv(const point_t<DIM>& a_size) : transformer<DIM>(a_size)
    {
      // m_cubesize = getSize();
      m_sizeHalf = transformer<DIM>::m_size;
      m_sizeHalf[0] = transformer<DIM>::m_size[0]/2 + 1;
      // look up this transform size in the database.
      // I would prefer if this was a constexpr kind of thing where we fail at compile time
      // The type cubesize_t is fixed to 3D.
      transformTuple_t* tupl = fftx_rconv_Tuple ( transformer<DIM>::m_cubesize );
      // I get segfault if I do this in base class, so repeat here.
      // transformer<DIM>::setInit(tupl);
      if (tupl == NULL)
        {
          printf("This size is not in the library.\n");
        }
      else
        {
          printf("This size is in the library. Initializing.\n");
          transformer<DIM>::init_spiral = *tupl->initfp;
          transform_spiral = *tupl->runfp;
          transformer<DIM>::destroy_spiral = *tupl->destroyfp;
        }

      transformer<DIM>::callInit(tupl);
      /*
      if (transformer<DIM>::init_spiral != nullptr)
        {
          transformer<DIM>::init_spiral();
#ifdef __CUDACC__

          cudaEvent_t& mystart = transformer<DIM>::m_start;
          cudaEvent_t& mystop = transformer<DIM>::m_stop;

          cudaEventCreate(&mystart);
          cudaEventCreate(&mystop);
#endif
        }
      */
    }

    ~rconv()
    {
      // in base class
      // if (destroy_spiral != nullptr) destroy_spiral();
    }

    /*
    double m_CPU_milliseconds = 0.;
    float  m_GPU_milliseconds = 0.;
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
    */
 

    inline fftx::handle_t transform(array_t<DIM, double>& a_src,
                                    array_t<DIM, double>& a_dst,
                                    array_t<DIM, double>& a_sym)
    { // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world

      // Check that a_src and a_dst are of size m_size,
      // and that a_sym is of size m_sizeHalf.

      box_t<DIM> srcDomain = a_src.m_domain;
      box_t<DIM> dstDomain = a_dst.m_domain;
      box_t<DIM> symDomain = a_sym.m_domain;

      point_t<DIM> srcExtents = srcDomain.extents();
      point_t<DIM> dstExtents = dstDomain.extents();
      point_t<DIM> symExtents = symDomain.extents();

      bool srcSame = (srcExtents == transformer<DIM>::m_size);
      bool dstSame = (dstExtents == transformer<DIM>::m_size);
      bool symSame = (symExtents == m_sizeHalf);
      if (!srcSame)
        {
          printf("rconv::transform wrong input array size\n");
        }
      if (!dstSame)
        {
          printf("rconv::transform wrong output array size\n");
        }
      if (!symSame)
        {
          printf("rconv::transform wrong symbol array size\n");
        }
      if (srcSame && dstSame && symSame)
        {
          double* inputLocal = (double*) (a_src.m_data.local());
          double* outputLocal = (double*) (a_dst.m_data.local());
          double* symLocal = (double*) (a_sym.m_data.local());

          transformer<DIM>::kernelStart();
          std::chrono::high_resolution_clock::time_point t1 =
            std::chrono::high_resolution_clock::now();
          transform_spiral(outputLocal, inputLocal, symLocal);
          transformer<DIM>::kernelStop();
          std::chrono::high_resolution_clock::time_point t2 =
            std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
          transformer<DIM>::m_CPU_milliseconds = time_span.count()*1000;
        }

      // dummy return handle for now
      fftx::handle_t rtn;
      return rtn;
    }
 
  protected:
    point_t<DIM> m_sizeHalf;
    
  private:
    // void (*init_spiral)() = nullptr;
    void (*transform_spiral)(double*, double*, double*) = nullptr;
    // void (*destroy_spiral)() = nullptr;
  };
};

#endif  
