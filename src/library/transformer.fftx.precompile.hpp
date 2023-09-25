#ifndef transformer_PRECOMPILE_H
#define transformer_PRECOMPILE_H

// This is included in device_macros.h
//#ifdef FFTX_HIP
//#include <hip/hip_runtime.h>
//#endif

#include "fftx3.hpp"
#include "device_macros.h"

/*
 Real 3D convolution class for precompiled transforms

 null contruction should fail if a transform of size [NX,NY,NZ] is not available
*/

namespace fftx {
  
  template <int DIM, typename T_IN, typename T_OUT>
  class transformer
  {
  public:
    transformer(const point_t<DIM>& a_size)
    {
      m_size = a_size;
      // May change these in derived class.
      m_inputSize = m_size;
      m_outputSize = m_size;
      m_defined = false;
      // Do this in the derived class:
      // transformTuple_t* tupl = fftx_transformer_Tuple ( m_size );
      // setInit(tuple);
    }
    
    ~transformer()
    {
      if (destroy_spiral != nullptr) destroy_spiral();
    }

#if defined(__CUDACC__) || defined(FFTX_HIP)
    DEVICE_EVENT_T m_start, m_stop;
    void kernelStart()
    {
      DEVICE_CHECK(DEVICE_EVENT_RECORD(m_start),
                   "device event record in kernelStart");
    }
    void kernelStop()
    {
      DEVICE_CHECK(DEVICE_EVENT_RECORD(m_stop),
                   "device event record in kernelStop");
      DEVICE_CHECK(DEVICE_SYNCHRONIZE(),
                   "device synchronize in kernelStop");
      DEVICE_CHECK(DEVICE_EVENT_SYNCHRONIZE(m_stop),
                   "device event synchronize in kernelStop");
      DEVICE_CHECK(DEVICE_EVENT_ELAPSED_TIME(&m_GPU_milliseconds, m_start, m_stop),
                   "device event elapsed time in kernelStop");
    }
#else
    void kernelStart(){ }
    void kernelStop(){ }
#endif

    virtual inline bool defined() = 0;

    // virtual fftx::handle_t transform(fftx::array_t<DIM, T_IN>& a_src,
    // fftx::array_t<DIM, T_OUT>& a_dst) = 0;
    inline fftx::handle_t transform2(array_t<DIM, T_IN>& a_src,
                                     array_t<DIM, T_OUT>& a_dst)
    { // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world

      if (transform_spiral == nullptr)
        {
          // dummy return handle for now
          fftx::handle_t rtn;
          return rtn;
        }

      // Check that a_src and a_dst are the right sizes.

      box_t<DIM> srcDomain = a_src.m_domain;
      box_t<DIM> dstDomain = a_dst.m_domain;

      point_t<DIM> srcExtents = srcDomain.extents();
      point_t<DIM> dstExtents = dstDomain.extents();

      bool srcSame = (srcExtents == m_inputSize);
      bool dstSame = (dstExtents == m_outputSize);
      if (!srcSame)
        {
          std::cout << "error: transformer<" << DIM << ">"
                    << m_size << "::transform"
                    << " called with input array size " << srcExtents
                    << std::endl;
        }
      if (!dstSame)
        {
          std::cout << "error: transformer<" << DIM << ">"
                    << m_size << "::transform"
                    << " called with output array size " << dstExtents
                    << std::endl;
        }

      if (srcSame && dstSame)
        {
          return transform2Buffers(a_src.m_data.local(), a_dst.m_data.local());
        }

      // dummy return handle for now
      fftx::handle_t rtn;
      return rtn;
    }
    
    inline fftx::handle_t transform2Buffers(T_IN* a_src,
                                            T_OUT* a_dst)
    {
      double* inputLocal = (double*) a_src;
      double* outputLocal = (double*) a_dst;
      double* symLocal = nullptr;
      
      kernelStart();
      std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
      transform_spiral(outputLocal, inputLocal, symLocal);
      kernelStop();
      std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
      m_CPU_milliseconds = time_span.count()*1000;
      
      // dummy return handle for now
      fftx::handle_t rtn;
      return rtn;
    }

    bool isDefined() { return m_defined; }

    double CPU_milliseconds() { return m_CPU_milliseconds; }
    double GPU_milliseconds() { return m_GPU_milliseconds; }

    point_t<DIM> size() { return m_size; }
    point_t<DIM> inputSize() { return m_inputSize; }
    point_t<DIM> outputSize() { return m_outputSize; }

    point_t<DIM> sizeHalf()
    {
      point_t<DIM> ret = m_size;
#if FFTX_COMPLEX_TRUNC_LAST
      ret[DIM-1] = m_size[DIM-1]/2 + 1;
#else
      ret[0] = m_size[0]/2 + 1;
#endif
      return ret;
    }

    virtual std::string shortname() = 0;
    
    virtual std::string name()
    {
#define BUFFERLEN 50
      char buffer[BUFFERLEN];
      snprintf(buffer, BUFFERLEN, "%s<%d>[%d,%d,%d]", shortname().c_str(), DIM,
               this->m_size[0], this->m_size[1], this->m_size[2]);
      std::string str(buffer);
      return str;
    }
      
  protected:

    bool m_defined;

    point_t<DIM> m_size;

    point_t<DIM> m_inputSize;
    point_t<DIM> m_outputSize;

    double m_CPU_milliseconds = 0.;
    float  m_GPU_milliseconds = 0.;

    void setInit(transformTuple_t* a_tupl)
    {
      // look up this transform size in the database.
      // I would prefer if this was a constexpr kind of thing where we fail at compile time
      // a_tupl = fftx_transformer_Tuple ( m_size );
      if (a_tupl == nullptr)
        {
          m_defined = false;
          printf("%s is not in the library.\n", name().c_str());
        }
      else
        {
          // printf("transformer: this size is in the library. Initializing.\n");
          printf("%s is in the library. Initializing.\n", name().c_str());
          // Still need to set transform_spiral in the derived class.
          init_spiral = *a_tupl->initfp;
          destroy_spiral = *a_tupl->destroyfp;
        }

      if (init_spiral != nullptr)
        {
          init_spiral();
#if defined(__CUDACC__) || defined(FFTX_HIP)
          DEVICE_CHECK(DEVICE_EVENT_CREATE(&m_start),
                       "device event create start in setInit");
          DEVICE_CHECK(DEVICE_EVENT_CREATE(&m_stop),
                       "device event create stop in setInit");
#endif
          m_defined = true;
        }
    }

    // private:
    void (*init_spiral)() = nullptr;
    void (*transform_spiral)(double*, double*, double*) = nullptr;
    void (*destroy_spiral)() = nullptr;
  };
}

#endif  
