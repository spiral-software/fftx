#ifndef transformer_PRECOMPILE_H
#define transformer_PRECOMPILE_H

#ifdef FFTX_HIP
#include <hip/hip_runtime.h>
#endif

#include "fftx3.hpp"
// #include <string>

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
      //      std::cout << "Defining transformer<" << DIM << ">" << m_size
      //                << std::endl;
      // Do this in the derived class:
      // transformTuple_t* tupl = fftx_transformer_Tuple ( m_size );
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
#ifdef FFTX_HIP
    hipEvent_t m_start, m_stop;
    void kernelStart() {hipEventRecord(m_start);}
    void kernelStop()
    {
      hipEventRecord(m_stop);
      hipDeviceSynchronize();
      hipEventSynchronize(m_stop);
      hipEventElapsedTime(&m_GPU_milliseconds, m_start, m_stop);
    }
#else
    void kernelStart(){ }
    void kernelStop(){ }
#endif
#endif

    // virtual fftx::handle_t transform(fftx::array_t<DIM, T_IN>& a_src,
    // fftx::array_t<DIM, T_OUT>& a_dst) = 0;
    inline fftx::handle_t transform2(array_t<DIM, T_IN>& a_src,
                                     array_t<DIM, T_OUT>& a_dst)
    { // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world

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
          double* inputLocal = (double*) (a_src.m_data.local());
          double* outputLocal = (double*) (a_dst.m_data.local());
          double* symLocal = NULL;

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
        }

      // dummy return handle for now
      fftx::handle_t rtn;
      return rtn;
    }
    
    double CPU_milliseconds() { return m_CPU_milliseconds; }
    double GPU_milliseconds() { return m_GPU_milliseconds; }

    point_t<DIM> size() { return m_size; }
    point_t<DIM> inputSize() { return m_inputSize; }
    point_t<DIM> outputSize() { return m_outputSize; }

    point_t<DIM> sizeHalf()
    {
      point_t<DIM> ret = m_size;
      ret[0] = m_size[0]/2 + 1;
      return ret;
    }

    virtual std::string shortname() = 0;
    
    virtual std::string name()
    {
      char buffer[50];
      sprintf(buffer, "%s<%d>[%d,%d,%d]", shortname().c_str(), DIM,
              this->m_size[0], this->m_size[1], this->m_size[2]);
      std::string str(buffer);
      return str;
    }
      
  protected:

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
      if (a_tupl == NULL)
        {
          // printf("transformer: this size is not in the library.\n");
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
#ifdef __CUDACC__
          cudaEventCreate(&m_start);
          cudaEventCreate(&m_stop);
#else
#ifdef FFTX_HIP
          hipEventCreate(&m_start);
          hipEventCreate(&m_stop);
#endif
#endif
        }
    }

    // private:
    void (*init_spiral)() = nullptr;
    void (*transform_spiral)(double*, double*, double*) = nullptr;
    void (*destroy_spiral)() = nullptr;
  };
};

#endif  
