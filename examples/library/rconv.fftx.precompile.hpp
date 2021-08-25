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
  class rconv : public transformer<DIM, double, double>
  {
  public:

    rconv(const point_t<DIM>& a_size) :
      transformer<DIM, double, double>(a_size)
    {
      // std::cout << "Defining rconv<" << DIM << ">" << this->m_size
      // << std::endl;
      m_sizeHalf = this->sizeHalf();
      this->m_inputSize = this->m_size;
      this->m_outputSize = this->m_size;
      // look up this transform size in the database.
      // I would prefer if this was a constexpr kind of thing where we fail at compile time
      transformTuple_t* tupl = fftx_rconv_Tuple ( this->m_size );
      this->setInit(tupl);
      if (tupl != NULL) transform_spiral = *tupl->runfp;
    }

    ~rconv()
    {
      // in base class
      // if (destroy_spiral != nullptr) destroy_spiral();
    }

    inline bool defined()
    {
      transformTuple_t* tupl = fftx_rconv_Tuple ( this->m_size );
      return (tupl != NULL);
    }

    inline fftx::handle_t transform(array_t<DIM, double>& a_src,
                                    array_t<DIM, double>& a_dst,
                                    array_t<DIM, double>& a_sym)
    { // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world

      // Check that a_src and a_dst are the right sizes,
      // and that a_sym is of size m_sizeHalf.

      box_t<DIM> srcDomain = a_src.m_domain;
      box_t<DIM> dstDomain = a_dst.m_domain;
      box_t<DIM> symDomain = a_sym.m_domain;

      point_t<DIM> srcExtents = srcDomain.extents();
      point_t<DIM> dstExtents = dstDomain.extents();
      point_t<DIM> symExtents = symDomain.extents();

      bool srcSame = (srcExtents == this->m_inputSize);
      bool dstSame = (dstExtents == this->m_inputSize);
      bool symSame = (symExtents == m_sizeHalf);
      if (!srcSame)
        {
          std::cout << "error: rconv<" << DIM << ">"  << (this->m_size) << "::transform"
                    << " called with input array size " << srcExtents
                    << std::endl;
        }
      if (!dstSame)
        {
          std::cout << "error: rconv<" << DIM << ">"  << (this->m_size) << "::transform"
                    << " called with output array size " << dstExtents
                    << std::endl;
        }
      if (!symSame)
        {
          std::cout << "error: rconv<" << DIM << ">"  << (this->m_size) << "::transform"
                    << " needs symbol array size " << m_sizeHalf
                    << " but called with size " << symExtents
                    << std::endl;
        }
      if (srcSame && dstSame && symSame)
        {
          double* inputLocal = (double*) (a_src.m_data.local());
          double* outputLocal = (double*) (a_dst.m_data.local());
          double* symLocal = (double*) (a_sym.m_data.local());

          this->kernelStart();
          std::chrono::high_resolution_clock::time_point t1 =
            std::chrono::high_resolution_clock::now();
          transform_spiral(outputLocal, inputLocal, symLocal);
          this->kernelStop();
          std::chrono::high_resolution_clock::time_point t2 =
            std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
          this->m_CPU_milliseconds = time_span.count()*1000;
        }

      // dummy return handle for now
      fftx::handle_t rtn;
      return rtn;
    }

    
    std::string shortname()
    {
      return "rconv";
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
