#ifndef imddft_PRECOMPILE_H
#define imddft_PRECOMPILE_H

#include "fftx3.hpp"
#include "transformer.fftx.precompile.hpp"

/*
 Inverse Complex to Complex DFT class for precompiled transforms

 null contruction should fail if a transform of size [NX,NY,NZ] is not available
*/

namespace fftx {
  
  template <int DIM>
  class imddft : public transformer<DIM>
  {
  public:
    imddft(const point_t<DIM>& a_size) : transformer<DIM>(a_size)
    {
      std::cout << "Defining imddft<" << DIM << ">" << this->m_size
                << std::endl;
      // look up this transform size in the database.
      // I would prefer if this was a constexpr kind of thing where we fail at compile time
      transformTuple_t* tupl = fftx_imddft_Tuple ( this->m_size );
      this->setInit(tupl);
      if (tupl != NULL) transform_spiral = *tupl->runfp;
    }
    
    ~imddft()
    {
      // in base class
      // if (destroy_spiral != nullptr) destroy_spiral();
    }

    inline fftx::handle_t transform(array_t<DIM, std::complex<double>>& a_src,
                                    array_t<DIM, std::complex<double>>& a_dst)
    { // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world

      // Check that a_src and a_dst are of size m_size.

      box_t<DIM> srcDomain = a_src.m_domain;
      box_t<DIM> dstDomain = a_dst.m_domain;

      point_t<DIM> srcExtents = srcDomain.extents();
      point_t<DIM> dstExtents = dstDomain.extents();

      bool srcSame = (srcExtents == this->m_size);
      bool dstSame = (dstExtents == this->m_size);
      if (!srcSame)
        {
          std::cout << "error: imddft<" << DIM << ">"  << (this->m_size) << "::transform"
                    << " called with input array size " << srcExtents
                    << std::endl;
        }
      if (!dstSame)
        {
          std::cout << "error: imddft<" << DIM << ">"  << (this->m_size) << "::transform"
                    << " called with output array size " << dstExtents
                    << std::endl;
        }

      if (srcSame && dstSame)
        {
          double* inputLocal = (double*) (a_src.m_data.local());
          double* outputLocal = (double*) (a_dst.m_data.local());
          double* symLocal = NULL;

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
 

  private:
    // void (*init_spiral)() = nullptr;
    void (*transform_spiral)(double*, double*, double*) = nullptr;
    // void (*destroy_spiral)() = nullptr;
  };
};

#endif  
