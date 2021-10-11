#ifndef mdprdft_PRECOMPILE_H
#define mdprdft_PRECOMPILE_H

#include "fftx3.hpp"
#include "transformer.fftx.precompile.hpp"

/*
 Inverse Complex to Complex DFT class for precompiled transforms

 null contruction should fail if a transform of size [NX,NY,NZ] is not available
*/

namespace fftx {
  
  template <int DIM>
  class mdprdft : public transformer<DIM, double, std::complex<double>>
  {
  public:
    mdprdft(const point_t<DIM>& a_size) :
      transformer<DIM, double, std::complex<double>>(a_size)
    {
      this->m_outputSize = this->sizeHalf();
      // look up this transform size in the database.
      // I would prefer if this was a constexpr kind of thing where we fail at compile time
      transformTuple_t* tupl = fftx_mdprdft_Tuple ( this->m_size );
      this->setInit(tupl);
      if (tupl != nullptr) this->transform_spiral = *tupl->runfp;
    }
    
    ~mdprdft()
    {
      // in base class
      // if (destroy_spiral != nullptr) destroy_spiral();
    }

    inline bool defined()
    {
      fftx::point_t<DIM> sz = this->m_size;
      // transformTuple_t* tupl = fftx_mdprdft_Tuple ( this->m_size );
      transformTuple_t* tupl = fftx_mdprdft_Tuple ( sz );
      return (tupl != nullptr);
    }

    inline fftx::handle_t transform(array_t<DIM, double>& a_src,
                                    array_t<DIM, std::complex<double>>& a_dst)
    { // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world
      return this->transform2(a_src, a_dst);
    }

    inline fftx::handle_t transformBuffers(double* a_src,
                                           std::complex<double>* a_dst)
    { // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world
      return this->transform2Buffers(a_src, a_dst);
    }

    std::string shortname()
    {
      return "mdprdft";
    }

  protected:
    // point_t<DIM> m_sizeHalf;

  private:
    // void (*init_spiral)() = nullptr;
    // void (*transform_spiral)(double*, double*, double*) = nullptr;
    // void (*destroy_spiral)() = nullptr;
  };
}

#endif  
