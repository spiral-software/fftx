#ifndef FFTX_mddft_PRECOMPILE_H
#define FFTX_mddft_PRECOMPILE_H

//
//  Copyright (c) 2018-2025, Carnegie Mellon University
//  All rights reserved.
//
//  See LICENSE file for full information
//

#include "fftxtransformer.precompile.hpp"

/*
 Inverse Complex to Complex DFT class for precompiled transforms

 null contruction should fail if a transform of size [NX,NY,NZ] is not available
*/

namespace fftx {
  
  template <int DIM>
  class mddft : public transformer<DIM, std::complex<double>, std::complex<double>>
  {
  public:
    mddft(const point_t<DIM>& a_size) :
      transformer<DIM, std::complex<double>, std::complex<double>>(a_size)
    {
      // fftx::OutStream() << "Defining mddft<" << DIM << ">" << this->m_size
      // << std::endl;
      // look up this transform size in the database.
      // I would prefer if this was a constexpr kind of thing where we fail at compile time
      transformTuple_t* tupl = fftx_mddft_Tuple ( this->m_size );
      this->setInit(tupl);
      if (tupl != NULL) this->transform_spiral = *tupl->runfp;
    }
    
    ~mddft()
    {
      // in base class
      // if (destroy_spiral != nullptr) destroy_spiral();
    }

    inline bool defined()
    {
      transformTuple_t* tupl = fftx_mddft_Tuple ( this->m_size );
      return (tupl != NULL);
    }

    inline fftx::handle_t transform(array_t<DIM, std::complex<double>>& a_src,
                                    array_t<DIM, std::complex<double>>& a_dst)
    { // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world
      return this->transform2(a_src, a_dst);
    }

    inline fftx::handle_t transformBuffers(std::complex<double>* a_src,
                                           std::complex<double>* a_dst)
    { // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world
      return this->transform2Buffers(a_src, a_dst);
    }

    std::string shortname()
    {
      return "mddft";
    }

  private:
    // void (*init_spiral)() = nullptr;
    // void (*transform_spiral)(double*, double*, double*) = nullptr;
    // void (*destroy_spiral)() = nullptr;
  };
}

#endif  
