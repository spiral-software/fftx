

    #ifndef mddft_CODEGEN_H
    #define mddft_CODEGEN_H

    #include "fftx3.hpp"

    extern void init_mddft_spiral(); 
    extern void mddft_spiral(double* Y, double* X, double* symvar); 
    extern void destroy_mddft_spiral();

   namespace mddft
   {
    double CPU_milliseconds=0;
    float  GPU_milliseconds=0;
#ifdef __CUDACC__
    cudaEvent_t start, stop;
    void kernelStart() {cudaEventRecord(start);}
    void kernelStop()
    {
     cudaEventRecord(stop);
     cudaDeviceSynchronize();
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&GPU_milliseconds, start, stop);
    }
#else
    void kernelStart(){ }
    void kernelStop(){ }
#endif
    inline void init(){ 
          init_mddft_spiral();
#ifdef __CUDACC__
         cudaEventCreate(&start);
         cudaEventCreate(&stop);
#endif
           }
    inline void trace();
    inline fftx::handle_t transform(fftx::array_t<3, std::complex<double>>& source,
                                    fftx::array_t<3, std::complex<double>>& destination,
                                    fftx::array_t<3, double>& symvar)
    {   // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world
        double* input;
        double* output;
        double* sym;
        input = (double*)(source.m_data.local());
        output = (double*)(destination.m_data.local());
        sym = (double*)(symvar.m_data.local());

        kernelStart();
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
           mddft_spiral(output, input, sym);
        kernelStop();
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
        CPU_milliseconds = time_span.count()*1000;
    // dummy return handle for now
      fftx::handle_t rtn;
      return rtn;
    }

 
    inline fftx::handle_t transform(fftx::array_t<3, std::complex<double>>& source,
                                    fftx::array_t<3, std::complex<double>>& destination)
    {   // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world
        double* input;
        double* output;
        double* sym=nullptr;
        input = (double*)(source.m_data.local());
        output = (double*)(destination.m_data.local());
  
        kernelStart();
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
           mddft_spiral(output, input, sym);
        kernelStop();
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
        CPU_milliseconds = time_span.count()*1000;

    // dummy return handle for now
      fftx::handle_t rtn;
      return rtn;
    }
    //inline void destroy(){ destroy_mddft_spiral();}
    inline void destroy(){ }
  };

 #endif  
