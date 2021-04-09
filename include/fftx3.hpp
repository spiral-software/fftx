
/*
    ___________________  __
   / ____/ ____/_  __/ |/ /
  / /_  / /_    / /  |   / 
 / __/ / __/   / /  /   |  
/_/   /_/     /_/  /_/|_|  
                           

*/

#ifndef FFTX_HEADER
#define FFTX_HEADER


#include <complex>
#include <regex>
#include <memory>
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <chrono>

#include <map>
#include <string>

#include <array>

#include <cassert>
#include <iomanip>
/*! \mainpage FFTX Package
 *
 * \section intro_sec Introduction
 *
 * This is the introduction.
 *
 * \section install_sec Installation
 *
 * \subsection step1 Step 1: Opening the box
 *
 * etc...
 */

namespace fftx
{

  /**
   Is this a FFTX codegen program, or is this application code using a generated transform.
  */
  bool tracing = false; // when creating a trace program user sets this to 'true'

  /**
    counter for genereated variable names duringn FFTX codegen tracing.  Not meant for FFTX users but can be
    used when debugging codegen itself
  */
  uint64_t ID=1; // variable naming counter
  
  typedef int intrank_t; // just useful for self-documenting code.

  struct handle_implem_t;

  struct handle_t
  {
  private:
    std::shared_ptr<handle_implem_t> m_implem;
  };

  /**
   non-owning global ptr object.  Can be used in place of upcxx::global_ptr

   Most of the FFTX API assumes that user application code owns their primary data structures.  This
   class encapsulates a user space raw data pointer for use within our transforms.  The destructor
   does not delete the pointer.  It is not reference-counting. It is not made to be unique.  They can be copied 
   moved, etc like a basic C struct.
  */
  template <typename T>
  class global_ptr
  {
    T* _ptr;
    intrank_t _domain;
    int _device;
  public:
    using element_type = T;
    global_ptr():_ptr{nullptr},_domain{0}, _device{0}{}
    /// strong constructor
    /**  Real strong constructor */
    global_ptr(T* ptr, int domain=0, int device=0)
      :_ptr{ptr}, _domain{domain}, _device{device}{ }

    bool is_null() const;

    /** Returns true is this local compute context can successfully call the local() function and expect
        to get a pointer that is dereferencable */
    bool is_local() const;

    /** what compute domain would answer "true" to "isLocal().  Currently this just tests if the MPI rank
        matches my_rank */
    intrank_t where() const {return _domain;}
    /** which GPU device is this pointer associated with */
    int device() const {return _device;}
    /** returns the raw pointer.  This pointer can only be dereferences is isLocal() ==true */
    T* local() {return _ptr;}
    /** returns the raw pointer.  This pointer can only be dereferences is isLocal() ==true */
    const T* local() const {return _ptr;}
    /** type erasure cast */
    operator global_ptr<void>(){ return global_ptr<void>(_ptr, _domain, _device);}
  };

 
    

  /** integer index into a Z^DIM space. */
  template<int DIM>
  struct point_t
  {
    int x[DIM];
    /** returns a new point_t in one lower dimension, dropping the last value */
    point_t<DIM-1> project() const;
       /** returns a new point_t in one lower dimension, dropping the first value */
    point_t<DIM-1> projectC() const;
    int operator[](unsigned char a_id) const {return x[a_id];}
    int& operator[](unsigned char a_id) {return x[a_id];}
    bool operator==(const point_t<DIM>& a_rhs) const;
    void operator=(int a_default);
    point_t<DIM> operator*(int scale) const;
    static point_t<DIM> Unit();
    static point_t<DIM> Zero();
    /** reverse the odering of the tuple, reutrn by value */
    point_t<DIM> flipped() const { point_t<DIM> rtn; for (int d=0; d<DIM; d++) { rtn[d] = x[DIM-1 - d]; } return rtn; }
  };


  /** a pair of point_t objects representing a contiguous range of points in Z^DIM */                            
  template<int DIM>
  struct box_t
  {
    box_t() = default;
    box_t(const point_t<DIM>&& a, const point_t<DIM>&& b)
      : lo(a), hi(b) { ; }
    point_t<DIM> lo, hi;
    std::size_t size() const;
    bool operator==(const box_t<DIM>& rhs) const {return lo==rhs.lo && hi == rhs.hi;}
    point_t<DIM> extents() const { point_t<DIM> rtn(hi); for(int i=0; i<DIM; i++) rtn[i]-=(lo[i]-1); return rtn;}
    /** returns a box_t object in one lower dimension, dropping the first coordinate value in both lo and hi */
    box_t<DIM-1> projectC() const
    {
      return box_t<DIM-1>(lo.projectC(),hi.projectC());
    }
  };

  /** non-owning view into a contiugous array of data.   This is a class that is foeshadowing a C++ class mdspan,
      a multi-dimensional extention to std::span
      
      if fftx::tracing == true, then array_t::array_t(const box_t<DIM>& ) construction is 
      a symbolic placeholder in a computational DAG that is translated into the code generator.

      if fftx::tracing == false, then array_t::array_t(const box_t<DIM>&) will allocate a global_ptr sized 
      to hold box_t::size elements of data.
 */
  template<int DIM, typename T>
  struct array_t
  {
 
    array_t() = default;
    /** string constructor from an aliased global_ptr object.  This constructor is an error when fftx::tracing==true*/
    array_t(global_ptr<T>&& p, const box_t<DIM>& a_box)
      :m_data(p), m_domain(a_box) {;}
    array_t(const box_t<DIM>& m_box):m_domain(m_box)
    { if(tracing)
        {
          m_data = global_ptr<T>((T*)ID);
          std::cout<<"var_"<<ID<<":= var(\"var_"<<ID<<"\", BoxND("<<m_box.extents()<<", TReal));\n";
          ID++;
        }
      else m_data = global_ptr<T>(new T[m_box.size()]);
    }
    
    global_ptr<T> m_data;
    box_t<DIM>    m_domain;
    array_t<DIM, T> subArray(box_t<DIM>&& subbox);
    uint64_t id() const { assert(tracing); return (uint64_t)m_data.local();}
  };

  
  /**
   * \defGroup FFA  global free functions in fftx namespace used in application programs
   */
  /// @page page1 FFTX Free functions for Applications
  /// @ingroup FFA
  
  /**
   * \defGroup FFS global free functions in fftx namespace used in code generation programs
   */
  
  /**apply function f to each point in array where Func has the signature
   *  void f(T& value, const point_t<DIM>& location)
   * \addToGroup FFA
  */
  template<int DIM, typename T, typename Func>
  void forall(Func f, array_t<DIM, T>& array);

  /**apply function f to each point in two arrays where Func has the signature
   * void f(T1& value, const T2&, const point_t<DIM>& location)
   *  \ingroup FFA
  */
  template<int DIM, typename T1, typename T2, typename Func>
  void forall(Func f, array_t<DIM, T1>& array, const array_t<DIM, T2>& array2);


  /** component alias  Subselects outer-most dimension (the not contiguous one) 
   *   \ingroup FFS
  */
  template<int DIM, typename T>
  array_t<DIM-1, T> nth(array_t<DIM, T>& array, int index)
  {
    box_t<DIM-1> b = array.m_domain.projectC();
    array_t<DIM-1, T> rtn(b);
    std::cout<<"var_"<<(uint64_t)rtn.m_data.local()<<":=nth(var_"<<(uint64_t)array.m_data.local()<<","<<index<<");\n";
    return rtn;
  }

  template<int DIM, typename T>
  void copy(array_t<DIM, T>& dest, const array_t<DIM, T>& src)
  {
    std::cout<<"    TDAGNode(TGath(fBox("<<src.m_domain.extents()<<")),var_"<<dest.id()<<", var_"<<src.id()<<"),\n";
  }

  void rawScript(const std::string& a_rawScript)
  {
    std::cout<<"\n"<<a_rawScript<<"\n";
  }

  template <typename T>
  struct TypeName
  {
    static const char* Get()
    {
      return typeid(T).name();
    }
  };

// a specialization for each type of those you want to support
// and don't like the string returned by typeid
  template <>
  struct TypeName<double>
  {
    static const char* Get()
    {
      return "double";
    }
  };
  template <>
  struct TypeName<std::complex<double>>
  {
    static const char* Get()
    {
      return "std::complex<double>";
    }
  };
  
  template<typename T, std::size_t COUNT>
  inline std::ostream& operator<<(std::ostream& os, const std::array<T, COUNT>& arr)
  {
    os<<std::fixed<<std::setprecision(2);
    os<<"["<<arr[0];
    for(int i=1; i<COUNT; i++) os<<","<<arr[i];
    os<<"]";
    return os;
  }
  
  template<int DIM>
  void MDDFT(const point_t<DIM>& extents, int batch,
             array_t<DIM, std::complex<double>>& destination,
             array_t<DIM, std::complex<double>>& source)
  {
    std::cout<<"   TDAGNode(TTensorI(MDDFT("<<extents<<",-1),"<<batch<<",APar, APar), var_"<<destination.id()<<",var_"<<source.id()<<"),\n";
  }
  
  template<int DIM>
  void IMDDFT(const point_t<DIM>& extents, int batch,
             array_t<DIM, std::complex<double>>& destination,
             array_t<DIM, std::complex<double>>& source)
  {
    std::cout<<"   TDAGNode(TTensorI(MDDFT("<<extents<<",1),"<<batch<<",APar, APar), var_"<<destination.id()<<",var_"<<source.id()<<"),\n";
  }
    
  template<int DIM>
  void MDPRDFT(const point_t<DIM>& extent, int batch,
               array_t<DIM+1, double>& destination,
               array_t<DIM+1, double>& source)
  {
    std::cout<<"    TDAGNode(TTensorI(MDPRDFT("<<extent<<",-1),"<<batch<<",APar,APar), var_"<<destination.id()<<",var_"<<source.id()<<"),\n";
  }

  template<int DIM>
  void IMDPRDFT(const point_t<DIM>& extent, int batch,
               array_t<DIM+1, double>& destination,
               array_t<DIM+1, double>& source)
  {
    std::cout<<"    TDAGNode(TTensorI(IMDPRDFT("<<extent<<",1),"<<batch<<",APar,APar), var_"<<destination.id()<<",var_"<<source.id()<<"),\n";
  }

  template<int DIM>
  void PRDFT(const point_t<DIM>& extent,
             array_t<DIM, std::complex<double>>& destination,
             array_t<DIM, double>& source)
  {
    std::cout<<"    TDAGNode(MDPRDFT("<<extent<<",-1), var_"<<destination.id()<<",var_"<<source.id()<<"),\n"; // FIXME: was 1, not -1.
  }

  template<int DIM>
  void IPRDFT(const point_t<DIM>& extent,
              array_t<DIM, double>& destination,
              array_t<DIM, std::complex<double>>& source)
  {
    std::cout<<"    TDAGNode(IMDPRDFT("<<extent<<",1), var_"<<destination.id()<<",var_"<<source.id()<<"),\n"; // FIXME: was -1, not 1.
  }

  template<int DIM>
  void kernel(const array_t<DIM, double>& symbol,
              array_t<DIM, std::complex<double>>& destination,
              const array_t<DIM, std::complex<double>>& source)
  {
    std::cout<<"    TDAGNode(Diag(diagTensor(FDataOfs(symvar,"<<symbol.m_domain.size()<<",0),fConst(TReal, 2, 1))), var_"<<destination.id()<<",var_"<<source.id()<<"),\n";
  }

  template<int DIM>
  void kernel(const array_t<DIM, std::complex<double>>& symbol,
              array_t<DIM, std::complex<double>>& destination,
              const array_t<DIM, std::complex<double>>& source)
  {
    std::cout<<"    TDAGNode(RCDiag(FDataOfs(symvar,"<<2*symbol.m_domain.size()<<",0)), var_"<<destination.id()<<",var_"<<source.id()<<"),\n";
  }

  void include(const char* includeFile)
  {
    std::cout<<"opts.includes:=opts.includes::["<<includeFile<<"];\n";
  }

  template<int DIM, typename T>
  void zeroEmbedBox(array_t<DIM, T>& destination, const array_t<DIM, T>& source)
  {
    std::cout<<"    TDAGNode(ZeroEmbedBox("<<destination.m_domain.extents()<<",[";
    for(int i=0; i<DIM; i++)
      {
        std::cout<<"["<<source.m_domain.lo[i]<<".."<<source.m_domain.hi[i]<<"]";
        if(i<DIM-1) std::cout<<",";
      }
    std::cout<<"]), var_"<<destination.id()<<",var_"<<source.id()<<"),\n";
  }

  template<int DIM, typename T>
  void extractBox(array_t<DIM, T>& destination, const array_t<DIM, T>& source)
  {
    std::cout<<"    TDAGNode(ExtractBox("<<source.m_domain.extents()<<",[";
    for(int i=0; i<DIM; i++)
      {
        std::cout<<"["<<destination.m_domain.lo[i]<<".."<<destination.m_domain.hi[i]<<"]";
        if(i<DIM-1) std::cout<<",";
      }
    std::cout<<"]), var_"<<destination.id()<<",var_"<<source.id()<<"),\n";
  }
  
  std::string inputType = "double";
  int inputCount = 1;
  std::string outputType = "double";
  int outputCount = 1;
  template<int DIM, typename T, std::size_t COUNT>

  
  void setInputs(const std::array<array_t<DIM, T>, COUNT>& a_inputs)
  {
    inputType = TypeName<T>::Get();
    for(int i=0; i<COUNT; i++)
      {
        std::cout<<"var_"<<a_inputs[i].id()<<":= nth(X,"<<i<<");\n";
      }
  }
  template<int DIM, typename T>
  void setInputs(const array_t<DIM, T>& a_inputs)
  {
    inputType = TypeName<T>::Get();
    std::cout<<"var_"<<a_inputs.id()<<":= X;\n";
  }
  
  template<int DIM, typename T, std::size_t COUNT>
  void setOutputs(const std::array<array_t<DIM, T>, COUNT>& a_outputs)
  {
    outputType = TypeName<T>::Get();
    for(int i=0; i<COUNT; i++)
      {
        std::cout<<"var_"<<a_outputs[i].id()<<":= nth(Y,"<<i<<");\n";
      }
  }
  template<int DIM, typename T>
  void setOutputs(const array_t<DIM, T>& a_outputs)
  {
    outputType = TypeName<T>::Get();
    std::cout<<"var_"<<a_outputs.id()<<":= Y;\n";
  }

  template<int DIM, typename T, std::size_t COUNT>
  void setSymbol(const  std::array<array_t<DIM, T>, COUNT>& a_symbol)
  {
    std::cout<<"symvar := var(\"sym\", TPtr(TPtr(TReal)));\n";
  }
  
  template<int DIM, typename T>
  void resample(const std::array<double, DIM>& shift,
                array_t<DIM,T>& destination,
                const array_t<DIM,T>& source)
  {
    std::cout<<"    TDAGNode(TResample("
             <<destination.m_domain.extents()<<","
             <<source.m_domain.extents()<<","<<shift<<"),"
             <<"var_"<<destination.id()<<","
             <<"var_"<<source.id()<<"),\n";
  }
  void openDAG()
  {
  //  std::cout<<"conf := FFTXGlobals.defaultWarpXConf();\n";
  //  std::cout<<"opts := FFTXGlobals.getOpts(conf);\n";                                     
  //  std::cout<<"symvar := var(\"sym\", TPtr(TPtr(TReal)));\n";
    std::cout<<"transform:= TFCall(TDecl(TDAG([\n";
  }

  void openScalarDAG()
  {
    std::cout<<"symvar := var(\"sym\", TPtr(TReal));\n";
    std::cout<<"transform:= TFCall(TDecl(TDAG([\n";
  }
  
  
  template<typename T, int DIM, unsigned long COUNT>
  void closeDAG(std::array<array_t<DIM,T>, COUNT>& localVars, const char* name)
  {
    static const char* header_template = R"(

    #ifndef PLAN_CODEGEN_H
    #define PLAN_CODEGEN_H

    #include "fftx3.hpp"

    extern void init_PLAN_spiral(); 
    extern void PLAN_spiral(double** Y, double** X, double** symvar); 
    extern void destroy_PLAN_spiral();

   namespace PLAN
   {
    inline void init(){ init_PLAN_spiral();}
    inline void trace();
    template<std::size_t IN_DIM, std::size_t OUT_DIM, std::size_t S_DIM>
    inline fftx::handle_t transform(std::array<fftx::array_t<DD, S_TYPE>,IN_DIM>& source,
                                    std::array<fftx::array_t<DD, D_TYPE>,OUT_DIM>& destination,
                                    std::array<fftx::array_t<DD, double>,S_DIM>& symvar)
    {   // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world
        double* input[IN_DIM];
        double* output[OUT_DIM];
        double* sym[S_DIM];
        for(int i=0; i<IN_DIM; i++) input[i] = (double*)(source[i].m_data.local());
        for(int i=0; i<OUT_DIM; i++) output[i] = (double*)(destination[i].m_data.local());
        for(int i=0; i<S_DIM; i++) sym[i] = (double*)(symvar[i].m_data.local());

        PLAN_spiral(output, input, sym);
   
    // dummy return handle for now
      fftx::handle_t rtn;
      return rtn;
    }

    template<std::size_t IN_DIM, std::size_t OUT_DIM>
    inline fftx::handle_t transform(std::array<fftx::array_t<DD, S_TYPE>,IN_DIM>& source,
                                    std::array<fftx::array_t<DD, D_TYPE>,OUT_DIM>& destination)
    {   // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world
        double* input[IN_DIM];
        double* output[OUT_DIM];
        double** sym=nullptr;
        for(int i=0; i<IN_DIM; i++) input[i] = (double*)(source[i].m_data.local());
        for(int i=0; i<OUT_DIM; i++) output[i] = (double*)(destination[i].m_data.local());
  

        PLAN_spiral(output, input, sym);
   
    // dummy return handle for now
      fftx::handle_t rtn;
      return rtn;
    }
    //inline void destroy(){ destroy_PLAN_spiral();}
    inline void destroy(){ }
  };

 #endif  )";
   
   tracing = false;
   std::string headerName = std::string(name)+std::string(".fftx.codegen.hpp");
   std::ofstream headerFile(headerName);
   //DataTypeT<SOURCE> s;
   //DataTypeT<DEST> d;
   std::string header_text = std::regex_replace(header_template,std::regex("PLAN"),name);
   header_text = std::regex_replace(header_text, std::regex("S_TYPE"), inputType);
   header_text = std::regex_replace(header_text, std::regex("D_TYPE"), outputType);
   header_text = std::regex_replace(header_text, std::regex("DD"), std::to_string(DIM-1));
   
   headerFile<<header_text<<"\n";
   headerFile.close();
   
   std::cout<<"\n]),\n   [";
   if(COUNT==0)
     {}
   else
     {
      std::cout<<"var_"<<(uint64_t)localVars[0].m_data.local();
      for(int i=1; i<COUNT; i++) std::cout<<", var_"<<(uint64_t)localVars[i].m_data.local();
     }
     std::cout<<"]\n),\n";
     std::cout<<"rec(XType:= TPtr(TPtr(TReal)), YType:=TPtr(TPtr(TReal)), fname:=\""<<name<<"_spiral\", params:= [symvar])\n"
              <<");\n";
     std::cout<<"prefix:=\""<<name<<"\";\n";
  }

  template<typename T, int DIM, unsigned long COUNT>
  std::string varNames(const std::array<array_t<DIM,T>, COUNT>& a_vars)
  {
   std::string rtn;
   for(int i=0; i<COUNT; i++)
      {
        rtn +="var_";
        rtn += std::to_string((uint64_t)a_vars[i].m_data.local());
        if(i+1<COUNT) rtn +=",";
      }
    return rtn;
  }                                           

template<int DIM>
  void closeScalarDAG(std::string localVarNames, const char* name)
  {
    static const char* header_template = R"(

    #ifndef PLAN_CODEGEN_H
    #define PLAN_CODEGEN_H

    #include "fftx3.hpp"

    extern void init_PLAN_spiral(); 
    extern void PLAN_spiral(double* Y, double* X, double* symvar); 
    extern void destroy_PLAN_spiral();

   namespace PLAN
   {
    double CPU_milliseconds=0;
    float  GPU_milliseconds=0;
#ifdef __CUDACC__
    cudaEvent_t start, stop;
    void cudaStart() {cudaEventRecord(start);}
    void cudaStop()
    {
     cudaEventRecord(stop);
     cudaDeviceSynchronize();
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&GPU_milliseconds, start, stop);
    }
#else
    void cudaStart(){ }
    void cudaStop(){ }
#endif
    inline void init(){ 
          init_PLAN_spiral();
#ifdef __CUDACC__
         cudaEventCreate(&start);
         cudaEventCreate(&stop);
#endif
           }
    inline void trace();
    inline fftx::handle_t transform(fftx::array_t<DD, S_TYPE>& source,
                                    fftx::array_t<DD, D_TYPE>& destination,
                                    fftx::array_t<DD, double>& symvar)
    {   // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world
        double* input;
        double* output;
        double* sym;
        input = (double*)(source.m_data.local());
        output = (double*)(destination.m_data.local());
        sym = (double*)(symvar.m_data.local());

        cudaStart();
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
           PLAN_spiral(output, input, sym);
        cudaStop();
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
        CPU_milliseconds = time_span.count()*1000;
    // dummy return handle for now
      fftx::handle_t rtn;
      return rtn;
    }

 
    inline fftx::handle_t transform(fftx::array_t<DD, S_TYPE>& source,
                                    fftx::array_t<DD, D_TYPE>& destination)
    {   // for the moment, the function signature is hard-coded.  trace will
      // generate this in our better world
        double* input;
        double* output;
        double* sym=nullptr;
        input = (double*)(source.m_data.local());
        output = (double*)(destination.m_data.local());
  
        cudaStart();
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
           PLAN_spiral(output, input, sym);
        cudaStop();
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
        CPU_milliseconds = time_span.count()*1000;

    // dummy return handle for now
      fftx::handle_t rtn;
      return rtn;
    }
    //inline void destroy(){ destroy_PLAN_spiral();}
    inline void destroy(){ }
  };

 #endif  )";

   tracing = false;
   std::string headerName = std::string(name)+std::string(".fftx.codegen.hpp");
   std::ofstream headerFile(headerName);
   //DataTypeT<SOURCE> s;
   //DataTypeT<DEST> d;
   std::string header_text = std::regex_replace(header_template,std::regex("PLAN"),name);
   header_text = std::regex_replace(header_text, std::regex("S_TYPE"), inputType);
   header_text = std::regex_replace(header_text, std::regex("D_TYPE"), outputType);
   header_text = std::regex_replace(header_text, std::regex("DD"), std::to_string(DIM));
   
   headerFile<<header_text<<"\n";
   headerFile.close();

   std::cout<<"\n]),\n   [";
    // if(COUNT==0){}
    // else
    //   {
    //     std::cout<<"var_"<<(uint64_t)localVars[0].m_data.local();
    //     for(int i=1; i<COUNT; i++) std::cout<<", var_"<<(uint64_t)localVars[i].m_data.local();
    //   }
   std::cout <<localVarNames;
   std::cout<<"]\n),\n";
   std::cout<<"rec(fname:=\""<<name<<"_spiral\", params:= [symvar])\n"
            <<");\n";
   std::cout<<"prefix:=\""<<name<<"\";\n";
} 
 
 
  template<typename T, int DIM, unsigned long COUNT>
  void closeScalarDAG(const std::array<array_t<DIM,T>, COUNT>& a_vars, const char* name)
  {
    closeScalarDAG<DIM>(varNames(a_vars), name);
  }

template<typename T, typename T2, int DIM, unsigned long COUNT, unsigned long COUNT2>
  void closeScalarDAG(const std::array<array_t<DIM,T>, COUNT>& a_vars,
                      const std::array<array_t<DIM,T2>, COUNT2>& a_vars2, const char* name)
  {
    closeScalarDAG<DIM>(varNames(a_vars)+','+varNames(a_vars2), name);
  }
  
  template<int DIM>
  inline point_t<DIM> lengthsBox(const box_t<DIM>& a_bx)
  {
    point_t<DIM> lo = a_bx.lo;
    point_t<DIM> hi = a_bx.hi;
    point_t<DIM> lengths;
    for (int d = 0; d < DIM; d++)
      {
        lengths.x[d] = hi[d] - lo[d] + 1;
      }
    return lengths;
  }

  template<int DIM>
  inline bool isInBox(point_t<DIM> a_pt, const box_t<DIM>& a_bx)
  {
    point_t<DIM> lo = a_bx.lo;
    point_t<DIM> hi = a_bx.hi;
    for (int d = 0; d < DIM; d++)
      {
        if (a_pt[d] < lo[d]) return false;
        if (a_pt[d] > hi[d]) return false;
      }
    return true;
  }

  template<int DIM>
  inline size_t positionInBox(point_t<DIM> a_pt, const box_t<DIM>& a_bx)
  {
    point_t<DIM> lo = a_bx.lo;
    point_t<DIM> lengths = lengthsBox(a_bx);

    /*
    // Row-major order: Last dimension changes fastest.
    size_t disp = a_pt[0] - lo[0];
    for (int d = 1; d < DIM; d++)
      {
        disp *= lengths[d];
        disp += a_pt[d] - lo[d];
      }
    */

    // Column-major order: First dimension changes fastest.
    size_t disp = a_pt[DIM-1] - lo[DIM-1];
    for (int d = DIM-2; d >= 0; d--)
      {
        disp *= lengths[d];
        disp += a_pt[d] - lo[d];
      }

    return disp;
  }

  // inverse of positionInBox()
  template<int DIM>
  inline point_t<DIM> pointFromPositionBox(size_t a_ind, const box_t<DIM>& a_bx)
  {
    point_t<DIM> lo = a_bx.lo;
    point_t<DIM> lengths = lengthsBox(a_bx);

    point_t<DIM> pt;

    /*
    // Row-major order: Last dimension changes fastest.
    size_t disp = a_ind;
    for (int d = DIM-1; d >= 0; d--)
       {
          pt[d] = lo[d] + disp % lengths[d];
          disp = (disp - (pt[d] - lo[d])) / lengths[d];
       }
    */

    // Column-major order: First dimension changes fastest.
    size_t disp = a_ind;
    for (int d = 0; d < DIM; d++)
       {
          pt[d] = lo[d] + disp % lengths[d];
          disp = (disp - (pt[d] - lo[d])) / lengths[d];
       }

    return pt;
  }

 
  // helper meta functions===============
  template<int DIM>
  void projecti(int out[], const int in[] );

  template<>
  inline void projecti<0>(int out[], const int in[]) { return; }

  template<int DIM>
  inline void projecti(int out[], const int in[] )
  {
    out[DIM-1]=in[DIM-1]; projecti<DIM-1>(out, in);
  }

  template<int DIM>
  std::size_t bsize(int const lo[], int const hi[]);
  template<>
  inline std::size_t bsize<0>(int const lo[], int const hi[]){return 1;}
  template<int DIM>
  inline std::size_t bsize(int const lo[], int const hi[]){ return (hi[DIM-1]-lo[DIM-1]+1)*bsize<DIM-1>(lo, hi);}

  template<int DIM>
  inline point_t<DIM-1> point_t<DIM>::project() const
  {
    point_t<DIM-1> rtn;
    projecti<DIM-1>(rtn.x, x);
    return rtn;
  }

  template<int DIM>
  inline point_t<DIM-1> point_t<DIM>::projectC() const
  {
    point_t<DIM-1> rtn;
    for(int i=0; i<DIM-1; i++) rtn[i] = x[i+1];
    return rtn;
  }
  
  template<unsigned char DIM>
  inline bool equalInts(const int* a, const int* b) { return (a[DIM-1]==b[DIM-1])&&equalInts<DIM-1>(a, b);}
  template<>
  inline bool equalInts<0>(const int* a, const int* b) {return true;}
  
  template<int DIM>
  inline std::size_t box_t<DIM>::size() const { return bsize<DIM>(lo.x,hi.x);}

  template<int DIM>
  inline bool point_t<DIM>::operator==(const point_t<DIM>& a_rhs) const
  {
    return equalInts<DIM>(x, a_rhs.x);
  }
  
  template<int DIM>
  inline void point_t<DIM>::operator=(int a_value)
  {
    for(int i=0; i<DIM; i++) x[i]=a_value;
  }

  template<int DIM>
  inline point_t<DIM> point_t<DIM>::operator*(int a_scale) const
  {
    point_t<DIM> rtn(*this);
    for(int i=0; i<DIM; i++) rtn.x[i]*=a_scale;
    return rtn;
  }

  template<int DIM>
  inline point_t<DIM> point_t<DIM>::Unit()
  {
    point_t<DIM> rtn;
    for(int i=0; i<DIM; i++) rtn.x[i]=1;
    return rtn;
  }

  template<int DIM>
  inline point_t<DIM> point_t<DIM>::Zero()
  {
    point_t<DIM> rtn;
    for(int i=0; i<DIM; i++) rtn.x[i]=0;
    return rtn;
  } 
  
  template<int DIM, typename T, typename Func_P>
  struct forallHelper
  {
    static void f(T*& __restrict ptr, int* pvect, int* lo, int* hi, Func_P fp)
    {
      for(int i=lo[DIM-1]; i<=hi[DIM-1]; ++i)
        {
          pvect[DIM-1]=i;
          forallHelper<DIM-1, T, Func_P>::f(ptr, pvect, lo, hi, fp);
        }
    }
    template<typename T2>
    static void f2(T*& __restrict ptr1, const T2*& __restrict ptr2, int* pvect, int* lo, int* hi, Func_P fp)
    {
      for(int i=lo[DIM-1]; i<=hi[DIM-1]; ++i)
        {
          pvect[DIM-1]=i;
          forallHelper<DIM-1, T, Func_P>::f2(ptr1, ptr2, pvect, lo, hi, fp);
        }
    }
  };
  
  template<typename T, typename Func_P>
  struct forallHelper<1, T, Func_P>
  {
    static void f(T*& __restrict ptr, int* pvect, int* lo, int* hi, Func_P fp)
    {
      for(int i=lo[0]; i<=hi[0]; i++, ptr++)
        {
          pvect[0]=i;
          fp(*ptr);
        }
    }
    template<typename T2>
    static void f2(T*& __restrict ptr1,  const T2*& __restrict ptr2, int* pvect, int* lo, int* hi, Func_P fp)
    {
      for(int i=lo[0]; i<=hi[0]; i++, ptr1++, ptr2++)
        {
          pvect[0]=i;
          fp(*ptr1, *ptr2);
        }
    }
 
  };
  
  template<int DIM, typename T, typename Func>
  inline void forall(Func f, array_t<DIM, T>& array)
  {
    int* lo=array.m_domain.lo.x;
    int* hi=array.m_domain.hi.x;
    point_t<DIM> p = array.m_domain.lo;
    auto fp = [&](T& v){f(v, p);};
    T* ptr = array.m_data.local();
    forallHelper<DIM, T,decltype(fp) >::f(ptr, p.x, lo, hi,fp);
  }
  template<int DIM, typename T1, typename T2, typename Func>
  inline void forall(Func f, array_t<DIM, T1>& array, const array_t<DIM, T2>& array2)
  {
    int* lo=array.m_domain.lo.x;
    int* hi=array.m_domain.hi.x;
    point_t<DIM> p = array.m_domain.lo;
    auto fp = [&](T1& v, const T2& v2){f(v, v2, p);};
    T1* ptr = array.m_data.local();
    const T2* ptr2 = array2.m_data.local();
    forallHelper<DIM, T1,decltype(fp) >::f2(ptr, ptr2, p.x, lo, hi,fp);
  }
     
  template<unsigned char DIM>
  inline size_t dimHelper(int* lo, int* hi) {return (hi[DIM-1]-lo[DIM-1]+1)*dimHelper<DIM-1>(lo, hi);}
  template<>
  inline size_t dimHelper<0>(int* lo, int* hi){ return 1;}
  
  template<int DIM>
  inline std::size_t normalization(box_t<DIM> a_transformBox)
  {
    //return dimHelper<DIM>(a_transformBox.lo.x, a_transformBox.hi.x);
    return a_transformBox.size();
  }

  template<int DIM>
  inline std::ostream& operator<<(std::ostream& output, const point_t<DIM> p)
  {
    output<<"[";
    for(int i=0; i<DIM-1; i++)
      {
        output<<p.x[i]<<",";
      }
    output<<p.x[DIM-1]<<"]";
    return output;
  }
  
  template<int DIM>
  inline std::ostream& operator<<(std::ostream& output, const box_t<DIM>& b)
  {
    output<<"["<<b.lo<<","<<b.hi<<"]";
    return output;
  }


           
 

  
}


namespace fftx_helper
{
  inline size_t reverseBits(size_t x, int n) {
    size_t result = 0;
    for (int i = 0; i < n; i++, x >>= 1)
      result = (result << 1) | (x & 1U);
    return result;
  }



  inline void multiply(std::complex<double>& a, const std::complex<double>& b){ a*=b;}

  template<int C>
  inline void multiply(std::complex<double>(&a)[C], const std::complex<double>(&b)[C])
  {
    for(int i=0; i<C; i++) { a[i]*=b[i]; }
  }

  inline void assign(std::complex<double>& a, const std::complex<double>& b){ a=b;}

  template<int C>
  inline void assign(std::complex<double>(&a)[C], const std::complex<double>(&b)[C])
  {
    for(int i=0; i<C; i++) { a[i]=b[i]; }
  }
  inline void subtract(std::complex<double>& a, const std::complex<double>& b){ a-=b; }

  template<int C>
  inline void subtract(std::complex<double>(&a)[C], const std::complex<double>(&b)[C])
  {
    for(int i=0; i<C; i++) { a[i]-=b[i]; }
  }
  inline void increment(std::complex<double>& a, const std::complex<double>& b){ a+=b; }

  template<int C>
  inline void increment(std::complex<double>(&a)[C], const std::complex<double>(&b)[C])
  {
    for(int i=0; i<C; i++) { a[i]+=b[i]; }
  }


  template<int BATCH, typename T, int DIR = 1>
  static void batchtransformRadix2(int n, int stride, T* dvec[])

  {  
    static std::vector<std::complex<double>> expTable;
    int levels = 0;  // Compute levels = floor(log2(n))
    for (size_t temp = n; temp > 1U; temp >>= 1)
      {
        levels++;
      }
    if (static_cast<size_t>(1U) << levels != n)
      {
        throw std::domain_error("Length is not a power of 2");
      }
  
    // Trigonometric table
    if (expTable.size() != n/2)
      {
        expTable.resize(n/2);
        // This must be int, not size_t, because we will negate it.
        for (int i = 0; i < n / 2; i++)
          {
            // std::complex<double> k = std::complex<double>(0, -(2*DIR*i)*M_PI/n);
            // std::complex<double> tw = std::exp(k);
            double th = -(2*DIR*i)*M_PI/(n*1.);
            std::complex<double> tw = std::complex<double>(cos(th), sin(th));
            expTable[i] = tw;
          }
      }
    
    // Bit-reversed addressing permutation
    for (size_t i = 0; i < n; i++)
      {
        size_t j = reverseBits(i, levels);
        // If j == i, then no change.
        // If j != i, then swap, but only if j > i, so as not to duplicate.
        if (j > i)
          {
            for(int b=0; b<BATCH; b++)
              {
                std::swap(dvec[b][i*stride], dvec[b][j*stride]);
              }
          }
      }
  
    // Cooley-Tukey decimation-in-time radix-2 FFT.
    // From algorithm iterative-fft in Wikipedia "Cooley-Tukey FFT algorithm"
    for (size_t size = 2; size <= n; size *= 2)
      {
        size_t halfsize = size / 2;
        size_t tablestep = n / size;
        for (size_t k = 0; k < n; k += size)
          {
            for (size_t j = 0; j < halfsize; j++)
              {
                size_t indkj = (k + j)*stride;
                size_t indkjhalfsize = (k + j + halfsize)*stride;
                for (int b=0; b<BATCH; b++)
                  {
                    T* vec = dvec[b];
                    T temp1;
                    assign(temp1, vec[indkjhalfsize]);
                    // std::cout << "size=" << size << " tablestep=" << tablestep << " : k=" << k << " j=" << j << " j*tablestep = " << (j*tablestep) << "\n";
                    multiply(temp1, expTable[j*tablestep]);
                    T temp2;
                    assign(temp2, vec[indkj]);
                    increment(vec[indkj], temp1);
                    assign(vec[indkjhalfsize], temp2);
                    subtract(vec[indkjhalfsize], temp1);
                  }
                // std::cout << "k=" << k << " j=" << j << "\n";
              }
          }
        if (size == n)  // Prevent overflow in 'size *= 2'
          {
            break;
          }
      }
  }
}

#endif /*  end include guard FFTX_H */
