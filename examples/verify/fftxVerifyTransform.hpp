#ifndef FFTX_VERIFY_TRANSFORM_HEADER
#define FFTX_VERIFY_TRANSFORM_HEADER

#include <cmath> // Without this, abs returns zero!
#include <random>

#include "fftx3.hpp"
#include "fftxutilities.hpp"
#include "fftxinterface.hpp"

// Define {init|destroy|run}TransformFunc and transformTuple if undefined.

#ifndef FFTX_INITTRANSFORMFUNC
#define FFTX_INITTRANSFORMFUNC
typedef void ( * initTransformFunc ) ( void );
#endif

#ifndef FFTX_DESTROYTRANSFORMFUNC
#define FFTX_DESTROYTRANSFORMFUNC
typedef void ( * destroyTransformFunc ) ( void );
#endif

#ifndef FFTX_RUNTRANSFORMFUNC
#define FFTX_RUNTRANSFORMFUNC
typedef void ( * runTransformFunc )
( double *output, double *input, double *sym );
#endif

#ifndef FFTX_TRANSFORMTUPLE_T
#define FFTX_TRANSFORMTUPLE_T
typedef struct transformTuple {
    initTransformFunc    initfp;
    destroyTransformFunc destroyfp;
    runTransformFunc     runfp;
} transformTuple_t;
#endif

#include "fftxdevice_macros.h"
#include "fftxtransformer.precompile.hpp"

std::mt19937 generator;
// unifRealDist is uniform over the reals in (-1/2, 1/2).
std::uniform_real_distribution<double> unifRealDist;

// DO THIS:
// std::random_device rd;
// generator = std::mt19937(rd());
// unifRealDist = std::uniform_real_distribution<double>(-0.5, 0.5);


// mddft:  complex on full domain to complex on full domain
// imddft:  complex on full domain to complex on full domain
// mdprdft:  real on full domain to complex on truncated domain
// imdprdft:  complex on truncated domain to real on full domain

template<int DIM>
fftx::box_t<DIM> inDomainFromSize(std::string& a_name,
                                  fftx::point_t<DIM>& a_size)
{
  fftx::box_t<DIM> bx = domainFromSize(a_size);
  if (a_name == "imdprdft")
    { // this is the only case of truncated input domain
      bx.hi = truncatedComplexDimensions(a_size);
    }
  return bx;
}

template<int DIM>
fftx::box_t<DIM> outDomainFromSize(std::string& a_name,
                                   fftx::point_t<DIM>& a_size)
{
  fftx::box_t<DIM> bx = domainFromSize(a_size);
  if (a_name == "mdprdft")
    { // this is the only case of truncated output domain
      bx.hi = truncatedComplexDimensions(a_size);
    }
  return bx;
}

template<int DIM, typename T>
double absMaxRelDiffArray(fftx::array_t<DIM, T>& a_arr1,
                          fftx::array_t<DIM, T>& a_arr2)
{
  double absDiffMax = absMaxDiffArray(a_arr1, a_arr2);
  double abs1Max = absMaxArray(a_arr1);
  double rel = absDiffMax / abs1Max;
  return rel;
}

#if defined(FFTX_CUDA) || defined(FFTX_HIP)

template<typename T_IN, typename T_OUT>
struct deviceTransform3dType
{
  deviceTransform3dType() { };

  deviceTransform3dType(FFTX_DEVICE_FFT_TYPE a_tp,
                        int a_dir = 0)
  {
    m_tp = a_tp;
    m_dir = a_dir;
  }
                  
  FFTX_DEVICE_FFT_TYPE m_tp;

  int m_dir;

  fftx::point_t<3> m_size;

  fftx::point_t<3> size(fftx::box_t<3> a_inputDomain,
                        fftx::box_t<3> a_outputDomain)
  {
    fftx::point_t<3> tfmSize = a_inputDomain.extents();
    if (m_tp == FFTX_DEVICE_FFT_Z2D)
      { // exception for complex-to-real
        tfmSize = a_outputDomain.extents();
      }
    return tfmSize;
  }

  FFTX_DEVICE_FFT_RESULT plan3d(FFTX_DEVICE_FFT_HANDLE& a_plan,
                           fftx::point_t<3> a_tfmSize)
  {
    return FFTX_DEVICE_FFT_PLAN3D(&a_plan,
                             a_tfmSize[0], a_tfmSize[1], a_tfmSize[2],
                             m_tp);
  }

  FFTX_DEVICE_FFT_RESULT exec(FFTX_DEVICE_FFT_HANDLE a_plan,
                         T_IN* a_in,
                         T_OUT* a_out)
  {
    if (m_tp == FFTX_DEVICE_FFT_Z2Z)
      {
        return FFTX_DEVICE_FFT_EXECZ2Z(a_plan,
                                  (FFTX_DEVICE_FFT_DOUBLECOMPLEX*) a_in,
                                  (FFTX_DEVICE_FFT_DOUBLECOMPLEX*) a_out,
                                  m_dir);
      }
    else if (m_tp == FFTX_DEVICE_FFT_D2Z)
      {
        return FFTX_DEVICE_FFT_EXECD2Z(a_plan,
                                  (FFTX_DEVICE_FFT_DOUBLEREAL*) a_in,
                                  (FFTX_DEVICE_FFT_DOUBLECOMPLEX*) a_out);
      }
    else if (m_tp == FFTX_DEVICE_FFT_Z2D)
      {
        return FFTX_DEVICE_FFT_EXECZ2D(a_plan,
                                  (FFTX_DEVICE_FFT_DOUBLECOMPLEX*) a_in,
                                  (FFTX_DEVICE_FFT_DOUBLEREAL*) a_out);
      }
    else
      {
        return (FFTX_DEVICE_FFT_RESULT) -1;
      }
  }
};
  

template<typename T_IN, typename T_OUT>
struct deviceTransform3d
{
  deviceTransform3d(deviceTransform3dType<T_IN, T_OUT>& a_dtype,
                    fftx::box_t<3> a_inputDomain,
                    fftx::box_t<3> a_outputDomain)
  {
    m_dtype = a_dtype;
    // m_inputDomain = a_inputDomain;
    // m_outputDomain = a_outputDomain;
    m_tfmSize = a_inputDomain.extents();
    if (m_dtype.m_tp == FFTX_DEVICE_FFT_Z2D)
      { // exception for complex-to-real
        m_tfmSize = a_outputDomain.extents();
      }
    FFTX_DEVICE_FFT_CHECK(FFTX_DEVICE_FFT_PLAN3D(&m_plan,
                                       m_tfmSize[0], m_tfmSize[1], m_tfmSize[2],
                                       m_dtype.m_tp),
                     "device FFT define plan");
  }

  ~deviceTransform3d()
  {
    FFTX_DEVICE_FFT_DESTROY(m_plan);
  }
  
  deviceTransform3dType<T_IN, T_OUT> m_dtype;

  // fftx::box_t<3> m_inputDomain;
  // fftx::box_t<3> m_outputDomain;
  
  fftx::point_t<3> m_tfmSize;

  FFTX_DEVICE_FFT_HANDLE m_plan;

  FFTX_DEVICE_FFT_RESULT exec(T_IN* a_in, T_OUT* a_out)
  {
    return m_dtype.exec(m_plan, a_in, a_out);
  }
};
  

deviceTransform3dType<std::complex<double>, std::complex<double> >
mddft3dDevice(FFTX_DEVICE_FFT_Z2Z, FFTX_DEVICE_FFT_FORWARD);

deviceTransform3dType<std::complex<double>, std::complex<double> >
imddft3dDevice(FFTX_DEVICE_FFT_Z2Z, FFTX_DEVICE_FFT_INVERSE);

deviceTransform3dType<double, std::complex<double> >
mdprdft3dDevice(FFTX_DEVICE_FFT_D2Z);

deviceTransform3dType<std::complex<double>, double>
imdprdft3dDevice(FFTX_DEVICE_FFT_Z2D);

#endif

template <int DIM, typename T_IN, typename T_OUT>
class TransformFunction
{
public:
  TransformFunction()
  {
    m_name = "";
    m_tp = EMPTY;
  }

  // constructor with FFTX handle
  TransformFunction(fftx::handle_t (*a_functionPtr)
                   (fftx::array_t<DIM, T_IN>&,
                    fftx::array_t<DIM, T_OUT>&),
                    fftx::box_t<DIM> a_inDomain,
                    fftx::box_t<DIM> a_outDomain,
                    fftx::point_t<DIM> a_fullExtents,
                    std::string& a_name,
                    int a_sign)
  {
    m_functionPtr = a_functionPtr;
    m_inDomain = a_inDomain;
    m_outDomain = a_outDomain;
    m_fullExtents = a_fullExtents;
    m_sign = a_sign;
    m_name = a_name;
    m_tp = FFTX_HANDLE;
  }
  
  // constructor with FFTX library transformer
  TransformFunction(fftx::transformer<DIM, T_IN, T_OUT>* a_transformerPtr,
                    int a_sign)
  {
    m_transformerPtr = a_transformerPtr;
    m_inDomain = domainFromSize<DIM>(m_transformerPtr->inputSize());
    m_outDomain = domainFromSize<DIM>(m_transformerPtr->outputSize());
    m_fullExtents = m_transformerPtr->size();
    m_sign = a_sign;
    m_name = m_transformerPtr->name();
    m_tp = FFTX_LIB;
  }
  
  // constructor with FFTXProblem
  TransformFunction(FFTXProblem* a_transformProblemPtr,
                    int a_sign)
  {
    m_transformProblemPtr = a_transformProblemPtr;
    std::vector<int>& sizes = m_transformProblemPtr->sizes;
    fftx::point_t<DIM> domainsize({{sizes[0], sizes[1], sizes[2]}});
    m_fullExtents = domainsize;
    m_name = m_transformProblemPtr->name;
    m_sign = a_sign;
    m_inDomain = inDomainFromSize(m_name, m_fullExtents);
    m_outDomain = outDomainFromSize(m_name, m_fullExtents);
    fftx::OutStream() << "input on " << m_inDomain << ", output on " << m_outDomain << std::endl;
    m_tp = FFTX_PROBLEM;
  }
  
#if defined(FFTX_CUDA) || defined(FFTX_HIP)
  // constructor with device library transformer
  TransformFunction(deviceTransform3dType<T_IN, T_OUT>& a_deviceTfm3dType,
                    fftx::point_t<DIM> a_fullExtents,
                    std::string& a_name,
                    int a_sign)
  {
    m_fullExtents = a_fullExtents;
    m_name = a_name;
    m_sign = a_sign;
    m_inDomain = inDomainFromSize(m_name, m_fullExtents);
    m_outDomain = outDomainFromSize(m_name, m_fullExtents);

    m_deviceTfm3dPtr = new deviceTransform3d<T_IN, T_OUT>(a_deviceTfm3dType,
                                                          m_inDomain,
                                                          m_outDomain);
    m_tp = FFTX_DEVICE_LIB;
  }
#endif
  
  ~TransformFunction()
  {
#if defined(FFTX_CUDA) || defined(FFTX_HIP)
    if (m_tp == FFTX_DEVICE_LIB)
      {
        // FIXME: This destructor gets called 3 times.
        // delete m_deviceTfm3dPtr;
      }
#endif
  }
  virtual bool isDefined()
  { return (m_tp != EMPTY); }

  fftx::box_t<DIM>& inDomain()
  { return m_inDomain; }

  fftx::box_t<DIM>& outDomain()
  { return m_outDomain; }

  int sign()
  { return m_sign; }

  std::string name()
  {
    return m_name;
  }

  fftx::point_t<DIM> size()
  {
    return m_fullExtents;
  }

  virtual void exec(fftx::array_t<DIM, T_IN>& a_inArray,
                    fftx::array_t<DIM, T_OUT>& a_outArray)
  {
    if (m_tp == EMPTY)
      {
        fftx::ErrStream() << "calling exec on empty TransformFunction" << std::endl;
      }
    else if (m_tp == FFTX_HANDLE ||
             m_tp == FFTX_LIB ||
             m_tp == FFTX_PROBLEM ||
             m_tp == FFTX_DEVICE_LIB)
      {
        T_IN* inputHostPtr = a_inArray.m_data.local();
        T_OUT* outputHostPtr = a_outArray.m_data.local();

        // auto sym_pts = m_fullExtents.product();
#if defined(FFTX_CUDA) || defined(FFTX_HIP) || defined(FFTX_SYCL)
        // on GPU
        auto input_pts = m_inDomain.size();
        auto output_pts = m_outDomain.size();

        // T_IN* inputDevicePtr;
        // T_OUT* outputDevicePtr;
        // double* symDevicePtr;

#if defined(FFTX_CUDA) || defined(FFTX_HIP)
        // auto sym_bytes = sym_pts * sizeof(double);
        auto input_bytes = input_pts * sizeof(T_IN);
        auto output_bytes = output_pts * sizeof(T_OUT);

	// 1. Copy input from host to device.
	FFTX_DEVICE_PTR inputDevicePtr;
	FFTX_DEVICE_PTR outputDevicePtr;
	FFTX_DEVICE_PTR symDevicePtr;

        FFTX_DEVICE_MALLOC((void **)&inputDevicePtr, input_bytes);
        FFTX_DEVICE_MALLOC((void **)&outputDevicePtr, output_bytes);
        // FFTX_DEVICE_MALLOC((void **)&symDevicePtr, sym_bytes);
        symDevicePtr = (FFTX_DEVICE_PTR) NULL;
        
        FFTX_DEVICE_MEM_COPY((void*)inputDevicePtr, inputHostPtr, input_bytes,
                        FFTX_MEM_COPY_HOST_TO_DEVICE);
	// 2. Perform the transform.
        if (m_tp == FFTX_DEVICE_LIB)
          {
            FFTX_DEVICE_FFT_CHECK(m_deviceTfm3dPtr->exec((T_IN*)inputDevicePtr,
                                                    (T_OUT*)outputDevicePtr),
                             "device FFT exec launch");
          }
        else
          {
            fftx::array_t<DIM, T_IN> inputDevice(fftx::global_ptr<T_IN>
                                                 ((T_IN*)inputDevicePtr, 0, 1),
                                                 m_inDomain);
            fftx::array_t<DIM, T_OUT> outputDevice(fftx::global_ptr<T_OUT>
                                                   ((T_OUT*)outputDevicePtr, 0, 1),
                                                   m_outDomain);
            if (m_tp == FFTX_HANDLE)
              {          
                (*m_functionPtr)(inputDevice, outputDevice);
              }
            else if (m_tp == FFTX_LIB)
              {
                m_transformerPtr->transform2(inputDevice, outputDevice);
              }
            else if (m_tp == FFTX_PROBLEM)
              {
		// Size and type of m_transformProblemPtr are already set.
#if defined(FFTX_CUDA)
                // std::vector<void*> args{&dY, &dX, &dsym};
		std::vector<void*> args{&outputDevicePtr, &inputDevicePtr, &symDevicePtr};
#elif defined(FFTX_HIP)
                // std::vector<void*> args{dY, dX, dsym};
		std::vector<void*> args{outputDevicePtr, inputDevicePtr, symDevicePtr};
#endif
                m_transformProblemPtr->setArgs(args);
                m_transformProblemPtr->transform();
              }
          }
	// 3. Copy output from device to host.
        FFTX_DEVICE_MEM_COPY(outputHostPtr, (void*)outputDevicePtr, output_bytes,
                        FFTX_MEM_COPY_DEVICE_TO_HOST);

	// 4. Clean up.
        FFTX_DEVICE_FREE((void*)inputDevicePtr);
        FFTX_DEVICE_FREE((void*)outputDevicePtr);
        // FFTX_DEVICE_FREE((void*)symDevicePtr);
#elif defined(FFTX_SYCL)
        if (m_tp == FFTX_HANDLE)
          {          
            (*m_functionPtr)(a_inArray, a_outArray);
          }
        else if (m_tp == FFTX_LIB)
          {
            m_transformerPtr->transform2(a_inArray, a_outArray);
          }
        else if (m_tp == FFTX_PROBLEM)
	  {
	    // 1. Set up buffers.

	    // Doesn't allocate enough space for complex if you do it this way:
	    // sycl::buffer<T_IN> buf_inputPtr(inputHostPtr, input_pts);
	    // sycl::buffer<T_OUT> buf_outputPtr(outputHostPtr, output_pts);
	    auto input_doubles = input_pts * sizeof(T_IN)/sizeof(double);
	    auto output_doubles = output_pts * sizeof(T_OUT)/sizeof(double);
	    sycl::buffer<double> inputBuffer((double *) inputHostPtr, input_doubles);
	    sycl::buffer<double> outputBuffer((double *) outputHostPtr, output_doubles);
	    // double* symHostPtr = new double[sym_pts];
	    // sycl::buffer<double> symBuffer(symHostPtr, sym_pts);
            sycl::buffer<double> symBuffer((double*) NULL, 0); // not needed
	    std::vector<void*> args{(void*)&(outputBuffer), (void*)&(inputBuffer), (void*)&(symBuffer)};

	    // 2. Perform the transform.
	    // Size and type of m_transformProblemPtr are already set.	
	    m_transformProblemPtr->setArgs(args);
	    m_transformProblemPtr->transform();

	    // 3. Clean up.
	    // delete[] symHostPtr;
	  }
#endif
#else
        // on CPU
        if (m_tp == FFTX_HANDLE)
          {          
            (*m_functionPtr)(a_inArray, a_outArray);
          }
        else if (m_tp == FFTX_LIB)
          {
            m_transformerPtr->transform2(a_inArray, a_outArray);
          }
        else if (m_tp == FFTX_PROBLEM)
          {
            double *dX, *dY, *dsym;
            dX = (double *) inputHostPtr;
            dY = (double *) outputHostPtr;
            // dsym = new double[sym_pts];
            dsym = (double*) NULL;
            // dsym = new std::complex<double>[m_fullExtents.product()];
	    // Size and type of m_transformProblemPtr are already set.
            std::vector<void*> args{(void*)dY, (void*)dX, (void*)dsym};
            m_transformProblemPtr->setArgs(args);
            m_transformProblemPtr->transform();
            // delete[] dsym;
          }
#endif
      }
  }

protected:

  enum TransformType { EMPTY = 0, FFTX_HANDLE = 1, FFTX_LIB = 2, FFTX_DEVICE_LIB = 3 , FFTX_PROBLEM = 4 };

  TransformType m_tp;
  fftx::box_t<DIM> m_inDomain;
  fftx::box_t<DIM> m_outDomain;
  int m_sign;
  fftx::point_t<DIM> m_fullExtents;

  std::string m_name;
  
  // case FFTX_HANDLE
  fftx::handle_t (*m_functionPtr) (fftx::array_t<DIM, T_IN>&,
                                  fftx::array_t<DIM, T_OUT>&);
  
  // case FFTX_LIB
  fftx::transformer<DIM, T_IN, T_OUT>* m_transformerPtr;

  // case FFTX_PROBLEM
  FFTXProblem* m_transformProblemPtr;

#if defined(FFTX_CUDA) || defined(FFTX_HIP)
  // case FFTX_DEVICE_LIB
  deviceTransform3d<T_IN, T_OUT>* m_deviceTfm3dPtr;
#endif
};


// Return random real number.
double unifReal()
{
  return unifRealDist(generator);
}

// Return random complex number.
std::complex<double> unifComplex()
{
  return std::complex<double>(unifReal(), unifReal());
}

inline void getUnifScalar(double& a_scalar)
{
  a_scalar = unifReal();
}

inline void getUnifScalar(std::complex<double>& a_scalar)
{
  a_scalar = unifComplex();
}

template<typename T>
inline T unifScalar()
{
  T ret;
  getUnifScalar(ret);
  return ret;
}

// a_scalarIn and a_scalarOut must have the same value.
// If they have the same type, then set them equal to a random of that type.
// If one is real and the other complex, set them to the same random real.
template<typename T_IN, typename T_OUT>
inline void getUnifScalarPair(T_IN& a_scalarIn,
                              T_OUT& a_scalarOut);

template<>
inline void getUnifScalarPair(std::complex<double>& a_scalarIn,
                              std::complex<double>& a_scalarOut)
{
  a_scalarIn = unifComplex();
  a_scalarOut = a_scalarIn;
}

template<>
inline void getUnifScalarPair(double& a_scalarIn,
                              std::complex<double>& a_scalarOut)
{
  a_scalarIn = unifReal();
  a_scalarOut = std::complex<double>(a_scalarIn, 0.);
}

template<>
inline void getUnifScalarPair(std::complex<double>& a_scalarIn,
                              double& a_scalarOut)
{
  a_scalarOut = unifReal();
  a_scalarIn = std::complex<double>(a_scalarOut, 0.);
}


template <int DIM, typename T_IN, typename T_OUT>
class VerifyTransform
{
public:

  VerifyTransform(TransformFunction<DIM, T_IN, T_OUT> a_tfm,
                  int a_rounds,
                  int a_verbosity)
  {
    if (!a_tfm.isDefined())
      {
        fftx::ErrStream() << "transformation not defined" << std::endl;
        return;
      }
    fftx::OutStream() << std::scientific << std::setprecision(5);
    
    m_tfm = a_tfm;
    m_rounds = a_rounds;
    m_verbosity = a_verbosity;

    m_sign = a_tfm.sign();
    m_inDomain = m_tfm.inDomain();
    m_outDomain = m_tfm.outDomain();

    fftx::point_t<DIM> inLo = m_inDomain.lo;
    fftx::point_t<DIM> inHi = m_inDomain.hi;
    for (int d = 0; d < DIM; d++)
      {
        m_unifInt[d] =  std::uniform_int_distribution<int>(inLo[d], inHi[d]);
      }
    // m_unifRealDist = std::uniform_real_distribution<double>(-0.5, 0.5);

    double err = 0.;
    updateMax(err, test1());
    updateMax(err, test2());
    updateMax(err, test3());
    // printf("%dD test on %s in %d rounds max relative error %11.5e\n",
    //        DIM, m_tfm.name().c_str(), m_rounds, err);
    fftx::OutStream() << DIM << "D test on " << m_tfm.name()
                      << " in " << m_rounds
                      << " rounds max relative error " << err
                      << std::endl;
  }

protected:

  enum VerbosityLevel { SHOW_CATEGORIES = 1, SHOW_SUBTESTS = 2, SHOW_ROUNDS = 3};

  TransformFunction<DIM, T_IN, T_OUT> m_tfm;
  
  int m_rounds;
  
  int m_verbosity;

  int m_sign;
  
  fftx::box_t<DIM> m_inDomain;
  fftx::box_t<DIM> m_outDomain;

  // m_unifInt[d] is uniform over the integers in domain.lo[d] : domain.hi[d]
  std::uniform_int_distribution<int> m_unifInt[DIM];

  // Return random point in domain.
  fftx::point_t<DIM> unifPoint()
  {
    fftx::point_t<DIM> ret;
    for (int d = 0; d < DIM; d++)
      {
        ret[d] = m_unifInt[d](generator);
      }
    return ret;
  }

  // Fill a_arr with real numbers distributed uniformly in (-1/2, 1/2).
  void unifRealArray(fftx::array_t<DIM, double>& a_arr)
  {
    forall([](double(&v),
              const fftx::point_t<DIM>& p)
           {
             v = unifReal();
           }, a_arr);
  }

  // Fill a_arr with complex numbers with real and imaginary components distributed uniformly in (-1/2, 1/2).
  void unifComplexArray(fftx::array_t<DIM, std::complex<double>>& a_arr)
  {
    forall([](std::complex<double>(&v),
              const fftx::point_t<DIM>& p)
           {
             v = unifComplex();
           }, a_arr);
  }

  template<typename T>
  void unifArray(fftx::array_t<DIM, T>& a_arr);

  void unifArray(fftx::array_t<DIM, double>& a_arr)
  {
    unifRealArray(a_arr);
  }

  void unifArray(fftx::array_t<DIM, std::complex<double>>& a_arr)
  {
    unifComplexArray(a_arr);
  }

  void setProductWaves(fftx::array_t<DIM, std::complex<double>>& a_arr,
                       const fftx::point_t<DIM>& a_extent,
                       const fftx::point_t<DIM>& a_fixed,
                       int a_sign)
  {
    fftx::point_t<DIM> lo = a_arr.m_domain.lo;
    std::complex<double> omega[DIM];
    for (int d = 0; d < DIM; d++)
      {
        double th = (a_sign*2*(a_fixed[d] - lo[d])) * M_PI / (a_extent[d] * 1.);
        omega[d] = std::complex<double>(cos(th), sin(th));
      }
    /*
    forall([omega, lo](std::complex<double>(&v),
                       const fftx::point_t<DIM>& p)
           {
             v = std::complex<double>(1., 0.);
             for (int d = 0; d < DIM; d++)
               {
                 v *= pow(omega[d], p[d] - lo[d]);
               }
           }, a_arr);
    */
    // Substitute for forall.
    auto dom = a_arr.m_domain;
    auto npts = dom.size();
    auto arrPtr = a_arr.m_data.local();
    for (size_t ind = 0; ind < npts; ind++)
      {
        fftx::point_t<DIM> p = pointFromPositionBox(ind, dom);
        std::complex<double> cval = std::complex<double>(1., 0.);
        for (int d = 0; d < DIM; d++)
          {
            cval *= pow(omega[d], p[d] - lo[d]);
          }
        arrPtr[ind] = cval;
      }
  }
 
  void setRotator(fftx::array_t<DIM, std::complex<double>>& a_arr,
                  const fftx::box_t<DIM>& a_dom,
                  int a_dim,
                  int a_shift)
  {
    fftx::point_t<DIM> lo = a_dom.lo;
    fftx::point_t<DIM> hi = a_dom.hi;
    fftx::point_t<DIM> fixed = lo;
    if (a_shift > 0)
      {
        fixed[a_dim] = lo[a_dim] + a_shift;
      }
    else if (a_shift < 0)
      {
        fixed[a_dim] = hi[a_dim] - (a_shift+1);
      }
    // fftx::OutStream() << "setRotator in " << a_dim << " shift " << a_shift
    // << " waves " << fixed << " of " << a_arr.m_domain << std::endl;
    setProductWaves(a_arr, a_dom.extents(), fixed, -1);
  }


  template<typename T>
  void setUnitImpulse(fftx::array_t<DIM, T>& a_arr,
                      const fftx::point_t<DIM>& a_fixed,
                      T a_scaling = scalarVal<T>(1.) )
  {
    /*
    forall([a_fixed, a_scaling](T(&v),
                                const fftx::point_t<DIM>& p)
           {
             if (p == a_fixed)
               {
                 v = a_scaling;
              }
             else
               {
                 v = scalarVal<T>(0.);
               }
           }, a_arr);
    */
    // Substitute for forall.
    //  forall([](T(&v),
    //            const fftx::point_t<DIM>& p)
    //         {
    //           v = scalarVal<T>(0.);
    //         }, a_arr);
    setConstant(a_arr, scalarVal<T>(0.));
    auto arrPtr = a_arr.m_data.local();
    auto dom = a_arr.m_domain;
    auto indFixed = positionInBox(a_fixed, dom);
    arrPtr[indFixed] = a_scaling;
  }

  
  double test1()
  {
    fftx::array_t<DIM, T_IN> inA(m_inDomain);
    fftx::array_t<DIM, T_IN> inB(m_inDomain);
    fftx::array_t<DIM, T_IN> LCin(m_inDomain);

    fftx::array_t<DIM, T_OUT> outA(m_outDomain);
    fftx::array_t<DIM, T_OUT> outB(m_outDomain);
    fftx::array_t<DIM, T_OUT> LCout(m_outDomain);
    fftx::array_t<DIM, T_OUT> outLCin(m_outDomain);

    double errtest1 = 0.;
    for (int itn = 1; itn <= m_rounds; itn++)
      {
        T_IN alphaIn, betaIn;
        T_OUT alphaOut, betaOut;
        getUnifScalarPair(alphaIn, alphaOut);
        getUnifScalarPair(betaIn, betaOut);
        unifArray(inA);
        unifArray(inB);
        sumArrays(LCin, inA, inB, alphaIn, betaIn);
        m_tfm.exec(inA, outA);
        m_tfm.exec(inB, outB);
        sumArrays(LCout, outA, outB, alphaOut, betaOut);
        m_tfm.exec(LCin, outLCin);
        double err = absMaxRelDiffArray(outLCin, LCout);
        updateMax(errtest1, err);
        if (m_verbosity >= SHOW_ROUNDS)
          {
            // printf("%dD linearity test round %d max relative error %11.5e\n", DIM, itn, err);
            fftx::OutStream() << DIM << "D linearity test round " << itn
                              << " max relative error " << err
                              << std::endl;
          }
      }
    if (m_verbosity >= SHOW_CATEGORIES)
      {        
        // printf("%dD Test 1 (linearity) in %d rounds: max relative error %11.5e\n", DIM, m_rounds, errtest1);
        fftx::OutStream() << DIM << "D Test 1 (linearity) in " << m_rounds
                          << ": max relative error " << errtest1
                          << std::endl;
      }
    return errtest1;
  }


  double test2()
  {
    double errtest2 = 0.;
    updateMax(errtest2, test2impulse1());
    updateMax(errtest2, test2impulsePlus());
    updateMax(errtest2, test2constant());
    updateMax(errtest2, test2constantPlus());
    T_OUT outputVar = scalarVal<T_OUT>(1.);
    updateMax(errtest2, test2impulseRandom(outputVar)); // only if output is complex
    if (m_verbosity >= SHOW_CATEGORIES)
      {
        // printf("%dD Test 2 (impulses) in %d rounds: max relative error %11.5e\n",
        // DIM, m_rounds, errtest2);
        fftx::OutStream() << DIM << "D Test 2 (impulses) in " << m_rounds
                          << " rounds: max relative error " << errtest2
                          << std::endl;
      }
    return errtest2;
  }

  double test2impulse1()
  { // Unit impulse at low corner.
    fftx::array_t<DIM, T_IN> inImpulse(m_inDomain);
    fftx::array_t<DIM, T_OUT> outImpulse(m_outDomain);
    fftx::array_t<DIM, T_OUT> all1out(m_outDomain);
    setUnitImpulse(inImpulse, m_inDomain.lo);
    setConstant(all1out, scalarVal<T_OUT>(1.));
    m_tfm.exec(inImpulse, outImpulse);
    double errtest2impulse1 = absMaxRelDiffArray(outImpulse, all1out);
    if (m_verbosity >= SHOW_SUBTESTS)
      {
        // printf("%dD unit impulse low corner test: max relative error %11.5e\n",
        // DIM, errtest2impulse1);
        fftx::OutStream() << DIM << "D unit impulse low corner test: max relative error "
                          << errtest2impulse1 << std::endl;
      }
    return errtest2impulse1;
  }

  
  double test2impulsePlus()
  { // Unit impulse at low corner.
    fftx::array_t<DIM, T_IN> inImpulse(m_inDomain);
    fftx::array_t<DIM, T_OUT> outImpulse(m_outDomain);
    fftx::array_t<DIM, T_OUT> all1out(m_outDomain);
    setUnitImpulse(inImpulse, m_inDomain.lo);
    setConstant(all1out, scalarVal<T_OUT>(1.));
    m_tfm.exec(inImpulse, outImpulse);

    fftx::array_t<DIM, T_IN> inRand(m_inDomain);
    fftx::array_t<DIM, T_IN> inImpulseMinusRand(m_inDomain);

    fftx::array_t<DIM, T_OUT> outRand(m_outDomain);
    fftx::array_t<DIM, T_OUT> outImpulseMinusRand(m_outDomain);
    fftx::array_t<DIM, T_OUT> mysum(m_outDomain);
  
    // Check that for random arrays inRand,
    // fft(inRand) + fft(inImpulse - inRand) = fft(inImpulse) = all1out.
    double errtest2impulsePlus = 0.;
    for (int itn = 1; itn <= m_rounds; itn++)
      {
        unifArray(inRand);
        m_tfm.exec(inRand, outRand);
        diffArrays(inImpulseMinusRand, inImpulse, inRand);
        m_tfm.exec(inImpulseMinusRand, outImpulseMinusRand);
        sumArrays(mysum, outRand, outImpulseMinusRand);
        double err = absMaxRelDiffArray(mysum, all1out);
        updateMax(errtest2impulsePlus, err);
        if (m_verbosity >= SHOW_ROUNDS)
          {
            // printf("%dD random + unit impulse low corner test round %d max relative error %11.5e\n", DIM, itn, err);
            fftx::OutStream() << DIM << "D random + unit impulse low corner test round " << itn
                              << " max relative error " << err
                              << std::endl;
          }
      }

    if (m_verbosity >= SHOW_SUBTESTS)
      {
        // printf("%dD unit impulse low corner test in %d rounds: max relative error %11.5e\n",
        // DIM, m_rounds, errtest2impulsePlus);
        fftx::OutStream() << DIM << "D unit impulse low corner test in " << m_rounds
                          << ": max relative error " << errtest2impulsePlus
                          << std::endl;
      }
    return errtest2impulsePlus;
  }

  
  double test2constant()
  { // Check that constant maps back to unit impulse at low corner.
    fftx::array_t<DIM, T_IN> all1in(m_inDomain);
    setConstant(all1in, scalarVal<T_IN>(1.));

    fftx::array_t<DIM, T_OUT> magImpulse(m_outDomain);
    auto npts = m_tfm.size().product();
    T_OUT mag = scalarVal<T_OUT>(npts * 1.);
    setUnitImpulse(magImpulse, m_outDomain.lo, mag);

    fftx::array_t<DIM, T_OUT> outImpulse(m_outDomain);
    m_tfm.exec(all1in, outImpulse);

    double errtest2constant = absMaxRelDiffArray(outImpulse, magImpulse);
    if (m_verbosity >= SHOW_SUBTESTS)
    {
      // printf("%dD constant test: max relative error %11.5e\n",
      // DIM, errtest2constant);
      fftx::OutStream() << DIM << "D constant test: max relative error "
                        << errtest2constant << std::endl;
    }
    return errtest2constant;
  }

  double test2constantPlus()
  {
    fftx::array_t<DIM, T_IN> all1in(m_inDomain);
    setConstant(all1in, scalarVal<T_IN>(1.));

    fftx::array_t<DIM, T_OUT> magImpulse(m_outDomain);
    auto npts = m_tfm.size().product();
    T_OUT mag = scalarVal<T_OUT>(npts * 1.);
    setUnitImpulse(magImpulse, m_outDomain.lo, mag);

    fftx::array_t<DIM, T_IN> inRand(m_inDomain);
    fftx::array_t<DIM, T_IN> inConstantMinusRand(m_inDomain);

    fftx::array_t<DIM, T_OUT> outRand(m_outDomain);
    fftx::array_t<DIM, T_OUT> outConstantMinusRand(m_outDomain);
    fftx::array_t<DIM, T_OUT> outSum(m_outDomain);

    // Check that for random arrays inRand,
    // fft(inRand) + fft(all1 - inRand) = fft(all1) = magImpulse.
    double errtest2constantPlus = 0.;
    for (int itn = 1; itn <= m_rounds; itn++)
      {
        unifArray(inRand);
        m_tfm.exec(inRand, outRand);

        diffArrays(inConstantMinusRand, all1in, inRand);
        m_tfm.exec(inConstantMinusRand, outConstantMinusRand);

        sumArrays(outSum, outRand, outConstantMinusRand);
      
        double err = absMaxRelDiffArray(outSum, magImpulse);
        updateMax(errtest2constantPlus, err);
        if (m_verbosity >= SHOW_ROUNDS)
          {
            // printf("%dD random + constant test round %d max relative error %11.5e\n",
            // DIM, itn, err);
            fftx::OutStream() << DIM << "D random + constant test round " << itn
                              << " max relative error " << err
                              << std::endl;
          }
      }

    if (m_verbosity >= SHOW_SUBTESTS)
      {
        // printf("%dD random + constant test in %d rounds: max relative error %11.5e\n",
        // DIM, m_rounds, errtest2constantPlus);
        fftx::OutStream() << DIM << "D random + constant test in " << m_rounds
                          << ": max relative error " << errtest2constantPlus
                          << std::endl;
      }
    return errtest2constantPlus;
  }

  double test2impulseRandom(double a_outputVar)
  {
    // Do nothing if output is real. Run this test only if output is complex.
    return 0.;
  }

  double test2impulseRandom(std::complex<double> a_outputVar)
  {
    // Check unit impulse at random position.
    fftx::array_t<DIM, T_IN> inImpulse(m_inDomain);
    fftx::array_t<DIM, std::complex<double>> outImpulse(m_outDomain);
    fftx::array_t<DIM, std::complex<double>> outCheck(m_outDomain);
    double errtest2impulseRandom = 0.;
    // fftx::point_t<DIM> fullExtents = m_tfm.size();
    fftx::point_t<DIM> fullExtents = m_inDomain.extents();
    for (int itn = 1; itn <= m_rounds; itn++)
      {
        fftx::point_t<DIM> rpoint = unifPoint();
        setUnitImpulse(inImpulse, rpoint);
        m_tfm.exec(inImpulse, outImpulse);
        // Recall m_inDomain is whole domain,
        // but m_outDomain may be truncated;
        // waves defined on m_outDomain,
        // but based on the full m_inDomain extents.
        setProductWaves(outCheck, fullExtents, rpoint, m_sign);
        double err = absMaxRelDiffArray(outImpulse, outCheck);
        updateMax(errtest2impulseRandom, err);
        if (m_verbosity >= SHOW_ROUNDS)
          {
            // printf("%dD random impulse test round %d max relative error %11.5e\n",
            // DIM, itn, err);
            fftx::OutStream() << DIM << "D random impulse test round " << itn
                              << " max relative error " << err
                              << std::endl;
          }
      }

    if (m_verbosity >= SHOW_SUBTESTS)
      {
        // printf("%dD random impulse in %d rounds: max relative error %11.5e\n",
        // DIM, m_rounds, errtest2impulseRandom);
        fftx::OutStream() << DIM << "D random impulse in " << m_rounds
                          << ": max relative error " << errtest2impulseRandom
                          << std::endl;
      }
    return errtest2impulseRandom;
  }
    
  double test3()
  {
    double errtest3 = 0.;
    T_OUT outputVar = scalarVal<T_OUT>(1.);
    updateMax(errtest3, test3time(outputVar)); // only if output is complex
    T_IN inputVar = scalarVal<T_IN>(1.);
    updateMax(errtest3, test3frequency(inputVar)); // only if input is complex
    if (m_verbosity >= SHOW_CATEGORIES)
      {
        // printf("%dD Test 3 (shifts) in %d rounds: max relative error %11.5e\n",
        // DIM, m_rounds, errtest3);
        fftx::OutStream() << DIM << "D Test 3 (shifts) in " << m_rounds
                          << ": max relative error " << errtest3
                          << std::endl;
      }
    return errtest3;
  }

  double test3time(double outputVar)
  {
    // Do nothing if output is real. Run this test only if output is complex.
    return 0.;
  }

  double test3time(std::complex<double> outputVar)
  {
    fftx::array_t<DIM, T_IN> inRand(m_inDomain);
    fftx::array_t<DIM, T_IN> inRandRot(m_inDomain);
    fftx::array_t<DIM, std::complex<double>> outRand(m_outDomain);
    fftx::array_t<DIM, std::complex<double>> outRandRot(m_outDomain);
    fftx::array_t<DIM, std::complex<double>> rotator(m_outDomain);
    fftx::array_t<DIM, std::complex<double>> outRandRotMult(m_outDomain);
    double errtest3timeDim[DIM];
    double errtest3time = 0.;
    for (int d = 0; d < DIM; d++)
      {
        errtest3timeDim[d] = 0.;
        setRotator(rotator, m_inDomain, d, -m_sign); // +1 for MDDFT, -1 for IMDDFT, -1 for PRDFT
        for (int itn = 1; itn <= m_rounds; itn++)
          {
            unifArray(inRand);
            // time-shift test in dimension d
            rotate(inRandRot, inRand, d, 1); // +1 for MDDFT, +1 for IMDDFT, +1 for PRDFT
            m_tfm.exec(inRand, outRand);
            m_tfm.exec(inRandRot, outRandRot);
            productArrays(outRandRotMult, outRandRot, rotator);
            double err = absMaxRelDiffArray(outRandRotMult, outRand);
            updateMax(errtest3timeDim[d], err);
            updateMax(errtest3time, errtest3timeDim[d]);
            if (m_verbosity >= SHOW_ROUNDS)
              {
                // printf("%dD dim %d time-shift test %d max relative error %11.5e\n",
                // DIM, d, itn, err);
                fftx::OutStream() << DIM << "D dim " << d << " time-shift test "
                                  << itn << " max relative error " << err
                                  << std::endl;
              }
          }
        if (m_verbosity >= SHOW_SUBTESTS)
          {
            // printf("%dD dim %d time-shift test in %d rounds: max relative error %11.5e\n",
            // DIM, d, m_rounds, errtest3timeDim[d]);
            fftx::OutStream() << DIM << "D dim " << d << " time-shift test in "
                              << m_rounds << " rounds: max relative error "
                              << errtest3timeDim[d] << std::endl;
          }
      }
    return errtest3time;
  }



  double test3frequency(double a_inVar)
  { // Do nothing if input is real. Run this test only if input is complex.
    return 0.;
  }

  double test3frequency(std::complex<double>& a_inVar)
  {
    fftx::array_t<DIM, std::complex<double>> inRand(m_inDomain);
    fftx::array_t<DIM, std::complex<double>> inRandMult(m_inDomain);
    fftx::array_t<DIM, T_OUT> outRand(m_outDomain);
    fftx::array_t<DIM, T_OUT> outRandMult(m_outDomain);
    fftx::array_t<DIM, std::complex<double>> rotatorUp(m_inDomain);
    fftx::array_t<DIM, T_OUT> outRandMultRot(m_outDomain);
    double errtest3frequencyDim[DIM];
    double errtest3frequency = 0.;
    for (int d = 0; d < DIM; d++)
      {
        // frequency-shift test in dimension d
        errtest3frequencyDim[d] = 0.;
        // Recall m_outDomain is whole domain,
        // but m_inDomain may be truncated;
        // rotatorUp is defined on m_inDomain,
        // but based on full m_outDomain.
        setRotator(rotatorUp, m_outDomain, d, 1);
        for (int itn = 1; itn <= m_rounds; itn++)
          {
            unifComplexArray(inRand);
            productArrays(inRandMult, inRand, rotatorUp);
            m_tfm.exec(inRand, outRand);
            m_tfm.exec(inRandMult, outRandMult);
            rotate(outRandMultRot, outRandMult, d, m_sign);
            double err = absMaxRelDiffArray(outRandMultRot, outRand);
            updateMax(errtest3frequencyDim[d], err);
            updateMax(errtest3frequency, errtest3frequencyDim[d]);
            if (m_verbosity >= SHOW_ROUNDS)
              {
                // printf("%dD dim %d frequency-shift test %d max relative error %11.5e\n", DIM, d, itn, err);
                fftx::OutStream() << DIM << "D dim " << d
                                  << " frequency-shift test " << itn
                                  << " max relative error " << err
                                  << std::endl;
              }
          }
        if (m_verbosity >= SHOW_SUBTESTS)
          {
            // printf("%dD dim %d frequency-shift test in %d rounds: max relative error %11.5e\n",
            // DIM, d, m_rounds, errtest3frequencyDim[d]);
            fftx::OutStream() << DIM << "D dim " << d
                              << " frequency-shift test in " << m_rounds
                              << " rounds: max relative error "
                              << errtest3frequencyDim[d]
                              << std::endl;
          }
      }
    return errtest3frequency;
  }
  
};


#endif
