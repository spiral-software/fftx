#ifndef FFTX_UTILITIES_HEADER
#define FFTX_UTILITIES_HEADER

template<int DIM>
inline fftx::point_t<DIM> Zero()
{
  fftx::point_t<DIM> rtn;
  for(int i=0; i<DIM; i++) rtn.x[i]=0;
  return rtn;
}

inline void getScalarVal(double& a_val,
                         double a_in)
{
  a_val = a_in;
}

inline void getScalarVal(std::complex<double>& a_val,
                         double a_in)
{
  a_val =std::complex<double>(a_in, 0.);
}

template<typename T>
inline T scalarVal(double a_in)
{
  T ret;
  getScalarVal(ret, a_in);
  return ret;
}

template<int DIM> inline
fftx::point_t<DIM> shiftInBox(fftx::point_t<DIM> a_pt,
                              fftx::point_t<DIM> a_shift,
                              fftx::box_t<DIM> a_domain)
{
  fftx::point_t<DIM> ptShift;
  for (int d = 0; d < DIM; d++)
    {
      ptShift[d] = a_pt[d] + a_shift[d];
      if (ptShift[d] > a_domain.hi[d])
        {
          ptShift[d] -= a_domain.extents()[d];
        }
      else if (ptShift[d] < a_domain.lo[d])
        {
          ptShift[d] += a_domain.extents()[d];
        }
    }
  return ptShift;
}

// Set a_arrOut = a_arrIn pointwise.
template<int DIM, typename T>
void copyArray(fftx::array_t<DIM, T>& a_arrOut,
               const fftx::array_t<DIM, T>& a_arrIn)
{
  auto arrInPtr = a_arrIn.m_data.local();
  auto arrOutPtr = a_arrOut.m_data.local();
  auto dom = a_arrOut.m_domain;
  size_t npts = dom.size();
  for (size_t ind = 0; ind < npts; ind++)
    {
      arrOutPtr[ind] = arrInPtr[ind];
    }
}

// Set a_arr += a_scaling * a_multiplier pointwise.
template<int DIM, typename T>
void addArray(fftx::array_t<DIM, T>& a_arr,
              const fftx::array_t<DIM, T>& a_summand,
              T a_scalingSummand = scalarVal<T>(1.),
              T a_scalingOrig = scalarVal<T>(1.))
{
  auto arrPtr = a_arr.m_data.local();
  auto summandPtr = a_summand.m_data.local();
  auto dom = a_arr.m_domain;
  size_t npts = dom.size();
  for (size_t ind = 0; ind < npts; ind++)
    {
      arrPtr[ind] *= a_scalingOrig;
      arrPtr[ind] += a_scalingSummand * summandPtr[ind];
    }
}

// Set a_arr *= a_multiplier pointwise.
template<int DIM, typename T>
void multiplyByArray(fftx::array_t<DIM, T>& a_arr,
                     const fftx::array_t<DIM, T>& a_multiplier)
{
  auto arrPtr = a_arr.m_data.local();
  auto multiplierPtr = a_multiplier.m_data.local();
  auto dom = a_arr.m_domain;
  size_t npts = dom.size();
  for (size_t ind = 0; ind < npts; ind++)
    {
      arrPtr[ind] *= multiplierPtr[ind];
    }
}

// Set a_sum = a_scaling1 * a_arr1 + a_scaling2 * a_arr2 pointwise.
template<int DIM, typename T>
void sumArrays(fftx::array_t<DIM, T>& a_sum,
               const fftx::array_t<DIM, T>& a_arr1,
               const fftx::array_t<DIM, T>& a_arr2,
               T a_scaling1 = scalarVal<T>(1.),
               T a_scaling2 = scalarVal<T>(1.))
               
{
  assert(a_sum.m_domain == a_arr1.m_domain);
  assert(a_sum.m_domain == a_arr2.m_domain);
  copyArray(a_sum, a_arr1);
  addArray(a_sum, a_arr2, a_scaling2, a_scaling1);
}

// Set a_diff = a_arr1 - a_arr2 pointwise.
template<int DIM, typename T>
void diffArrays(fftx::array_t<DIM, T>& a_diff,
                const fftx::array_t<DIM, T>& a_arr1,
                const fftx::array_t<DIM, T>& a_arr2)
               
{
  sumArrays(a_diff, a_arr1, a_arr2, scalarVal<T>(1.), scalarVal<T>(-1.));
}

// Set a_prod = a_arr1 * a_arr2 pointwise.
template<int DIM, typename T>
void productArrays(fftx::array_t<DIM, T>& a_prod,
                   const fftx::array_t<DIM, T>& a_arr1,
                   const fftx::array_t<DIM, T>& a_arr2)
{
  assert(a_prod.m_domain == a_arr1.m_domain);
  assert(a_prod.m_domain == a_arr2.m_domain);
  copyArray(a_prod, a_arr1);
  multiplyByArray(a_prod, a_arr2);
}

// Set a_arr to constant a_val.
template<int DIM, typename T>
void setConstant(fftx::array_t<DIM, T>& a_arr,
                 const T& a_val)
{
  forall([a_val](T(&v),
                 const fftx::point_t<DIM>& p)
         {
           v = a_val;
         }, a_arr);
}

// Set every element of a_arr to its complex conjugate.
template<int DIM>
void conjugateArray(fftx::array_t<DIM, std::complex<double>>& a_arr)
{
  forall([](std::complex<double>(&v),
            const fftx::point_t<DIM>& p)
         {
           v = std::conj(v);
         }, a_arr);
}

template<int DIM, typename T>
void writeArray(fftx::array_t<DIM, T>& a_arr)
{
  auto dom = a_arr.m_domain;
  auto arrPtr = a_arr.m_data.local();
  size_t npts = dom.size();
  for (size_t ind = 0; ind < npts; ind++)
    {
      std::cout << ind << "  " << pointFromPositionBox(ind, dom)
                << "  " << arrPtr[ind] << std::endl;
    }
}

inline void updateMax(double& a_max,
                      double a_here)
{
  if (a_here > a_max)
    {
      a_max = a_here;
    }
}

template<typename T>
inline void updateMaxAbs(double& a_max,
                         const T& a_here)
{
  double absHere = std::abs(a_here);
  if (absHere > a_max)
    {
      a_max = absHere;
    }
}

// Return max(abs(a_arr)).
template<int DIM, typename T>
double absMaxArray(fftx::array_t<DIM, T>& a_arr)
{
  auto dom = a_arr.m_domain;
  auto arrPtr = a_arr.m_data.local();
  size_t npts = dom.size();
  double absMax = 0.;
  forall([&absMax](T(&v),
                  const fftx::point_t<DIM>& p)
         {
           updateMaxAbs(absMax, v);
         }, a_arr);
  return absMax;
}

// Return max(abs(a_arr1 - a_arr2)).
template<int DIM, typename T>
double absMaxDiffArray(fftx::array_t<DIM, T>& a_arr1,
                       fftx::array_t<DIM, T>& a_arr2)
{
  auto dom = a_arr1.m_domain;
  assert(dom == a_arr2.m_domain);
  auto arr1Ptr = a_arr1.m_data.local();
  auto arr2Ptr = a_arr2.m_data.local();
  size_t npts = dom.size();
  double absDiffMax = 0.;
  for (size_t ind = 0; ind < npts; ind++)
    {
      T diffHere = arr1Ptr[ind] - arr2Ptr[ind];
      updateMaxAbs(absDiffMax, diffHere);
    }
  return absDiffMax;
}

template<int DIM, typename T>
void rotate(fftx::array_t<DIM, T>& a_arrOut,
            const fftx::array_t<DIM, T>& a_arrIn,
            int a_dim,
            int a_shift)
{
  auto dom = a_arrIn.m_domain;
  assert(a_arrOut.m_domain == dom);
  fftx::point_t<DIM> shift;
  for (int d = 0; d < DIM; d++)
    {
      shift[d] = 0;
    }
  shift[a_dim] = a_shift;
  auto inPtr = a_arrIn.m_data.local();
  /*
  forall([inPtr, shift, dom](T(&v),
                             const fftx::point_t<DIM>& p)
         {
           fftx::point_t<DIM> pRot = shiftInBox(p, shift, dom);
           size_t indRot = positionInBox(pRot, dom);
           v = inPtr[indRot];
           }, a_arrOut);
  */
  // Substitute for forall.
  auto npts = dom.size();
  auto outPtr = a_arrOut.m_data.local();
  for (size_t ind = 0; ind < npts; ind++)
    {
      fftx::point_t<DIM> p = pointFromPositionBox(ind, dom);
      fftx::point_t<DIM> pRot = shiftInBox(p, shift, dom);
      size_t indRot = positionInBox(pRot, dom);
      outPtr[ind] = inPtr[indRot];
    }
}

// Set 2nd-order discrete laplacian of periodic array.
template<int DIM, typename T>
void laplacian2periodic(fftx::array_t<DIM, T>& a_laplacian,
                        const fftx::array_t<DIM, T>& a_arr)
{
  auto dom = a_arr.m_domain;
  auto inPtr = a_arr.m_data.local();
  /*
  forall([inPtr, dom](T(&laplacianElem),
                      const T(&inElem),
                      const fftx::point_t<DIM>& p)
         {
           laplacianElem = scalarVal<T>(0.);
           for (int d = 0; d < DIM; d++)
             {
               for (int sgn = -1; sgn <= 1; sgn += 2)
                 {p
                   fftx::point_t<DIM> shift = Zero<DIM>();
                   shift[d] = sgn;
                   fftx::point_t<DIM> pShift = shiftInBox(p, shift, dom);
                   size_t indShift = positionInBox(pShift, dom);
                   laplacianElem += inPtr[indShift] - inElem;
                 }
             }
         }, a_laplacian, a_arr);
  */
  // Substitute for forall.
  auto arrPtr = a_arr.m_data.local();
  auto laplacianPtr = a_laplacian.m_data.local();
  auto npts = dom.size();
  for (size_t ind = 0; ind < npts; ind++)
    {
      auto p = pointFromPositionBox(ind, dom);
      auto arrElem = arrPtr[ind];
      auto laplacianElem = scalarVal<T>(0.);
      for (int d = 0; d < DIM; d++)
        {
          for (int sgn = -1; sgn <= 1; sgn += 2)
            {
              fftx::point_t<DIM> shift = Zero<DIM>();
              shift[d] = sgn;
              fftx::point_t<DIM> pShift = shiftInBox(p, shift, dom);
              size_t indShift = positionInBox(pShift, dom);
              laplacianElem += arrPtr[indShift] - arrElem;
            }
        }
      laplacianPtr[ind] = laplacianElem;
    }
}

inline int sym_index(int i, int lo, int hi)
{ // index of i in array lo:hi that should be conjugate if Hermitian symmetry
  int ret = i;
  if (! ((i == lo) || (2*i == lo + hi + 1)) )
    {
      ret = lo + hi + 1 - i;
    }
  return ret;
}

template<int DIM>
fftx::point_t<DIM> sym_point(fftx::point_t<DIM> a_pt,
                             fftx::box_t<DIM> a_bx)
{
  fftx::point_t<DIM> ptRef;
  for (int d = 0; d < DIM; d++)
    {
      ptRef[d] = sym_index(a_pt[d], a_bx.lo[d], a_bx.hi[d]);
    }
  return ptRef;
}

template<int DIM, typename T_IN, typename T_OUT>
void symmetrizeHermitian(fftx::array_t<DIM, T_IN>& a_arrIn,
                         fftx::array_t<DIM, T_OUT>& a_arrOut);


template<int DIM>
void symmetrizeHermitian(fftx::array_t<DIM, double>& a_arrIn,
                         fftx::array_t<DIM, double>& a_arrOut)
{ };

template<int DIM>
void symmetrizeHermitian(fftx::array_t<DIM, double>& a_arrIn,
                         fftx::array_t<DIM, std::complex<double>>& a_arrOut)
{ };

template<int DIM>
void symmetrizeHermitian(fftx::array_t<DIM, std::complex<double>>& a_arrIn,
                         fftx::array_t<DIM, std::complex<double>>& a_arrOut)
{ };

template<int DIM>
void symmetrizeHermitian(fftx::array_t<DIM, std::complex<double> >& a_arrIn,
                         fftx::array_t<DIM, double>& a_arrOut)
{
  fftx::box_t<DIM> inputDomain = a_arrIn.m_domain;
  std::complex<double>* arrPtr = a_arrIn.m_data.local();

  fftx::box_t<DIM> outputDomain = a_arrOut.m_domain;
  fftx::point_t<DIM> lo = outputDomain.lo;
  // fftx::point_t<DIM> hi = outputDomain.hi;
  fftx::point_t<DIM> extent = outputDomain.extents();

  auto npts = inputDomain.size();
  for (size_t ind = 0; ind < npts; ind++)
    {
      fftx::point_t<DIM> pt = pointFromPositionBox(ind, inputDomain);

      // If indices of pt are all at either low or (if extent even) middle,
      // then array element must be real.
      bool mustBeReal = true;
      for (int d = 0; d < DIM; d++)
        {
          mustBeReal = mustBeReal &&
            ((pt[d] == lo[d]) || (2*pt[d] == 2*lo[d] + extent[d]));
        }
      if (mustBeReal)
        {
          arrPtr[ind].imag(0.);
        }
      else
        {
          // If pt is outside lower octant, set array element to 
          // conjugate of a mapped point in lower octant,
          // if that point is in the array domain.
          bool someUpperDim = false;
          for (int d = 0; d < DIM; d++)
            {
              someUpperDim = someUpperDim ||
                (2*pt[d] >= 2*lo[d] + extent[d]);
            }
          if (someUpperDim)
            {
              fftx::point_t<DIM> ptRef = sym_point(pt, outputDomain);
              if (isInBox(ptRef, inputDomain))
                {
                  size_t indRef = positionInBox(ptRef, inputDomain);
                  arrPtr[ind] = std::conj(arrPtr[indRef]);
                }
              // If ptRef is not in inputDomain,
              // then you don't need to worry about setting pt.
            }
        }
    }
}

template<int DIM, typename T_IN, typename T_OUT>
bool checkSymmetryHermitian(fftx::array_t<DIM, T_IN>& a_arrIn,
                            fftx::array_t<DIM, T_OUT>& a_arrOut);


template<int DIM>
bool checkSymmetryHermitian(fftx::array_t<DIM, double>& a_arrIn,
                            fftx::array_t<DIM, std::complex<double>>& a_arrOut)
{
  return true;
}

template<int DIM>
bool checkSymmetryHermitian(fftx::array_t<DIM, std::complex<double>>& a_arrIn,
                            fftx::array_t<DIM, std::complex<double>>& a_arrOut)
{
  return true;
}

template<int DIM>
bool checkSymmetryHermitian
(fftx::array_t<DIM, std::complex<double> >& a_arrIn,
 fftx::array_t<DIM, double>& a_arrOut)
{
  fftx::box_t<DIM> inputDomain = a_arrIn.m_domain;
  std::complex<double>* arrPtr = a_arrIn.m_data.local();

  fftx::box_t<DIM> outputDomain = a_arrOut.m_domain;
  fftx::point_t<DIM> lo = outputDomain.lo;
  fftx::point_t<DIM> hi = outputDomain.hi;

  auto npts = inputDomain.size();
  bool is_symmetric = true;
  for (size_t ind = 0; ind < npts; ind++)
    {
      fftx::point_t<DIM> pt = pointFromPositionBox(ind, inputDomain);
      fftx::point_t<DIM> ptRef;
      for (int d = 0; d < DIM; d++)
        {
          ptRef[d] = sym_index(pt[d], lo[d], hi[d]);
        }
      if (isInBox(ptRef, inputDomain))
        {
          size_t indRef = positionInBox(ptRef, inputDomain);
          if (arrPtr[indRef] != std::conj(arrPtr[ind]))
            {
              is_symmetric = false;
              break;
            }
        }
      // If ptRef is not in inputDomain, then no symmetry to check for pt.
    }
  return is_symmetric;
}

template<int DIM>
void fillSymmetric(fftx::array_t<DIM, std::complex<double> >& a_arrOut,
                   fftx::array_t<DIM, std::complex<double> >& a_arrIn)
{
  std::cout << "*** symmetrizing C2R" << std::endl;
  fftx::box_t<DIM> inputDomain = a_arrIn.m_domain;
  std::complex<double>* arrInPtr = a_arrIn.m_data.local();

  fftx::box_t<DIM> outputDomain = a_arrOut.m_domain;
  std::complex<double>* arrOutPtr = a_arrOut.m_data.local();

  fftx::point_t<DIM> lo = outputDomain.lo;
  fftx::point_t<DIM> hi = outputDomain.hi;

  auto nptsOut = outputDomain.size();
  for (size_t indOut = 0; indOut < nptsOut; indOut++)
    {
      fftx::point_t<DIM> ptOut = pointFromPositionBox(indOut, outputDomain);
      if (isInBox(ptOut, inputDomain))
        {
          size_t indIn = positionInBox(ptOut, inputDomain);
          arrOutPtr[indOut] = arrInPtr[indIn];
        }
      else
        { // Find reflected point, and set arrOutPtr[ind] to conjugate.
          fftx::point_t<DIM> ptRefIn = sym_point(ptOut, outputDomain);
          if (isInBox(ptRefIn, inputDomain))
            {
              size_t indRefIn = positionInBox(ptRefIn, inputDomain);
              arrOutPtr[indOut] = std::conj(arrInPtr[indRefIn]);
            }
          else
            {
              std::cout << "fillSymmetric: " << ptOut << " from " << ptRefIn
                        << " which is not in input domain" << std::endl;
            }
        }
    }
}

#endif /*  end include guard FFTX_UTILITIES_HEADER */
