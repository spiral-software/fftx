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
  forall([](T(&arrOutElem),
            const T(&arrInElem),
            const fftx::point_t<DIM>& p)
         {
           arrOutElem = arrInElem;
         }, a_arrOut, a_arrIn);
}

// Set a_arr += a_scaling * a_multiplier pointwise.
template<int DIM, typename T>
void addArray(fftx::array_t<DIM, T>& a_arr,
              const fftx::array_t<DIM, T>& a_summand,
              T a_scalingSummand = scalarVal<T>(1.),
              T a_scalingOrig = scalarVal<T>(1.))
{
  forall([a_scalingSummand, a_scalingOrig](T(&arrElem),
            const T(&summandElem),
            const fftx::point_t<DIM>& p)
         {
           arrElem *= a_scalingOrig;
           arrElem += a_scalingSummand * summandElem;
         }, a_arr, a_summand);
}

// Set a_arr *= a_multiplier pointwise.
template<int DIM, typename T>
void multiplyByArray(fftx::array_t<DIM, T>& a_arr,
                     const fftx::array_t<DIM, T>& a_multiplier)
{
  forall([](T(&arrElem),
            const T(&multiplierElem),
            const fftx::point_t<DIM>& p)
         {
           arrElem *= multiplierElem;
         }, a_arr, a_multiplier);
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
  auto dom = a_prod.m_domain;
  assert(dom == a_arr1.m_domain);
  assert(dom == a_arr2.m_domain);
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
  fftx::array_t<DIM, T> arrDiff(dom);
  diffArrays(arrDiff, a_arr1, a_arr2);
  double absDiffMax = absMaxArray(arrDiff);
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
  forall([inPtr, shift, dom](T(&v),
                             const fftx::point_t<DIM>& p)
         {
           fftx::point_t<DIM> pRot = shiftInBox(p, shift, dom);
           size_t indRot = positionInBox(pRot, dom);
           v = inPtr[indRot];
         }, a_arrOut);
}

// Set 2nd-order discrete laplacian of periodic array.
template<int DIM, typename T>
void laplacian2periodic(fftx::array_t<DIM, T>& a_laplacian,
                        const fftx::array_t<DIM, T>& a_arr)
{
  auto dom = a_arr.m_domain;
  auto inPtr = a_arr.m_data.local();
  forall([inPtr, dom](T(&laplacianElem),
                      const T(&inElem),
                      const fftx::point_t<DIM>& p)
         {
           laplacianElem = scalarVal<T>(0.);
           for (int d = 0; d < DIM; d++)
             {
               for (int sgn = -1; sgn <= 1; sgn += 2)
                 {
                   fftx::point_t<DIM> shift = Zero<DIM>();
                   shift[d] = sgn;
                   fftx::point_t<DIM> pShift = shiftInBox(p, shift, dom);
                   size_t indShift = positionInBox(pShift, dom);
                   laplacianElem += inPtr[indShift] - inElem;
                 }
             }
         }, a_laplacian, a_arr);
}


