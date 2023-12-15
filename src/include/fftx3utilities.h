#ifndef FFTX_UTILITIES_HEADER
#define FFTX_UTILITIES_HEADER

/** \relates fftx::box_t
    Returns a box having the dimensions of the first argument,
    with its lower corner in each direction being 1
    plus the coordinate in that direction in the optional second argument.
 */
template<int DIM>
fftx::box_t<DIM> domainFromSize(const fftx::point_t<DIM>& a_size,
                                const fftx::point_t<DIM>& a_offset = fftx::point_t<DIM>::Zero())
{
  fftx::box_t<DIM> bx;
  for (int d = 0; d < DIM; d++)
    {
      bx.lo[d] = a_offset[d] + 1;
      bx.hi[d] = a_offset[d] + a_size[d];
    }
  return bx;
}

/** \internal */
inline void getScalarVal(double& a_val,
                         double a_in)
{
  a_val = a_in;
}

/** \internal */
inline void getScalarVal(std::complex<double>& a_val,
                         double a_in)
{
  a_val =std::complex<double>(a_in, 0.);
}

/** \internal */
template<typename T>
inline T scalarVal(double a_in)
{
  T ret;
  getScalarVal(ret, a_in);
  return ret;
}

/** \relates fftx::box_t
    Returns a shifted point in a periodic domain.

    The returned point is the input point <tt>a_pt</tt>
    shifted by <tt>a_shift</tt>
    and wrapped around if necessary to fit in <tt>a_domain</tt>.
 */
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

/** \relates fftx::array_t
   Sets the contents of the first array to the contents of the second array.

   Both arrays <tt>a_arrIn</tt> and <tt>a_arrOut</tt> must be
   defined on the same domain.
*/
template<int DIM, typename T>
void copyArray(fftx::array_t<DIM, T>& a_arrOut,
               const fftx::array_t<DIM, T>& a_arrIn)
{
  assert(a_arrIn.m_domain == a_arrOut.m_domain);
  auto arrInPtr = a_arrIn.m_data.local();
  auto arrOutPtr = a_arrOut.m_data.local();
  auto dom = a_arrOut.m_domain;
  size_t npts = dom.size();
  for (size_t ind = 0; ind < npts; ind++)
    {
      arrOutPtr[ind] = arrInPtr[ind];
    }
}

/** \relates fftx::array_t
    Increments the contents of the first array by
    the contents of the second array,
    optionally with scalar multiplication of each array:<br>
    Set <tt>a_arr = a_scalingOrig * a_arr + a_scalingSummand * a_summand</tt>
    at each point.

    Both arrays <tt>a_arr</tt> and <tt>a_summand</tt>
    must be defined on the same domain.
*/
template<int DIM, typename T>
void addArray(fftx::array_t<DIM, T>& a_arr,
              const fftx::array_t<DIM, T>& a_summand,
              T a_scalingSummand = scalarVal<T>(1.),
              T a_scalingOrig = scalarVal<T>(1.))
{
  assert(a_arr.m_domain == a_summand.m_domain);
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

/** \relates fftx::array_t
    Multiplies the contents of the first array by
    the contents of the second array, pointwise:<br>
    Set <tt>a_arr = a_multiplier * a_arr</tt>
    at each point.

    Both arrays <tt>a_arr</tt> and <tt>a_multiplier</tt>
    must be defined on the same domain.
*/
template<int DIM, typename T>
void multiplyByArray(fftx::array_t<DIM, T>& a_arr,
                     const fftx::array_t<DIM, T>& a_multiplier)
{
  assert(a_arr.m_domain == a_multiplier.m_domain);
  auto arrPtr = a_arr.m_data.local();
  auto multiplierPtr = a_multiplier.m_data.local();
  auto dom = a_arr.m_domain;
  size_t npts = dom.size();
  for (size_t ind = 0; ind < npts; ind++)
    {
      arrPtr[ind] *= multiplierPtr[ind];
    }
}

/** \relates fftx::array_t
    Sets the contents of the first array to
    a sum of the contents of the second and third arrays,
    optionally with scalar multiplication of each input array:<br>
    Set <tt>a_sum = a_scaling1 * a_arr1 + a_scaling2 * a_arr2</tt>
    at each point.

    The output array <tt>a_sum</tt> and the two input arrays
    <tt>a_arr1</tt> and <tt>a_arr2</tt>
    must all be defined on the same domain.
*/
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

/** \relates fftx::array_t
    Sets the contents of the first array to
    the difference between the contents of 
    second and third arrays:<br>
    Set <tt>a_diff = a_arr1 - a_arr2</tt>
    at each point.

    The output array <tt>a_diff</tt> and the two input arrays
    <tt>a_arr1</tt> and <tt>a_arr2</tt>
    must all be defined on the same domain.
*/
template<int DIM, typename T>
void diffArrays(fftx::array_t<DIM, T>& a_diff,
                const fftx::array_t<DIM, T>& a_arr1,
                const fftx::array_t<DIM, T>& a_arr2)
               
{
  assert(a_diff.m_domain == a_arr1.m_domain);
  assert(a_diff.m_domain == a_arr2.m_domain);
  sumArrays(a_diff, a_arr1, a_arr2, scalarVal<T>(1.), scalarVal<T>(-1.));
}

/** \relates fftx::array_t
    Sets the contents of the first array to
    the pointwise product of the contents of 
    second and third arrays:<br>
    Set <tt>a_prod = a_arr1 * a_arr2</tt>
    at each point.

    The output array <tt>a_prod</tt> and the two input arrays
    <tt>a_arr1</tt> and <tt>a_arr2</tt>
    must all be defined on the same domain.
*/
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

/** \relates fftx::array_t
    Sets every element of the argument array to the argument value.
 */
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

/** \relates fftx::array_t
    Modifies the argument array by setting every element
    to its complex conjugate.
 */
template<int DIM>
void conjugateArray(fftx::array_t<DIM, std::complex<double>>& a_arr)
{
  forall([](std::complex<double>(&v),
            const fftx::point_t<DIM>& p)
         {
           v = std::conj(v);
         }, a_arr);
}

/** \relates fftx::array_t
    Writes out every element of the array,
    together with its <tt>DIM</tt>-dimensional index in the array
    and its position in the array starting from 0.
 */
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

/** \internal */
inline void updateMax(double& a_max,
                      double a_here)
{
  if (a_here > a_max)
    {
      a_max = a_here;
    }
}

/** \internal */
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

/** \relates fftx::array_t
    Returns the maximum value of the <tt>std::abs</tt> function
    applied to elements of the argument array.
 */
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

/** \relates fftx::array_t
    Returns the maximum value of the <tt>std::abs</tt> function
    applied to the differences in elements of the argument arrays
    with the same index.

    The argument arrays
    <tt>a_arr1</tt> and <tt>a_arr2</tt>
    must be defined on the same domain.
 */
template<int DIM, typename T>
double absMaxDiffArray(fftx::array_t<DIM, T>& a_arr1,
                       fftx::array_t<DIM, T>& a_arr2)
{
  assert(a_arr1.m_domain == a_arr2.m_domain);
  auto arr1Ptr = a_arr1.m_data.local();
  auto arr2Ptr = a_arr2.m_data.local();
  auto dom = a_arr1.m_domain;
  size_t npts = dom.size();
  double absDiffMax = 0.;
  for (size_t ind = 0; ind < npts; ind++)
    {
      T diffHere = arr1Ptr[ind] - arr2Ptr[ind];
      updateMaxAbs(absDiffMax, diffHere);
    }
  return absDiffMax;
}

/** \relates fftx::array_t
    Sets the first argument array to a
    rotation of the second argument array.
    The rotation is specified by a periodic shift
    in dimension <tt>a_dim</tt> (in range 0 to <tt>DIM</tt> - 1)
    by amount <tt>a_shift</tt>.

    The argument arrays
    <tt>a_arrOut</tt> and <tt>a_arrIn</tt>
    must be defined on the same domain.
 */
template<int DIM, typename T>
void rotate(fftx::array_t<DIM, T>& a_arrOut,
            const fftx::array_t<DIM, T>& a_arrIn,
            int a_dim,
            int a_shift)
{
  assert(a_arrIn.m_domain == a_arrOut.m_domain);
  fftx::point_t<DIM> shift;
  for (int d = 0; d < DIM; d++)
    {
      shift[d] = 0;
    }
  shift[a_dim] = a_shift;
  auto inPtr = a_arrIn.m_data.local();
  auto dom = a_arrIn.m_domain;
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

/** \relates fftx::array_t
    Sets the first array to the second-order discrete Laplacian
    of the second array, which is taken as being periodic,
    with unit mesh spacing.

    The argument arrays
    <tt>a_laplacian</tt> and <tt>a_arr</tt>
    must be defined on the same domain.
 */
template<int DIM, typename T>
void laplacian2periodic(fftx::array_t<DIM, T>& a_laplacian,
                        const fftx::array_t<DIM, T>& a_arr)
{
  assert(a_laplacian.m_domain == a_arr.m_domain);
  auto inPtr = a_arr.m_data.local();
  auto dom = a_arr.m_domain;
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
                   fftx::point_t<DIM> shift = fftx::point_t<DIM>::Zero();
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
              fftx::point_t<DIM> shift = fftx::point_t<DIM>::Zero();
              shift[d] = sgn;
              fftx::point_t<DIM> pShift = shiftInBox(p, shift, dom);
              size_t indShift = positionInBox(pShift, dom);
              laplacianElem += arrPtr[indShift] - arrElem;
            }
        }
      laplacianPtr[ind] = laplacianElem;
    }
}

/** \relates fftx::array_t
    Returns the dimensions of a truncated complex array
    that is the result of a
    multidimensional discrete Fourier transform on a real-valued array
    having the dimensions in the argument.
 */
template<int DIM>
fftx::point_t<DIM> truncatedComplexDimensions(fftx::point_t<DIM>& a_size)
{
  fftx::point_t<DIM> truncSize = a_size;
#if FFTX_COMPLEX_TRUNC_LAST
  truncSize[DIM-1] = a_size[DIM-1]/2 + 1;
#else
  truncSize[0] = a_size[0]/2 + 1;
#endif
  return truncSize;
}


/** \internal */
inline int sym_index(int i, int lo, int hi)
{ // index of i in array lo:hi that should be conjugate if Hermitian symmetry
  int ret = i;
  if (! ((i == lo) || (2*i == lo + hi + 1)) )
    {
      ret = lo + hi + 1 - i;
    }
  return ret;
}

/** \internal */
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

/** \relates fftx::array_t
    If T_IN is <tt>std::complex<double></tt>
    and T_OUT is <tt>double</tt>, then
    modifies the first array where necessary to give it Hermitian symmetry.

    A multidimensional discrete Fourier transform
    applied to a real-valued array
    gives a complex-valued array with Hermitian symmetry,
    and any complex-valued array with Hermitian symmetry can be the result
    of a multidimensional discrete Fourier transform applied to
    a real-valued array.
    When we perform tests of complex-to-real multimensional discrete
    Fourier transforms, we use arrays of random complex data that have been
    modified with this function so that the transform gives a real-valued
    result.

    Hermitian symmetry requires that array elements in some positions be real,
    so this function sets the imaginary part of those elements to zero.<br>
    Hermitian symmetry also requires that array elements in some positions
    be the complex conjugate of array elements in some other positions,
    so this function sets those elements accordingly.

    This function does not change the second argument array, but only
    checks that its domain is correct for a real-valued multidimensional array
    that is the result of a multidimensional discrete Fourier transform
    on the domain of the first array.

    If T_IN is not <tt>std::complex<double></tt>
    or T_OUT is not <tt>double</tt>, then
    this function has no effect.

 */
template<int DIM, typename T_IN, typename T_OUT>
void symmetrizeHermitian(fftx::array_t<DIM, T_IN>& a_arrIn,
                         fftx::array_t<DIM, T_OUT>& a_arrOut);


/** \internal */
template<int DIM>
void symmetrizeHermitian(fftx::array_t<DIM, double>& a_arrIn,
                         fftx::array_t<DIM, double>& a_arrOut)
{ };

/** \internal */
template<int DIM>
void symmetrizeHermitian(fftx::array_t<DIM, double>& a_arrIn,
                         fftx::array_t<DIM, std::complex<double>>& a_arrOut)
{ };

/** \internal */
template<int DIM>
void symmetrizeHermitian(fftx::array_t<DIM, std::complex<double>>& a_arrIn,
                         fftx::array_t<DIM, std::complex<double>>& a_arrOut)
{ };

/** \internal */
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

  // Check that a complex-valued array on the input domain
  // transforms to a real-valued array on the output domain.
  fftx::point_t<DIM> inputLo = inputDomain.lo;
  assert(inputLo == lo);
  fftx::point_t<DIM> inputDimsNeeded = truncatedComplexDimensions(extent);
  fftx::point_t<DIM> inputDims = inputDomain.extents();
  assert(inputDimsNeeded == inputDims);

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


/** \internal */
template<int DIM, typename T_IN, typename T_OUT>
bool checkSymmetryHermitian(fftx::array_t<DIM, T_IN>& a_arrIn,
                            fftx::array_t<DIM, T_OUT>& a_arrOut);


/** \internal */
template<int DIM>
bool checkSymmetryHermitian(fftx::array_t<DIM, double>& a_arrIn,
                            fftx::array_t<DIM, std::complex<double>>& a_arrOut)
{
  return true;
}

/** \internal */
template<int DIM>
bool checkSymmetryHermitian(fftx::array_t<DIM, std::complex<double>>& a_arrIn,
                            fftx::array_t<DIM, std::complex<double>>& a_arrOut)
{
  return true;
}

// Not a good idea to be checking equality of reals.
/** \internal */
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

/** \internal */
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
