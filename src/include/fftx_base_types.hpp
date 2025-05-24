#pragma once

//  Copyright (c) 2018-2025, Carnegie Mellon University
//   All rights reserved.
//
//  See LICENSE file for full information

/*
 *  FFTX base types definitions: point_t, box_t, array_t, and global_ptr
 *
 *  These are defined here (and included by fftx.hpp) for convenience and to overcome
 *  limitations with doxygen which is unable to correctly extract and format the XML
 *  documentation wehn they ar eall embedded directly in fftx.hpp
 *
 *  This file is not intended to be included directly -- use "fftx.hpp"
 */

/*!
 * \addtogroup fftx
 * @{
 */
namespace fftx
{

  /**
     Is this a FFTX codegen program, or is this application code using a generated transform.
  */
  static bool tracing = false; // when creating a trace program user sets this to 'true'

  /**
     For writing things out.
   */

  static std::ostream* osout = &std::cout;
  
  static std::ostream* oserr = &std::cerr;

  inline std::ostream& OutStream() { return *osout; }

  inline std::ostream& ErrStream() { return *oserr; }

  inline void setOutStream(std::ostream* os) { osout = os; }

  inline void setErrStream(std::ostream* os) { oserr = os; }

  static std::ostream osnull(NULL);
  
  inline void setOutSilent() { osout = &osnull; }

  inline void setErrSilent() { oserr = &osnull; }

  /**
     Counter for generated variable names during FFTX codegen tracing.
     Not meant for FFTX users, but can be used when debugging codegen itself.
  */
  static uint64_t ID=1; // variable naming counter
  
  /** \internal */
  typedef int intrank_t; // just useful for self-documenting code.

  /** \internal */
  struct handle_implem_t;

  /** \internal */
  struct handle_t
  {
  private:
    std::shared_ptr<handle_implem_t> m_implem;
  };

  /**
   * @brief A non-owning pointer to memory in a global or multi-device context.
   *
   * This pointer wrapper does not manage ownership or lifetime, and may refer
   * to memory allocated by the host application or externally shared across
   * distributed systems or GPUs.
   *
   * This class is functionally similar to `upcxx::global_ptr`, but used
   * internally in FFTX to mark data as non-owning. It can be copied, moved,
   * and reassigned like a raw pointer.
   *
   * @tparam T Element data type pointed to
   */
  template <typename T>
  class global_ptr
  {
    T* _ptr;
    intrank_t _domain;
    int _device;

  public:
    using element_type = T;

    /** Default constructor with no data, no domain, no device. */
    global_ptr():_ptr{nullptr},_domain{0}, _device{0}{}

    /** Real strong constructor. */
    global_ptr(T* ptr, int domain=0, int device=0)
      :_ptr{ptr}, _domain{domain}, _device{device}{ }

    /** Returns true if this object has no data array assigned to it. */
    bool is_null() const { return (_ptr == nullptr); }

    /** Returns true if this local compute context can successfully call
        the local() function and expect to get a pointer that
        can be dereferenced. */
    bool is_local() const;

    /** Returns the compute domain that would answer "true" to <tt>isLocal()</tt>.
        Currently this just tests if the MPI rank matches my_rank. */
    intrank_t where() const {return _domain;}

    /** Returns the GPU device that this pointer associated with. */
    int device() const {return _device;}

    /** Returns the raw pointer.  This pointer can only be dereferenced if
        <tt> is_local()</tt> is true. */
    T* local() {return _ptr;}

    /** Returns the raw pointer.  This pointer can only be dereferenced if
        <tt> is_local()</tt> is true. */
    const T* local() const {return _ptr;}

    /** type erasure cast */
    //operator global_ptr<void>(){ return global_ptr<void>(_ptr, _domain, _device);}
  };

 
  /**
   * @brief A tuple of integer coordinates in Z^DIM space.
   *
   * This struct represents a point in a DIM-dimensional integer lattice.
   * It provides coordinate access, assignment, and projection utilities.
   *
   * Example usage:
   * \code
   * fftx::point_t<3> pt({1, 2, 3});
   * \endcode
   *
   * @tparam DIM Number of spatial dimensions.
   */

  template<int DIM>
  struct point_t
  {
    /** Array containing the coordinate in each direction. */
    int x[DIM];

    /** Assigns this <tt>point_t</tt> by setting the coordinates in every dimension to the argument. */
    void operator=(int a_default);

    /** Returns the value of the coordinate in the specified direction. */
    int operator[](unsigned char a_id) const {return x[a_id];}

    /** Returns a reference to the coordinate in the specified direction. */
    int& operator[](unsigned char a_id) {return x[a_id];}

    /** Returns true if all coordinates of this <tt>point_t</tt> are the same as the corresponding coordinates in the argument <tt>point_t</tt>, and false if any of the coordinates differ. */
    bool operator==(const point_t<DIM>& a_rhs) const;

    /** Modifies this <tt>point_t</tt> by multiplying all of its coordinates by the argument. */
    point_t<DIM> operator*(int scale) const;

    /** Returns the dimension of this <tt>point_t</tt>. */
    static int dim() {return DIM;}

    /** Returns the product of the components of this <tt>point_t</tt>. */
    int product();
    
    /** Returns a new <tt>point_t</tt> in one lower dimension, dropping the last coordinate value. */
    point_t<DIM-1> project() const;

    /** Returns a new <tt>point_t</tt> in one lower dimension, dropping the first coordinate value. */
    point_t<DIM-1> projectC() const;

    /** Returns a <tt>point_t</tt> with all components equal to one. */
    static point_t<DIM> Unit();

    /** Returns a <tt>point_t</tt> with all components equal to zero. */
    static point_t<DIM> Zero();

    /** Returns a <tt>point_t</tt> with the same coordinates as this <tt>point_t</tt> but with their ordering reversed. */
    point_t<DIM> flipped() const { point_t<DIM> rtn; for (int d=0; d<DIM; d++) { rtn[d] = x[DIM-1 - d]; } return rtn; }
  };

  /**
   * @brief A rectangular domain on an integer lattice in DIM-dimensional integer space.
   *
   * Represents a region defined by its low and high corners in Z^DIM.
   *
   * Example usage:
   * \code
   * fftx::box_t<3> domain ( fftx::point_t<3> ( { { 1, 1, 1 } } ),
   *                         fftx::point_t<3> ( { { mm, nn, kk } } ));
   * \endcode
   *
   * @tparam DIM Number of spatial dimensions.
   */
  template<int DIM>
  struct box_t
  {
    /** Default constructor leaves box in an undefined state. */
    box_t() = default;

    /** Constructs a box having the given inputs as the low and high corners, respectively. */
    box_t(const point_t<DIM>&& a, const point_t<DIM>&& b)
      : lo(a), hi(b) { ; }

    /** Constructs a box having the given inputs as the low and high corners, respectively. */
    box_t(const point_t<DIM>& a, const point_t<DIM>& b)
      : lo(a), hi(b) { ; }

    /** The low corner of the box in index space. */
    point_t<DIM> lo;

    /** The high corner of the box in index space. */
    point_t<DIM> hi;

    /** Returns the number of points in the box in index space. */
    std::size_t size() const;
    
    /** Returns true if the corners of this box are the same as the corners of the argument box. */
    bool operator==(const box_t<DIM>& rhs) const {return lo==rhs.lo && hi == rhs.hi;}

    /** Returns a <tt>point_t</tt> object containing the length of the box in each coordinate direction. */
    point_t<DIM> extents() const { point_t<DIM> rtn(hi); for(int i=0; i<DIM; i++) rtn[i]-=(lo[i]-1); return rtn;}

    /** Returns a <tt>box_t</tt> object in one lower dimension, dropping the first coordinate value in both <tt>box_t::lo</tt> and <tt>box_t::hi</tt>. */
    box_t<DIM-1> projectC() const
    {
      return box_t<DIM-1>(lo.projectC(),hi.projectC());
    }
  };

  /**
   * @brief A non-owning view into a contiguous array of DIM-dimensional data.
   *
   * This class associates storage with a `box_t<DIM>` domain.
   * The array holds values of type `T`, indexed by integer coordinates.
   *
   * Example usage:
   * \code
   * fftx::array_t<3,double> realFFTXHostArray(domain);
   * \endcode
   *
   * @tparam DIM Number of spatial dimensions.
   * @tparam T   Type of values stored in the array.
   */
  template<int DIM, typename T>
  struct array_t
  {
 
    array_t() = default;

    /** Strong constructor from an aliased <tt>global_ptr</tt> object.
        This constructor is an error when <tt>fftx::tracing</tt> is true. */
    array_t(global_ptr<T>&& p, const box_t<DIM>& a_box)
      :m_data(p), m_domain(a_box) {;}

    /** Constructor from a domain.

      If <tt>fftx::tracing</tt> is true, then this constructor
      is a symbolic placeholder in a computational DAG that
      is translated into the code generator.
        
      If <tt>fftx::tracing</tt> is false, then this constructor
      will allocate a <tt>global_ptr</tt> that is sized 
      to hold <tt>a_box.size()</tt> elements of data of type T.
    */
    array_t(const box_t<DIM>& a_box):m_domain(a_box)
    {
      if (tracing)
        {
          m_data = global_ptr<T>((T*)ID);
          OutStream()<<"var_"<<ID<<":= var(\"var_"<<ID<<"\", BoxND("<<a_box.extents()<<", TReal));\n";
          ID++;
        }
      else
        {
          m_local_data = new T[m_domain.size()];
          m_data = global_ptr<T>(m_local_data);
        }
    }

    /** Destructor, which deletes the local data if there is any. */
    ~array_t()
    {
      if (m_local_data != nullptr)
        {
          delete[] m_local_data;
        }
    }

    /** Swaps the contents of first array and second array. */
    friend void swap(array_t& first, array_t& second)
    {
      using std::swap;
      swap(first.m_local_data, second.m_local_data);
      swap(first.m_data, second.m_data);
      swap(first.m_domain, second.m_domain);
    }

    T* m_local_data = nullptr;

    /** Pointer to a block in memory containing the data. */
    global_ptr<T> m_data;

    /** The domain (box) on which the array is defined. */
    box_t<DIM>    m_domain;

    array_t<DIM, T> subArray(box_t<DIM>&& subbox);

    uint64_t id() const { assert(tracing); return (uint64_t)m_data.local();}
  };


#ifdef FFTX_DOXYGEN
    // FFTX_DOXYGEN should never be defined when compiling code; it's only for use with doxygen to generate doc
    // The block below is processed by doxygen (to instantiate the templates) but is NOT emitted in the final docs
    
/// \cond DOXYGEN_SHOULD_SKIP_THIS
    template struct point_t<3>;
    template struct box_t<3>;
    template struct array_t<3, double>;
    template class global_ptr<double>;
/// \endcond
#endif


} // namespace fftx
/*! @} */
