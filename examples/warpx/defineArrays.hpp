// make

#ifndef DEFINE_ARRAYS_HPP
#define DEFINE_ARRAYS_HPP

#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>

// using namespace fftx;

const int nx=80, ny=80, nz=80;
const int norm_box = nx*ny*nz;
const int npx = nx+1, npy = ny+1, npz = nz+1;

inline void defineBigBoxes(std::array<fftx::array_t<4, double>,4>& a_bigBoxes)
{
#if FFTX_ROW_MAJOR_ORDER
  /*  C ordering */
  fftx::box_t<4> bigBox0(fftx::point_t<4>({{1,1,1,1}}),
                         fftx::point_t<4>({{11, nz, ny, nx}}));
  fftx::box_t<4> bigBox1(fftx::point_t<4>({{1,1,1,1}}),
                         fftx::point_t<4>({{11,nz,ny,nx+2}}));

  fftx::box_t<4> bigBox2(fftx::point_t<4>({{1,1,1,1}}),
                         fftx::point_t<4>({{6,nz,ny,nx+2}}));
  fftx::box_t<4> bigBox3(fftx::point_t<4>({{1,1,1,1}}),
                         fftx::point_t<4>({{6,nz,ny,nx}}));
#else
  /*  Fortran ordering */
  fftx::box_t<4> bigBox0(fftx::point_t<4>({{1,1,1,1}}),
                         fftx::point_t<4>({{nx, ny, nz, 11}}));
  fftx::box_t<4> bigBox1(fftx::point_t<4>({{1,1,1,1}}),
                         fftx::point_t<4>({{nx+2,ny,nz,11}}));

  fftx::box_t<4> bigBox2(fftx::point_t<4>({{1,1,1,1}}),
                         fftx::point_t<4>({{nx+2,ny,nz,6}}));
  fftx::box_t<4> bigBox3(fftx::point_t<4>({{1,1,1,1}}),
                         fftx::point_t<4>({{nx,ny,nz,6}}));
#endif
  
  std::array<fftx::array_t<4, double>,4> bigBoxes = {bigBox0,
                                                     bigBox1,
                                                     bigBox2,
                                                     bigBox3};
  std::swap(bigBoxes, a_bigBoxes);
}
  
       
inline void defineArrays(std::array<fftx::array_t<3,double>,11>& a_inputs,
                         std::array<fftx::array_t<3,double>,6 >& a_outputs,
                         std::array<fftx::array_t<3,double>,8 >& a_symvars)
{
#if FFTX_ROW_MAJOR_ORDER
  /*  C ordering */
  fftx::box_t<3> cell(fftx::point_t<3>({{1,1,1}}),
                      fftx::point_t<3>({{nz,ny,nx}}));
  fftx::box_t<3> xface(fftx::point_t<3>({{1,1,1}}),
                       fftx::point_t<3>({{nz,ny,npx}}));
  fftx::box_t<3> yface(fftx::point_t<3>({{1,1,1}}),
                       fftx::point_t<3>({{nz,npy,nx}}));
  fftx::box_t<3> zface(fftx::point_t<3>({{1,1,1}}),
                       fftx::point_t<3>({{npz,ny,nx}}));
  fftx::box_t<3> xedge(fftx::point_t<3>({{1,1,1}}),
                       fftx::point_t<3>({{npz,npy,nx}}));
  fftx::box_t<3> yedge(fftx::point_t<3>({{1,1,1}}),
                       fftx::point_t<3>({{npz,ny,npx}}));
  fftx::box_t<3> zedge(fftx::point_t<3>({{1,1,1}}),
                       fftx::point_t<3>({{nz,npy,npx}}));
#else
  /*  Fortran ordering */
  fftx::box_t<3> cell(fftx::point_t<3>({{1,1,1}}),
                      fftx::point_t<3>({{nx,ny,nz}}));
  fftx::box_t<3> xface(fftx::point_t<3>({{1,1,1}}),
                       fftx::point_t<3>({{npx,ny,nz}}));
  fftx::box_t<3> yface(fftx::point_t<3>({{1,1,1}}),
                       fftx::point_t<3>({{nx,npy,nz}}));
  fftx::box_t<3> zface(fftx::point_t<3>({{1,1,1}}),
                       fftx::point_t<3>({{nx,ny,npz}}));
  fftx::box_t<3> xedge(fftx::point_t<3>({{1,1,1}}),
                       fftx::point_t<3>({{nx,npy,npz}}));
  fftx::box_t<3> yedge(fftx::point_t<3>({{1,1,1}}),
                       fftx::point_t<3>({{npx,ny,npz}}));
  fftx::box_t<3> zedge(fftx::point_t<3>({{1,1,1}}),
                       fftx::point_t<3>({{npx,npy,nz}}));
#endif
  
  //  fftx::array_t<1,double> Ex(xedge), Ey(yedge),Ez(zedge);
  //  fftx::array_t<1,double> Bx(xface), By(yface),Bz(zface);
  // fftx::array_t<1,double> Cx(xedge), Cy(yedge),Cz(Cedge);
  // fftx::array_t<1,double> rho0(cell), rho1(cell);

  std::array<fftx::array_t<3,double>,11> input = {xedge, yedge, zedge,/*E*/
                                                  xface, yface, zface,/*B*/
                                                  xedge, yedge, zedge,/*J*/
                                                  cell, cell};


#if FFTX_ROW_MAJOR_ORDER
  /* C ordering */
  fftx::box_t<3> zbox(fftx::point_t<3>({{1,1,1}}),fftx::point_t<3>({{nz,1,1}}));
  fftx::box_t<3> ybox(fftx::point_t<3>({{1,1,1}}),fftx::point_t<3>({{1,ny,1}}));
  fftx::box_t<3> xbox(fftx::point_t<3>({{1,1,1}}),fftx::point_t<3>({{1,1,nx}}));
#else
  /* Fortran ordering */
  fftx::box_t<3> xbox(fftx::point_t<3>({{1,1,1}}),fftx::point_t<3>({{nx,1,1}}));
  fftx::box_t<3> ybox(fftx::point_t<3>({{1,1,1}}),fftx::point_t<3>({{1,ny,1}}));
  fftx::box_t<3> zbox(fftx::point_t<3>({{1,1,1}}),fftx::point_t<3>({{1,1,nz}}));
#endif

  std::array<fftx::array_t<3,double>,8> symvars = {xbox,/*fmkx*/
                                                   ybox,/*fmky*/
                                                   zbox,/*fmkz*/
                                                   cell,/*fc*/
                                                   cell,/*fsckv*/
                                                   cell,/*fx1v*/
                                                   cell,/*fx2v*/
                                                   cell /*fx3v*/};

  std::array<fftx::array_t<3, double >,6> output ={zedge, yedge, xedge,
                                                   zface, yface, xface};

  std::swap(input,  a_inputs);
  std::swap(output, a_outputs);
  std::swap(symvars, a_symvars);
}

#endif
