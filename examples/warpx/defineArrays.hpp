// make

#ifndef DEFINE_ARRAYS_HPP
#define DEFINE_ARRAYS_HPP

#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>

using namespace fftx;

const int nx=80, ny=80, nz=80;
const int norm = nx*ny*nz;
const int npx = nx+1, npy = ny+1, npz = nz+1;

inline void defineBigBoxes(std::array<array_t<4, double>,4>& a_bigBoxes)
{
  /*  Fortran ordering
  box_t<4> bigBox0(point_t<4>({{1,1,1,1}}),
                   point_t<4>({{nx, ny, nz, 11}}));
  box_t<4> bigBox1(point_t<4>({{1,1,1,1}}),
                   point_t<4>({{nx+2,ny,nz,11}}));
  
  box_t<4> bigBox2(point_t<4>({{1,1,1,1}}),
                   point_t<4>({{nx+2,ny,nz,6}}));
  box_t<4> bigBox3(point_t<4>({{1,1,1,1}}),
                   point_t<4>({{nx,ny,nz,6}}));
  */

  /*  C ordering */
  box_t<4> bigBox0(point_t<4>({{1,1,1,1}}),
                   point_t<4>({{11, nz, ny, nx}}));
  box_t<4> bigBox1(point_t<4>({{1,1,1,1}}),
                   point_t<4>({{11,nz,ny,nx+2}}));
  
  box_t<4> bigBox2(point_t<4>({{1,1,1,1}}),
                   point_t<4>({{6,nz,ny,nx+2}}));
  box_t<4> bigBox3(point_t<4>({{1,1,1,1}}),
                   point_t<4>({{6,nz,ny,nx}}));
  std::array<array_t<4, double>,4> bigBoxes = {bigBox0,
                                               bigBox1,
                                               bigBox2,
                                               bigBox3};
  std::swap(bigBoxes, a_bigBoxes);
}
  
       
inline void defineArrays(std::array<array_t<3,double>,11>& a_inputs,
                         std::array<array_t<3,double>,6 >& a_outputs,
                         std::array<array_t<3,double>,8 >& a_symvars)
{

  /*  Fortran ordering  
  box_t<3> cell(point_t<3>({{1,1,1}}),
                 point_t<3>({{nx,ny,nz}}));
  box_t<3> xface(point_t<3>({{1,1,1}}),
                 point_t<3>({{npx,ny,nz}}));
  box_t<3> yface(point_t<3>({{1,1,1}}),
                 point_t<3>({{nx,npy,nz}}));
  box_t<3> zface(point_t<3>({{1,1,1}}),
                 point_t<3>({{nx,ny,npz}}));
  box_t<3> xedge(point_t<3>({{1,1,1}}),
                 point_t<3>({{nx,npy,npz}}));
  box_t<3> yedge(point_t<3>({{1,1,1}}),
                 point_t<3>({{npx,ny,npz}}));
  box_t<3> zedge(point_t<3>({{1,1,1}}),
                 point_t<3>({{npx,npy,nz}}));
  */

  /*  C ordering  */
  box_t<3> cell(point_t<3>({{1,1,1}}),
                 point_t<3>({{nz,ny,nx}}));
  box_t<3> xface(point_t<3>({{1,1,1}}),
                 point_t<3>({{nz,ny,npx}}));
  box_t<3> yface(point_t<3>({{1,1,1}}),
                 point_t<3>({{nz,npy,nx}}));
  box_t<3> zface(point_t<3>({{1,1,1}}),
                 point_t<3>({{npz,ny,nx}}));
  box_t<3> xedge(point_t<3>({{1,1,1}}),
                 point_t<3>({{npz,npy,nx}}));
  box_t<3> yedge(point_t<3>({{1,1,1}}),
                 point_t<3>({{npz,ny,npx}}));
  box_t<3> zedge(point_t<3>({{1,1,1}}),
                 point_t<3>({{nz,npy,npx}}));



  
  //  array_t<1,double> Ex(xedge), Ey(yedge),Ez(zedge);
  //  array_t<1,double> Bx(xface), By(yface),Bz(zface);
  // array_t<1,double> Cx(xedge), Cy(yedge),Cz(Cedge);
  // array_t<1,double> rho0(cell), rho1(cell);

  std::array<array_t<3,double>,11> input = {xedge, yedge, zedge,/*E*/
                                             xface, yface, zface,/*B*/
                                             xedge, yedge, zedge,/*J*/
                                             cell, cell};


  /* Fortran ordering  
  box_t<3> xbox(point_t<3>({{1,1,1}}),point_t<3>({{nx,1,1}}));
  box_t<3> ybox(point_t<3>({{1,1,1}}),point_t<3>({{1,ny,1}}));
  box_t<3> zbox(point_t<3>({{1,1,1}}),point_t<3>({{1,1,nz}}));
  */
  /* C ordering */
  box_t<3> zbox(point_t<3>({{1,1,1}}),point_t<3>({{nz,1,1}}));
  box_t<3> ybox(point_t<3>({{1,1,1}}),point_t<3>({{1,ny,1}}));
  box_t<3> xbox(point_t<3>({{1,1,1}}),point_t<3>({{1,1,nx}}));

  std::array<array_t<3,double>,8> symvars = {xbox,/*fmkx*/
                                             ybox,/*fmky*/
                                             zbox,/*fmkz*/
                                             cell,/*fc*/
                                             cell,/*fsckv*/
                                             cell,/*fx1v*/
                                             cell,/*fx2v*/
                                             cell /*fx3v*/};



  std::array<array_t<3, double >,6> output ={zedge, yedge, xedge,
                                              zface, yface, xface};

  std::swap(input,  a_inputs);
  std::swap(output, a_outputs);
  std::swap(symvars, a_symvars);
}

#endif
