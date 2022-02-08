// make

//  #include "fftx3.hpp"
//  #include <cstdio>
//  #include <cassert>

#include "warpx.fftx.codegen.hpp"

using namespace fftx;

//  #include "warpx.fftx.source.cpp"

#include "defineArrays.hpp"

int main(int argc, char* argv[])
{

 
  std::array<array_t<3,double>,11> inputs;
  std::array<array_t<3,double>,6>  outputs;
  std::array<array_t<3,double>,8>  symvars;

  defineArrays(inputs, outputs, symvars);
  
  forall([](double& jz, const point_t<3>& p)
         {
           jz = 0.0;
         }, inputs[8]);

   
  warpx::init();
  warpx::transform(inputs, outputs, symvars);

  // BEGIN DEBUG
  if (true)
    {
      FILE* fout = fopen("fftxout", "w");
      forall([fout](double& v, const point_t<3>& point)
             {
               fprintf(fout, "%20.8e %4d %4d %4d\n",
                       v, point[0], point[1], point[2]);
             }, outputs[0]);
      fclose(fout);
    }
  // END DEBUG


  warpx::destroy();
  return 0;
}
