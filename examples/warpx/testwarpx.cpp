// make

#include "fftx3.hpp"
#include <cstdio>
#include <cassert>

using namespace fftx;

#include "psatd.fftx.codegen.hpp"

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

   
  psatd::init();
  psatd::transform(inputs, outputs, symvars);

  // BEGIN DEBUG
  if (true)
    {
      FILE* fout = fopen("fftxout", "w");
      forall([fout](double& v, const point_t<3>& point)
             {

               fprintf(fout, "%20.8e %20.8e\n", v);
             }, outputs[0]);
      fclose(fout);
    }
  // END DEBUG


  psatd::destroy();
  return 0;
}
