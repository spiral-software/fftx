// make

#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>


using namespace fftx;


int main(int argc, char* argv[])
{
  const int l = 224;
  const int m = 224;
  const int n = 100;
  const char* name="DFT_224_224_100";
#include "forward.h"

  return 0;
}
