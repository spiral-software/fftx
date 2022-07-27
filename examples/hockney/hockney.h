using namespace fftx;

namespace hockney
{
  // Here's the picture:

  //               rdomain
  // |<-------------- n ------------->|
  // |<- ns ->|              |<- nd ->|
  //    sbox                    dbox

  // n >= ns + nd
  const int ns = 33; // 45;
  const int nd = 96; // 31;
  const int n  = 130; // 80;

  box_t<3> sbox({{0, 0, 0}}, {{ns-1, ns-1, ns-1}});
  box_t<3> dbox({{n-1-nd, n-1-nd, n-1-nd}}, {{n-1, n-1, n-1}});
  box_t<3> rdomain({{0, 0, 0}}, {{n-1, n-1, n-1}});
#if FFTX_COMPLEX_TRUNC_LAST
  box_t<3> freq({{0, 0, 0}}, {{n-1, n-1, (n-1)/2+1}});
#else
  box_t<3> freq({{0, 0, 0}}, {{(n-1)/2+1, n-1, n-1}});
#endif
}
