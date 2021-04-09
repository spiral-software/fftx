using namespace fftx;

namespace hockney
{
  // Here's the picture:

  //               rdomain
  // |<-------------- n ------------->|
  // |<- ns ->|              |<- nd ->|
  //    sbox                    dbox

  // n >= ns + nd
  const int ns =  3; // 8; // 33;
  const int nd =  5; // 24; // 96;
  const int n  = 10; // 33; // 130;
  const int noutShift = n + ns;

  // box_t<3> sbox({{0, 0, 0}}, {{ns-1, ns-1, ns-1}});
  box_t<3> sbox({{0, 0, 0}}, {{2*ns, 2*ns, 2*ns}});
  // box_t<3> dbox({{n-nd, n-nd, n-nd}}, {{n-1, n-1, n-1}});
  box_t<3> dbox({{noutShift-nd, noutShift-nd, noutShift-nd}},
                {{noutShift+nd, noutShift+nd, noutShift+nd}});
  // box_t<3> rdomain({{0, 0, 0}}, {{n-1, n-1, n-1}});
  box_t<3> rdomain({{0, 0, 0}}, {{2*n-1, 2*n-1, 2*n-1}});
  // box_t<3> freq({{0, 0, 0}}, {{(n-1)/2+1, n-1, n-1}});
  box_t<3> freq({{0, 0, 0}}, {{(2*n-1)/2+1, 2*n-1, 2*n-1}});
}
