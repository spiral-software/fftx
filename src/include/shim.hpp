#include <iostream>
// #include <complex>
// #include <fftw3.h>
#include <algorithm>
#include <any>
#include <string>
#include <cstdlib>
#include <vector>
#include <functional>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <tuple>
#include <utility>
#if defined FFTX_HIP
#include "hipbackend.hpp"
#elif defined FFTX_CUDA
#include "hipbackend.hpp"
#else
#include "cpubackend.hpp"
#endif
#pragma once

typedef std::tuple<int, int, int, int> keys_t;
 
struct key_hash : public std::unary_function<keys_t, std::size_t>
{
std::size_t operator()(const keys_t& k) const
{
return std::get<0>(k) ^ std::get<1>(k) ^ std::get<2>(k) ^ std::get<3>(k);
}
};
 
struct key_equal : public std::binary_function<keys_t, keys_t, bool>
{
bool operator()(const keys_t& v0, const keys_t& v1) const
{
return (
std::get<0>(v0) == std::get<0>(v1) &&
std::get<1>(v0) == std::get<1>(v1) &&
std::get<2>(v0) == std::get<2>(v1) &&
std::get<3>(v0) == std::get<3>(v1) 
);
}
};

typedef std::unordered_map<const keys_t,std::string,key_hash,key_equal> map_t;

map_t stored_mddft_jit;


namespace fftx_fftw {

    hipfftResult hipfftExecZ2Z(hipfftHandle plan, hipfftDoubleComplex *idata, hipfftDoubleComplex *odata, int direction) {

    }

     void copy2(const std::vector<double> input, std::vector<double> output) {
        std::cout << "We are inside fftx_fftw::copy2" << std::endl;
        convolve_output_begin = output.begin();
        std::vector<std::any> *test = new std::vector<std::any>();
        test->push_back(input);
        test->push_back(output);
        dag.addArg(test); 
        dag.setName(__FUNCTION__);
    }
    // Taken from https://www.fftw.org/fftw3_doc/FFTW-Reference.html
    fftw_plan fftw_plan_dft_r2c_1d(int n0, double *in, fftw_complex *out,
     unsigned flags) {
        std::cout << "We are inside fftx_fftw::fftw_plan_dft_r2c_1d" << std::endl;
        std::cout <<  __FUNCTION__ << std::endl;

        std::vector<std::any> *test = new std::vector<std::any>();
        test->push_back(n0);
        test->push_back(in);
        test->push_back(out);
        test->push_back(flags);
        dag.addArg(test);
        dag.setName(__FUNCTION__);
        std::string spiral = "DFT(" + std::to_string(n0) + ", -1)";
        dag.setSpiralInfo(spiral);

        // return ::fftw_plan_dft_r2c_1d(n0, in, out, flags);
        return NULL;
    }

    fftw_plan fftw_plan_dft_c2r_1d(
        int n0, fftw_complex *in, double *out, unsigned flags) {
        std::cout << "We are inside fftx_fftw::fftw_plan_dft_c2r_1d" << std::endl;
        std::cout <<  __FUNCTION__ << std::endl;
        std::vector<std::any> *test = new std::vector<std::any>();
        test->push_back(n0);
        test->push_back(in);
        test->push_back(out);
        test->push_back(flags);
        dag.addArg(test);
        dag.setName(__FUNCTION__);
        std::string spiral = "DFT(" + std::to_string(n0) + ", 1)";
        dag.setSpiralInfo(spiral);
        // return ::fftw_plan_dft_c2r_1d(n0, in, out, flags);
        return NULL;
    }

    void fftw_execute(const fftw_plan plan){
        std::cout << "We are inside fftx_fftw::fftw_execute" << std::endl;
        std::cout <<  __FUNCTION__ << std::endl;
        std::vector<std::any> *test  = new std::vector<std::any>();
        test->push_back(plan);
        dag.addArg(test);
        dag.setName(__FUNCTION__);
    }

}

// #if defined(HP) && !defined(HP_USING_STD)
// namespace std {
//     namespace fftx_fftw {
//         // Taken from https://en.cppreference.com/w/cpp/algorithm/transform
//         template< class InputIt1, class InputIt2,
//               class OutputIt, class BinaryOperation >
//         OutputIt transforms( InputIt1 first1, InputIt1 last1, InputIt2 first2,
//                         OutputIt d_first, BinaryOperation binary_op ) {
//             std::cout << "We are inside fftx_fftw::transform" << std::endl;
//             std::cout << __FUNCTION__ << std::endl;
//             std::vector<std::any> *test = new std::vector<std::any>();
//             test->push_back(first1);
//             test->push_back(last1);
//             test->push_back(first2);
//             test->push_back(d_first);
//             test->push_back(binary_op);
//             dag.addArg(test);
//             dag.setName(__FUNCTION__);
//             std::string spiral = "Diag(FDataOfs(symvar," + std::to_string(last1-first1) + ", 0))";
//             dag.setSpiralInfo(spiral);
//             return d_first;
//         }

//         // Taken from https://en.cppreference.com/w/cpp/algorithm/copy
//         template< class InputIt, class OutputIt >
//         OutputIt copys( InputIt first, InputIt last, OutputIt d_first ) {
//             std::cout << "We are inside fftx_fftw::copy" << std::endl;
//             convolve_output_begin = d_first;
//             std::cout << __FUNCTION__ << std::endl;
//             std::vector<std::any> *test = new std::vector<std::any>();
//             test->push_back(first);
//             test->push_back(last);
//             test->push_back(d_first);
//             dag.addArg(test);
//             dag.setName(__FUNCTION__);
//             return d_first;
//         }
//     }
// }
// #endif


#define hipfftExecZ2Z fftx_fftw::hipfftExecZ2Z

// #define fftw_plan_dft_r2c_1d fftx_fftw::fftw_plan_dft_r2c_1d
// #define fftw_plan_dft_c2r_1d fftx_fftw::fftw_plan_dft_c2r_1d
// #define fftw_execute fftx_fftw::fftw_execute
// #define transform(a,b,c,d,e) fftx_fftw::transforms(a,b,c,d,e)
// #define copy(a,b,c) fftx_fftw::copys(a,b,c)
// #define copy2 fftx_fftw::copy2
