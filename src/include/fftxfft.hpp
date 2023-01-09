#include <cstdlib>
#include <vector>
#include <functional>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <tuple>
#include <utility>
// #include "cudabackend.hpp"
#include "hipbackend.hpp"
#include "fftx3.hpp"
#include <array>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <sys/stat.h>
#include <fcntl.h>
#include <memory>
#include <unistd.h>    // dup2
#include <sys/types.h> // rest for open/close
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <array>
#include "interface.hpp"
#include "mddftlib.hpp"
#include "imddftlib.hpp"
#include "dftbatlib.hpp"
#include "idftbatlib.hpp"
#pragma once

typedef std::tuple<int, int, int> keys_t;
 
struct key_hash : public std::unary_function<keys_t, std::size_t>
{
std::size_t operator()(const keys_t& k) const
{
return std::get<0>(k) ^ std::get<1>(k) ^ std::get<2>(k);
}
};
 
struct key_equal : public std::binary_function<keys_t, keys_t, bool>
{
bool operator()(const keys_t& v0, const keys_t& v1) const
{
return (
std::get<0>(v0) == std::get<0>(v1) &&
std::get<1>(v0) == std::get<1>(v1) &&
std::get<2>(v0) == std::get<2>(v1)
);
}
};

typedef std::unordered_map<const keys_t,std::string,key_hash,key_equal> map_t;

map_t stored_jit;

void mddft(int x, int y, int z, hipDeviceptr_t Y, hipDeviceptr_t X) {
    std::cout << "Entered mddft fftx api call" << std::endl;
    hipDeviceptr_t dsym;
    hipMalloc((void **)&dsym,  1* sizeof(std::complex<double>));
    std::vector<void*> args{Y,X, dsym};
    std::vector<int> sizes{x,y,z};
    if(stored_jit.find(std::make_tuple(x,y,z)) != stored_jit.end()) {
        std::cout << "running cached instance" << std::endl;
        Executor e;
        e.execute(stored_jit.at(std::make_tuple(x,y,z)));
    }
    else {
        MDDFTProblem mdp(args, sizes);
        mdp.transform();
        stored_jit[std::make_tuple(x,y,z)] = mdp.returnJIT();
    }
}