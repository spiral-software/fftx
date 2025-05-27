//
//  Copyright (c) 2018-2025, Carnegie Mellon University
//  All rights reserved.
//
//  See LICENSE file for full information.
//

#ifndef FFTX_COMMON_UTIL_H
#define FFTX_COMMON_UTIL_H

#include <mpi.h>

// using namespace std;

inline double max_diff(double start, double end, const MPI_Comm&  comm) {
  double my_time = end - start;
  double max_time = my_time;
  MPI_Reduce(&my_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);  
  return max_time;
}

inline double min_diff(double start, double end, const MPI_Comm& comm) {
  double my_time = end - start;
  double min_time = my_time;
  MPI_Reduce(&my_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
  return min_time;
}

 #endif
