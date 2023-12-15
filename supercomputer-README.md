
If you are building on a supercomputing platform at NERSC or OLCF,
compilation requires having the appropriate modules loaded.

* #### On **perlmutter** system at National Energy Research Scientific Computing Center (NERSC):
```
module purge
module load cmake cudatoolkit PrgEnv-gnu
export LIBRARY_PATH=$CUDATOOLKIT_HOME/../../math_libs/lib64
export CPATH=$CUDATOOLKIT_HOME/../../math_libs/include
module load openmpi
module load python
```

* #### On **frontier** or **crusher** system at Oak Ridge Leadership Computing Facility (OLCF):
```
module purge
module load rocm
module load PrgEnv-cray
module load cray-python
```

* #### On **aurora** system at Argonne Leadership Computing Facility (ALCF):
```
module use /soft/modulefiles
module load spack-pe-gcc
module load cmake
module load python
module load PrgEnv-gnu
export ONEAPI_DEVICE_SELECTOR=opencl:gpu
```
