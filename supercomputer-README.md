
If you are building on a supercomputing platform at NERSC or OLCF or ALCF,
compilation requires having the appropriate modules loaded.

* #### On **perlmutter** system at National Energy Research Scientific Computing Center (NERSC):
```
module load gcc-native/13.2
module load python
export CPATH=$CRAY_MPICH_DIR/include:$CPATH
export MPICH_GPU_SUPPORT_ENABLED=0
```

* #### On **frontier** system at Oak Ridge Leadership Computing Facility (OLCF):
```
module purge
module load rocm
module load PrgEnv-gnu
module load python
```

* #### On **sunspot** system at Argonne Leadership Computing Facility (ALCF):
```
module use /soft/modulefiles
module load cmake
module load python
export ONEAPI_DEVICE_SELECTOR=opencl:gpu
```

* #### On **aurora** system at Argonne Leadership Computing Facility (ALCF):
```
module use /soft/modulefiles
module load spack-pe-gcc
module load cmake
module load python
module load oneapi
export ONEAPI_DEVICE_SELECTOR=opencl:gpu
```
