
If you are building on a supercomputing platform at NERSC or OLCF,
compilation requires having the appropriate modules loaded.

* #### On NERSC system **perlmutter**:
```
module purge
module load cmake cudatoolkit PrgEnv-gnu 
export LIBRARY_PATH=$CUDATOOLKIT_HOME/../../math_libs/lib64
export CPATH=$CUDATOOLKIT_HOME/../../math_libs/include
```

* #### On OLCF system **spock** or **crusher** or **frontier**:
```
module purge
module load rocm
```
