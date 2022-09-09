
If you are building on a supercomputing platform at NERSC or OLCF,
compilation requires having the appropriate modules loaded.

* #### On **cori** at NERSC:
```
module purge
module load cgpu cuda PrgEnv-gnu craype-haswell
```

* #### On **perlmutter** at NERSC:
```
module purge
module load cmake cudatoolkit PrgEnv-gnu 
export LIBRARY_PATH=$CUDATOOLKIT_HOME/../../math_libs/lib64
export CPATH=$CUDATOOLKIT_HOME/../../math_libs/include
```

* #### On **spock** at OLCF:
```
module purge
module load rocm
```

* #### On **crusher** at OLCF:
```
module purge
module load rocm
```

