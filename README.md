# PySimulator

A python framework for qubit simulations and optimal control.  This code provides classes to assist in setting up Hamiltonians for multi-qubit superconducting qubit simulations.  After setting up the system there is both a pure python and a C++ implementation of an open and closed system simulator and an optimal control module based on the GRAPE algorithm.  The C++ backend relies on the [Eigen](http://eigen.tuxfamily.org) library for matrix manipulations and eignsolvers.  

## Dependencies

These are the latest versions I have worked with. Nearby versions should be just fine too. 

* Python 2.7.4 
* numpy 1.9 
* scipy 0.13
* Cython 0.20 (for C++ backend) (note Cython 0.16-0.19 had a bug that broke assigning to std::vector)
* Eigen 3.2 (for C++ backend)
 
## Building C++ Backend

The pure python implementation should always work as a fall back.  However, particularly for small systems, the C++ back-end can be significantly faster.  For better or worse, the build script is written in scons. You must pass it the path to the eigen install. 

```bash
cd PySim
#Clean any old build files
scons -c
#Build
scons EIGENDIR=/path/to/eigen
```

## Examples

The [SimulatorTests.py](tests/SimulatorTests.py) in the tests folder gives some ideas of how to get going. 

More examples to come...

