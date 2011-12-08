#cython: boundscheck=False
#cython: wraparound=False

from libcpp.vector cimport vector
cimport numpy as np
import numpy as np
from libcpp cimport bool
from cython.operator cimport dereference as deref

from libc.stdlib cimport malloc, free


cdef extern from "CPPBackEnd.h":
    void evolution_unitary(int numControlLines, int numTimeSteps, int dim, double * Hnat, double * timeSteps, double ** Hcontrols, complex * totU)

def Cy_evolution_unitary(pulseSequence, systemParams):
    
    #Some error checking
    assert pulseSequence.numControlLines==systemParams.numControlHams, 'Oops! We need the same number of control Hamiltonians as control lines.'
    
    #Initialize the total unitary
    cdef np.ndarray totU = np.eye(systemParams.dim, dtype=np.complex128)
    cdef np.ndarray timeSteps = pulseSequence.timeSteps
    cdef np.ndarray Hnat = systemParams.Hnat.matrix
    cdef double **Hcontrols
    Hcontrols = <double**> malloc(4*sizeof(double*))
    cdef int ct
    for ct in range(4):
        Hcontrols[ct] = <double*> malloc(4*sizeof(double))   
    evolution_unitary(pulseSequence.numControlLines, pulseSequence.numTimeSteps, systemParams.dim, <double *> Hnat.data, <double *> timeSteps.data, Hcontrols, <complex *> totU.data    )
    return totU
