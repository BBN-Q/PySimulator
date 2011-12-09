#cython: boundscheck=False
#cython: wraparound=False

from libcpp.vector cimport vector
cimport numpy as np
import numpy as np
from libcpp cimport bool
from cython.operator cimport dereference as deref

from libc.stdlib cimport malloc, free


cdef extern from "CPPBackEnd.h":
    void evolution_unitary_CPP(int numControlLines, int numTimeSteps, int dim, double * Hnat, double * timeSteps, complex ** Hcontrols, complex * totU)

def Cy_evolution_unitary(pulseSequence, systemParams):
    
    #Some error checking
    assert pulseSequence.numControlLines==systemParams.numControlHams, 'Oops! We need the same number of control Hamiltonians as control lines.'
    
    #Initialize the total unitary
    cdef np.ndarray totU = np.eye(systemParams.dim, dtype=np.complex128)
    cdef np.ndarray timeSteps = pulseSequence.timeSteps
    cdef np.ndarray Hnat = systemParams.Hnat.matrix
    cdef complex **Hcontrols
    Hcontrols = <complex**> malloc(2*pulseSequence.numControlLines*sizeof(complex*))
    cdef int ct
    for ct in range(pulseSequence.numControlLines):
        Hcontrols[2*ct] = <complex*> np.PyArray_DATA(systemParams.controlHams[ct]['inphase'].matrix)
        Hcontrols[2*ct+1] = <complex*> np.PyArray_DATA(systemParams.controlHams[ct]['quadrature'].matrix)
            
    evolution_unitary_CPP(pulseSequence.numControlLines, pulseSequence.numTimeSteps, systemParams.dim, <double *> Hnat.data, <double *> timeSteps.data, Hcontrols, <complex *> totU.data    )
    return totU
