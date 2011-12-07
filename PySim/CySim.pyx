#cython: boundscheck=False
#cython: wraparound=False

from libcpp.vector cimport vector
cimport numpy as np
import numpy as np
from libcpp cimport bool
from cython.operator cimport dereference as deref

from libc.stdlib cimport malloc, free


cdef extern from "CPPBackEnd.h":
#    void evolution_unitary(int numControlLines, int numTimeSteps, int dim, double * Hnat, double * timeSteps, double ** Hcontrols, complex * totU)
    void evolution_unitary(complex *)

def Cy_evolution_unitary(pulseSequence, systemParams):
    
    #Some error checking
    assert pulseSequence.numControlLines==systemParams.numControlHams, 'Oops! We need the same number of control Hamiltonians as control lines.'
    
    #Initialize the total unitary
    cdef np.ndarray totU = np.eye(systemParams.dim, dtype=np.complex128)
    cdef double **Hcontrols
#    evolution_unitary(pulseSequence.numControlLines, pulseSequence.numTimeSteps, systemParams.dim, <double *> systemParams.Hnat.matrix.data, <double *> pulseSequence.timeSteps, Hcontrols, <complex *> totU.data    )
    evolution_unitary(<complex*> totU.data)    
    return totU
