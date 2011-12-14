#cython: boundscheck=False
#cython: wraparound=False

from libcpp.vector cimport vector
cimport numpy as np
import numpy as np
from libcpp cimport bool
from cython.operator cimport dereference as deref

from libc.stdlib cimport malloc, free


cdef extern from "CPPBackEnd.h":
    cdef cppclass PulseSequence:
        size_t numControlLines
        size_t numTimeSteps
        double * timeSteps
    
    cdef cppclass ControlHam:
        complex * inphase
        complex * quadrature
    
    cdef cppclass SystemParams:
        size_t numControlHams
        size_t dim
        vector[ControlHam] controlHams
        complex * Hnat

    void evolution_unitary_CPP(PulseSequence , SystemParams, complex * totU)

def Cy_evolution_unitary(pulseSeqIn, systemParamsIn):
    
    #Some error checking
    assert pulseSeqIn.numControlLines==systemParamsIn.numControlHams, 'Oops! We need the same number of control Hamiltonians as control lines.'
    
    cdef PulseSequence *pulseSeq = new PulseSequence()
    pulseSeq.numControlLines = pulseSeqIn.numControlLines
    pulseSeq.numTimeSteps = pulseSeqIn.numTimeSteps
    pulseSeq.timeSteps = <double *> np.PyArray_DATA(pulseSeqIn.timeSteps)

    cdef SystemParams *systemParams = new SystemParams()    
    systemParams.Hnat = <complex *> np.PyArray_DATA(systemParamsIn.Hnat.matrix)
    systemParams.numControlHams = systemParamsIn.numControlHams
    systemParams.dim = systemParamsIn.dim
    cdef int ct
    systemParams.controlHams.resize(pulseSeq.numControlLines)
    for ct in range(pulseSeq.numControlLines):
        systemParams.controlHams[ct].inphase = <complex*> np.PyArray_DATA(systemParamsIn.controlHams[ct]['inphase'].matrix)
        systemParams.controlHams[ct].quadrature = <complex*> np.PyArray_DATA(systemParamsIn.controlHams[ct]['quadrature'].matrix)
            
    #Initialize the total unitary output memory to the identity
    cdef np.ndarray totU = np.eye(systemParamsIn.dim, dtype=np.complex128)

    evolution_unitary_CPP(deref(pulseSeq), deref(systemParams), <complex *> totU.data    )
    return totU
