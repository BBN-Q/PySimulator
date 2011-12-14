#cython: boundscheck=False
#cython: wraparound=False

from libcpp.vector cimport vector
cimport numpy as np
import numpy as np
from cython.operator cimport dereference as deref

cdef extern from "CPPBackEnd.h":
    cdef cppclass ControlLine:
        double freq
        double phase
        int controlType
        
    cdef cppclass PulseSequence:
        size_t numControlLines
        size_t numTimeSteps
        double * timeStepsPtr
        double maxTimeStep
        double * controlAmpsPtr
        vector[ControlLine] controlLines
        complex * H_intPtr
    
    cdef cppclass ControlHam:
        complex * inphasePtr
        complex * quadraturePtr
    
    cdef cppclass SystemParams:
        size_t numControlHams
        size_t dim
        vector[ControlHam] controlHams
        complex * HnatPtr

    void evolution_unitary_CPP(PulseSequence , SystemParams, complex * totU)

def Cy_evolution_unitary(pulseSeqIn, systemParamsIn):
    
    #Some error checking
    assert pulseSeqIn.numControlLines==systemParamsIn.numControlHams, 'Oops! We need the same number of control Hamiltonians as control lines.'
    
    cdef int ct
    
    cdef PulseSequence *pulseSeq = new PulseSequence()
    pulseSeq.numControlLines = pulseSeqIn.numControlLines
    pulseSeq.numTimeSteps = pulseSeqIn.numTimeSteps
    pulseSeq.timeStepsPtr = <double *> np.PyArray_DATA(pulseSeqIn.timeSteps)
    pulseSeq.maxTimeStep = pulseSeqIn.maxTimeStep
    pulseSeq.controlAmpsPtr = <double *> np.PyArray_DATA(pulseSeqIn.controlAmps)
    pulseSeq.controlLines.resize(pulseSeq.numControlLines)
    for ct in range(pulseSeq.numControlLines):
        pulseSeq.controlLines[ct].freq = pulseSeqIn.controlLines[ct].freq
        pulseSeq.controlLines[ct].phase = pulseSeqIn.controlLines[ct].phase
        if pulseSeqIn.controlLines[ct].controlType=='rotating':
            pulseSeq.controlLines[ct].controlType = 1
        else:
            pulseSeq.controlLines[ct].controlType = 0
    pulseSeq.H_intPtr = <complex *> np.PyArray_DATA(pulseSeqIn.H_int.matrix) if pulseSeqIn.H_int is not None else NULL
            

    cdef SystemParams *systemParams = new SystemParams()    
    systemParams.HnatPtr = <complex *> np.PyArray_DATA(systemParamsIn.Hnat.matrix)
    systemParams.numControlHams = systemParamsIn.numControlHams
    systemParams.dim = systemParamsIn.dim
    systemParams.controlHams.resize(systemParams.numControlHams)
    for ct in range(pulseSeq.numControlLines):
        systemParams.controlHams[ct].inphasePtr = <complex*> np.PyArray_DATA(systemParamsIn.controlHams[ct]['inphase'].matrix)
        systemParams.controlHams[ct].quadraturePtr = <complex*> np.PyArray_DATA(systemParamsIn.controlHams[ct]['quadrature'].matrix) if systemParamsIn.controlHams[ct]['quadrature'] is not None else NULL
            
    #Initialize the total unitary output memory to the identity
    cdef np.ndarray totU = np.eye(systemParamsIn.dim, dtype=np.complex128)

    evolution_unitary_CPP(deref(pulseSeq), deref(systemParams), <complex *> totU.data    )
    return totU
