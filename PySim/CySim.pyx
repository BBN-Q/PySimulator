#cython: boundscheck=False
#cython: wraparound=False

from libcpp.vector cimport vector
cimport numpy as np
import numpy as np
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free

#Load some classes and functions from the C++ backend.  We use these classes for passing data back and forth between Python and C++
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
        vector[complex *] dissipatorPtrs
        
    cdef cppclass OptimParams(PulseSequence):
        OptimParams(complex *, complex *, complex *, size_t, size_t)
        size_t dimC2
        int derivType
        int optimType
        
    cdef cppclass PropResults:
        PropResults(size_t, size_t)
        

    void evolve_propagator_CPP(PulseSequence, SystemParams, int, complex * )

    void opt_evolve_propagator_CPP(OptimParams, SystemParams, complex ***, PropResults)

    void eval_derivs(OptimParams, SystemParams, complex ***, PropResults, double *)

    double eval_pulse_fitness(OptimParams, PropResults)



#Python versions of the classes that will be accessible from Python. 
#Basically we pass the class a Python version of itself and it initializes a C++ version that can be passed to
#the C++ back end.
cdef class PyPulseSequence(object):
    cdef PulseSequence *thisPtr 
    def __cinit__(self, pulseSeqIn):
        self.thisPtr = new PulseSequence()
        self.thisPtr.numControlLines = pulseSeqIn.numControlLines
        self.thisPtr.numTimeSteps = pulseSeqIn.numTimeSteps
        self.thisPtr.timeStepsPtr = <double *> np.PyArray_DATA(pulseSeqIn.timeSteps)
        self.thisPtr.maxTimeStep = pulseSeqIn.maxTimeStep
        #Error check for data ordering
        assert pulseSeqIn.controlAmps.flags['C_CONTIGUOUS'], "Uhoh! We need row-major ordering for controlAmps for passing data to C++. Use np.copy(order='C')."
        self.thisPtr.controlAmpsPtr = <double *> np.PyArray_DATA(pulseSeqIn.controlAmps)
        self.thisPtr.controlLines.resize(pulseSeqIn.numControlLines)
        for ct in range(pulseSeqIn.numControlLines):
            self.thisPtr.controlLines[ct].freq = pulseSeqIn.controlLines[ct].freq
            self.thisPtr.controlLines[ct].phase = pulseSeqIn.controlLines[ct].phase
            self.thisPtr.controlLines[ct].controlType = 1 if pulseSeqIn.controlLines[ct].controlType=='rotating' else 0
        #Error check for data ordering
        if pulseSeqIn.H_int is not None: 
            assert pulseSeqIn.H_int.matrix.flags['C_CONTIGUOUS'], "Uhoh! We need row-major ordering for H_int for passing data to C++. Use np.copy(order='C')."
        self.thisPtr.H_intPtr = <complex *> np.PyArray_DATA(pulseSeqIn.H_int.matrix) if pulseSeqIn.H_int is not None else NULL
    def __dealloc__(self):
        del self.thisPtr

    #Setter to update the control amplitudes
    property controlAmps:
        def __set__(self, controlAmps):
            #Error check for data ordering
            assert controlAmps.flags['C_CONTIGUOUS'], "Uhoh! We need row-major ordering for controlAmps for passing data to C++. Use np.copy(order='C')."
            self.thisPtr.controlAmpsPtr = <double *> np.PyArray_DATA(controlAmps)

        
cdef class PySystemParams(object):
    cdef SystemParams *thisPtr
    def __cinit__(self, systemParamsIn):
        self.thisPtr = new SystemParams()
        #Error check for data ordering
        assert systemParamsIn.Hnat.matrix.flags['C_CONTIGUOUS'], "Uhoh! We need row-major ordering for Hnat for passing data to C++. Use np.copy(order='C')."
        self.thisPtr.HnatPtr = <complex *> np.PyArray_DATA(systemParamsIn.Hnat.matrix)
        self.thisPtr.numControlHams = systemParamsIn.numControlHams
        self.thisPtr.dim = systemParamsIn.dim
        self.thisPtr.controlHams.resize(systemParamsIn.numControlHams)
        for ct in range(systemParamsIn.numControlHams):
            #Error check for data ordering
            assert systemParamsIn.controlHams[ct]['inphase'].matrix.flags['C_CONTIGUOUS'], "Uhoh! We need row-major ordering for inphase controlHams for passing data to C++. Use np.copy(order='C')."
            self.thisPtr.controlHams[ct].inphasePtr = <complex*> np.PyArray_DATA(systemParamsIn.controlHams[ct]['inphase'].matrix)
            #Error check for data ordering
            if systemParamsIn.controlHams[ct]['quadrature'] is not None:
                assert systemParamsIn.controlHams[ct]['quadrature'].matrix.flags['C_CONTIGUOUS'], "Uhoh! We need row-major ordering for quadrature controlHams for passing data to C++. Use np.copy(order='C')."
            self.thisPtr.controlHams[ct].quadraturePtr = <complex*> np.PyArray_DATA(systemParamsIn.controlHams[ct]['quadrature'].matrix) if systemParamsIn.controlHams[ct]['quadrature'] is not None else NULL

#        self.thisPtr.dissipatorPtrs.resize(len(systemParamsIn.dissipators))
#        for ct in range(len(systemParamsIn.dissipators)):
#            #Error check for data ordering
#            assert systemParamsIn.dissipators[ct].matrix.flags['C_CONTIGUOUS'], "Uhoh! We need row-major ordering for dissipators for passing data to C++. Use np.copy(order='C')."
#            self.thisPtr.dissipatorPtrs[ct] = <complex*> np.PyArray_DATA(systemParamsIn.dissipators[ct].matrix)
 
    def __dealloc__(self):
        del self.thisPtr


#Hold pointer to the interaction frame control Hamiltonians 
#TODO: This is ugly.  Better to have a vector<vector<Mapcd>>  
cdef class PyControlHams_int(object):
    cdef complex *** dataPtrs
    cdef size_t dim
    cdef size_t numControlHams
    cdef size_t numTimeSteps
    def __init__(self, controlHams_int):
        #Assume we get a 4D array of the control Hams in
        self.dim = controlHams_int.shape[2]
        self.numControlHams = controlHams_int.shape[0]
        self.numTimeSteps = controlHams_int.shape[1]
         
        #Allocate memory.  We have a 2D array of pointers but that is all i.e. we don't have to allocate memory for the actual arrays
        #as we'll map them with Eign 
        self.dataPtrs = < complex ***> malloc(self.numControlHams * sizeof(complex *))
        for controlct in range(self.numControlHams):
            self.dataPtrs[controlct] = <complex **> malloc(self.numTimeSteps * sizeof(complex *))
            for timect in range(self.numTimeSteps):
                #Error check for data ordering
                assert controlHams_int[controlct, timect].flags['C_CONTIGUOUS'], "Uhoh! We need row-major ordering for controlHams_int for passing data to C++. Use np.copy(order='C')."
                self.dataPtrs[controlct][timect] = <complex*> np.PyArray_DATA(controlHams_int[controlct, timect])
                
    def __dealloc__(self):
        for controlct in range(self.numControlHams):
            free(self.dataPtrs[controlct])
        
        
#Hold some references to the optimization 
#It should somehow be possible to subclass PyPulseSequence here but then I can't redefine thisPtr
#cdef class PyOptimParams(PyPulseSequence):
cdef class PyOptimParams(object):
    cdef OptimParams *thisPtr
    def __cinit__(self, optimParamsIn):
        #Error check for data ordering
        if optimParamsIn.Ugoal is not None:
            assert optimParamsIn.Ugoal.flags['C_CONTIGUOUS'], "Uhoh! We need row-major ordering for Ugoal for passing data to C++. Use np.copy(order='C')."
        if optimParamsIn.rhoStart is not None:
            assert optimParamsIn.rhoStart.flags['C_CONTIGUOUS'], "Uhoh! We need row-major ordering for rhoStart for passing data to C++. Use np.copy(order='C')."
        if optimParamsIn.rhoGoal is not None:         
            assert optimParamsIn.rhoGoal.flags['C_CONTIGUOUS'], "Uhoh! We need row-major ordering for rhoGoal for passing data to C++. Use np.copy(order='C')."
        self.thisPtr = new OptimParams(<complex *> np.PyArray_DATA(optimParamsIn.Ugoal), <complex*> np.PyArray_DATA(optimParamsIn.rhoStart), <complex*> np.PyArray_DATA(optimParamsIn.rhoGoal), optimParamsIn.dim, optimParamsIn.dimC2)    
        self.thisPtr.numControlLines = optimParamsIn.numControlLines
        self.thisPtr.numTimeSteps = optimParamsIn.numTimeSteps
        self.thisPtr.timeStepsPtr = <double *> np.PyArray_DATA(optimParamsIn.timeSteps)
        self.thisPtr.maxTimeStep = optimParamsIn.maxTimeStep
        if optimParamsIn.controlAmps is not None:
            assert optimParamsIn.controlAmps.flags['C_CONTIGUOUS'], "Uhoh! We need row-major ordering for controlAmps for passing data to C++. Use np.copy(order='C')."
            self.thisPtr.controlAmpsPtr = <double *> np.PyArray_DATA(optimParamsIn.controlAmps)
        self.thisPtr.controlLines.resize(optimParamsIn.numControlLines)
        for ct in range(optimParamsIn.numControlLines):
            self.thisPtr.controlLines[ct].freq = optimParamsIn.controlLines[ct].freq
            self.thisPtr.controlLines[ct].phase = optimParamsIn.controlLines[ct].phase
        self.thisPtr.controlLines[ct].controlType = 1 if optimParamsIn.controlLines[ct].controlType=='rotating' else 0
        #Error check for data ordering
        if optimParamsIn.H_int is not None: 
            assert optimParamsIn.H_int.matrix.flags['C_CONTIGUOUS'], "Uhoh! We need row-major ordering for H_int for passing data to C++. Use np.copy(order='C')."
        self.thisPtr.H_intPtr = <complex *> np.PyArray_DATA(optimParamsIn.H_int.matrix) if optimParamsIn.H_int is not None else NULL
        derivTypeMap = {'finiteDiff':0, 'approx':1, 'exact':2}
        self.thisPtr.derivType = derivTypeMap[optimParamsIn.derivType]
        optimTypeMap = {'unitary':0, 'state2state':1}
        self.thisPtr.optimType = optimTypeMap[optimParamsIn.optimType]

    def __dealloc__(self):
        del self.thisPtr   

    #Setter to update the control amplitudes
    property controlAmps:
        def __set__(self, controlAmps):
            #Error check for data ordering
            assert controlAmps.flags['C_CONTIGUOUS'], "Uhoh! We need row-major ordering for controlAmps for passing data to C++. Use np.copy(order='C')."
            self.thisPtr.controlAmpsPtr = <double *> np.PyArray_DATA(controlAmps)

#This class basically holds all the propagator evolution results for the optimization so that 
#we don't have to reallocate memory each iteration.   
cdef class PyPropResults:
    cdef PropResults *thisPtr
    def __cinit__(self, numTimeSteps, dim):
        self.thisPtr = new PropResults(numTimeSteps, dim)
    def __dealloc__(self):
        del self.thisPtr                                                        
    

#Pass-thru function to evaluate the goodness of a pulse
def Cy_eval_pulse(PyOptimParams optimParamsIn, PySystemParams systemParamsIn, PyControlHams_int controlHams_int, PyPropResults propResults):
    #Pass everything through to the C++ function
    opt_evolve_propagator_CPP(deref(optimParamsIn.thisPtr), deref(systemParamsIn.thisPtr), controlHams_int.dataPtrs, deref(propResults.thisPtr))
    
    #Calculate the goodness
    return eval_pulse_fitness(deref(optimParamsIn.thisPtr), deref(propResults.thisPtr))

#Pass-thru function to evaluate the derivatives of a pulse
def Cy_eval_derivs(PyOptimParams optimParamsIn, PySystemParams systemParamsIn, PyControlHams_int controlHams_int, PyPropResults propResults):
    #Allocate space for the derivatives
    derivs = np.zeros((controlHams_int.numControlHams, controlHams_int.numTimeSteps), dtype=np.float64) 
    
    #Pass on to the C++ function
    eval_derivs(deref(optimParamsIn.thisPtr), deref(systemParamsIn.thisPtr), controlHams_int.dataPtrs, deref(propResults.thisPtr), <double*> np.PyArray_DATA(derivs))
            
    return -derivs.flatten()

#Pass-thru function to evaluate the evolution propagator for either unitary or lindblad.
def Cy_evolution(pulseSeqIn, systemParamsIn, simType):
    
    #Some error checking
    assert pulseSeqIn.numControlLines==systemParamsIn.numControlHams, 'Oops! We need the same number of control Hamiltonians as control lines.'
    
    pulseSeq = PyPulseSequence(pulseSeqIn)
    
    systemParams = PySystemParams(systemParamsIn)
            
    #Initialize the total unitary output memory to the identity
    cdef np.ndarray totProp
    if simType == 'unitary':
        totProp = np.eye(systemParamsIn.dim, dtype=np.complex128)
        evolve_propagator_CPP(deref(pulseSeq.thisPtr), deref(systemParams.thisPtr), 0, <complex *> totProp.data )
    elif simType == 'lindblad':
        totProp = np.eye(systemParamsIn.dim**2, dtype=np.complex128)
        evolve_propagator_CPP(deref(pulseSeq.thisPtr), deref(systemParams.thisPtr), 1, <complex *> totProp.data )
    
    return totProp


