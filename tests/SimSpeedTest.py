# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:44:22 2012

@author: cryan
"""

import numpy as np
from numpy import sin, cos

from scipy.constants import pi
from scipy.linalg import expm, eigh

from PySim.SystemParams import SystemParams
from PySim.PulseSequence import PulseSequence
from PySim.Simulation import simulate_sequence_stack, simulate_sequence
from PySim.QuantumSystems import SCQubit, Hamiltonian, Dissipator

from numba import *

import matplotlib.pyplot as plt
from timeit import timeit

#Try to load the CPPBackEnd
try:
    import PySim.CySim
    CPPBackEnd = True
except ImportError:
    CPPBackEnd = False
    
#@jit(c16[:,:](c16[:,:], c16))
def expm_eigen(matIn, mult):
    '''
    Helper function to compute matrix exponential of Hermitian matrix
    '''
    dim = matIn.shape[0]
    D, V = eigh(matIn)
    return V.dot(np.diag(np.exp(mult*D))).dot(V.conj().T)

#@jit(c16[:,:](c16[:,:], c16[:,:,:], f8[:,:], f8[:]))
#@autojit
def evolution_numpy(Hnat, controlHams, controlFields, controlFreqs):
    
    timeStep = 0.01
    curTime = 0.0
    
    Uprop = np.eye(Hnat.shape[0])
    for timect in range(controlFields.shape[1]):
        tmpH = np.copy(Hnat)
        for controlct in range(controlFields.shape[0]):
            tmpH += controlFields[controlct, timect]*cos(2*pi*curTime*controlFreqs[controlct])*controlHams[controlct]
        
        Uprop = np.dot(expm_eigen(tmpH,-1j*2*pi*timeStep)[0],Uprop)
        curTime += timeStep
        
    return Uprop

@autojit()
def sum1d(my_double_array):
    sum = 0.0
    for i in range(my_double_array.shape[0]):
        sum += my_double_array[i]
    return sum

def sim_setup(dimension, numTimeSteps, numControls):
    #Create a random natural hamiltonian 
    tmpMat = np.random.randn(dimension, dimension) + 1j*np.random.randn(dimension, dimension)
    Hnat = tmpMat+tmpMat.conj().T

    #Create random control Hamiltonians
    controlHams = np.zeros((numControls,dimension, dimension), dtype=np.complex128)
    for ct in range(numControls):
        tmpMat = np.random.randn(dimension, dimension) + 1j*np.random.randn(dimension, dimension)
        controlHams[ct] = tmpMat+tmpMat.conj().T
        
    #Create random controlfields
    controlFields = np.random.randn(numControls, numTimeSteps)
        
    #Control frequencies
    controlFreqs = np.random.randn(numControls)

    return Hnat, controlHams, controlFields, controlFreqs

def sim_setup_cython(Hnat, controlHams, controlFields, controlFreqs):
    systemParams = SystemParams()
    systemParams.Hnat = Hamiltonian(Hnat)
    pulseSeq = PulseSequence()
    pulseSeq.controlAmps = controlFields
    for ct in range(len(controlHams)):
        systemParams.add_control_ham(inphase=Hamiltonian(controlHams[ct]))
        pulseSeq.add_control_line(freq = controlFreqs[ct], phase=0, controlType='sinusoidal')
    for ct in range(np.int(np.log2(Hnat.shape[0]))):
        systemParams.add_sub_system(SCQubit(2,0e9, name='Q1', T1=1e-6))
    pulseSeq.timeSteps = 0.01*np.ones(controlFields.shape[1])
    pulseSeq.maxTimeStep = 1e6
    
    return systemParams, pulseSeq
    
         
if __name__ == '__main__':
    
    dims = 2**np.arange(1,6)
    cythonTimes = []
    numpyTimes = []
    for dim in dims:
        print(dim)
        Hnat, controlHams, controlFields, controlFreqs = sim_setup(dim, 2000, 4)
        systemParams, pulseSeq = sim_setup_cython(Hnat, controlHams, controlFields, controlFreqs)
        numpyTimes.append(timeit('evolution_numpy(Hnat, controlHams, controlFields, controlFreqs)', 
                      setup='from __main__ import evolution_numpy, Hnat, controlHams, controlFields, controlFreqs', number=3)/3)
        cythonTimes.append(timeit('simulate_sequence(pulseSeq, systemParams)', setup='from __main__ import simulate_sequence, pulseSeq, systemParams', number=3)/3)
        
    plt.plot(dims, numpyTimes)
    plt.plot(dims, cythonTimes)
    plt.legend(('Numpy', 'Cython'))
    plt.xlabel('System Dimension')
    plt.show()
    
    

