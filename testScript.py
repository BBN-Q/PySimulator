'''
Created on Nov 6, 2011

@author: cryan

Rough script for testing the python simulator
'''

import numpy as np

import matplotlib.pyplot as plt

from SystemParams import SystemParams
from QuantumSystems import Hamiltonian, Dissipator
from PulseSequence import PulseSequence
from Simulation import simulate_sequence_stack, simulate_sequence
from QuantumSystems import SCQubit

if __name__ == '__main__':
    
    #Setup the system
    systemParams = SystemParams()
    systemParams.Hnat = Hamiltonian(2*np.pi*np.array([[0,0], [0,0]], dtype = np.complex128))
    systemParams.add_control_ham(inphase = Hamiltonian(2*np.pi*0.5*np.array([[0,1],[1,0]], dtype = np.complex128)), quadrature = Hamiltonian(2*np.pi*0.5*np.array([[0,-1j],[1j,0]], dtype = np.complex128)))
    systemParams.dim = 2
    systemParams.measurement = np.array([[1,0],[0,-1]])
    
    #Setup the Hermite polynomial 
    numPoints = 240
    x = np.linspace(-2,2,numPoints)
    #Hermite 180
#    HermiteAmps = (1-0.956*x**2)*np.exp(-(x**2))
    #Hermite 90
    HermiteAmps = (1-0.677*x**2)*np.exp(-(x**2))
    
    #Setup the pulseSequences
    pulseSeqs = []
#    controlAmp = 1.2/0.9
    freqs = np.linspace(-20e6,20e6,100)
    controlAmp = 0.568/1.333
    for offRes in freqs:
    
        tmpPulseSeq = PulseSequence()
        tmpPulseSeq.add_control_line(freq=-offRes, initialPhase=0)
        tmpPulseSeq.controlAmps = (10e6*controlAmp*HermiteAmps).reshape((1,numPoints))
        tmpPulseSeq.timeSteps = (1/1.2e9)*np.ones(numPoints)
        tmpPulseSeq.maxTimeStep = np.Inf
        tmpPulseSeq.H_int = None
        
        pulseSeqs.append(tmpPulseSeq)
        
    systemParams.measurement = np.array([[0,1],[1,0]])
    resultsX = simulate_sequence_stack(pulseSeqs, systemParams, np.array([[1,0],[0,0]]), simType='unitary')
    systemParams.measurement = np.array([[0,-1j],[1j,0]])
    resultsY = simulate_sequence_stack(pulseSeqs, systemParams, np.array([[1,0],[0,0]]), simType='unitary')
    
    
    plt.figure()
    plt.plot(freqs/1e6,np.sqrt(resultsX**2 + resultsY**2))
    plt.show()
    
    
    
