'''
Created on Nov 6, 2011

@author: cryan

Rough script for testing the python simulator
'''

import numpy as np

import matplotlib.pyplot as plt

from SystemParams import SystemParams
from Hamiltonian import Hamiltonian
from PulseSequence import PulseSequence
from Simulation import simulate_sequence_stack

if __name__ == '__main__':
    
    #Setup the system
    systemParams = SystemParams()
    systemParams.Hnat = Hamiltonian(2*np.pi*np.array([[0e9,0], [0, 5e9]], dtype = np.complex128))
    systemParams.add_control_ham(inphase = Hamiltonian(2*np.pi*0.5*np.array([[0,1],[1,0]], dtype = np.complex128)), quadrature = Hamiltonian(2*np.pi*0.5*np.array([[0,-1j],[1j,0]], dtype = np.complex128)))
    systemParams.dim = 2
    systemParams.measurement = np.array([[1,0],[0,-1]])
    
    #Setup the pulseSequences
    pulseSeqs = []
    for delayLength in np.linspace(0,100e-9,101):
    
        tmpPulseSeq = PulseSequence()
        tmpPulseSeq.add_control_line(freq=5.01e9, initialPhase=0)
        tmpPulseSeq.controlAmps = 10e6*np.array([[1, 0, -1]], dtype=np.float64)
        tmpPulseSeq.timeSteps = np.array([25e-9, delayLength, 25e-9])
        tmpPulseSeq.maxTimeStep = np.Inf
        tmpPulseSeq.H_int = Hamiltonian(2*np.pi*np.array([[0,0], [0, 5.003e9]], dtype = np.complex128))
        
        pulseSeqs.append(tmpPulseSeq)
        
    results = simulate_sequence_stack(pulseSeqs, systemParams, np.array([[1,0],[0,0]]), simType='unitary')
    
    plt.figure()
    plt.plot(results)
    plt.show()
    
    
    
