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
    
#    #Setup the system
#    systemParams = SystemParams()
#    systemParams.Hnat = Hamiltonian(2*np.pi*np.array([[0,0], [0,0]], dtype = np.complex128))
#    systemParams.add_control_ham(inphase = Hamiltonian(2*np.pi*0.5*np.array([[0,1],[1,0]], dtype = np.complex128)), quadrature = Hamiltonian(2*np.pi*0.5*np.array([[0,-1j],[1j,0]], dtype = np.complex128)))
#    systemParams.dim = 2
#    systemParams.measurement = np.array([[1,0],[0,-1]])
#    
#    #Setup the Hermite polynomial 
#    numPoints = 240
#    AWGFreq = 1.2e9
#    x = np.linspace(-2,2,numPoints)
#    #Hermite 180
##    pulseAmps = (1-0.956*x**2)*np.exp(-(x**2))
#    #Hermite 90
#    pulseAmps = (1-0.677*x**2)*np.exp(-(x**2))
#    #Gaussian
##    pulseAmps = np.exp(-(x**2))
#    #Setup the pulseSequences
#    pulseSeqs = []
#    freqs = np.linspace(-20e6,20e6,100)
#    controlScale = 0.25/(np.sum(pulseAmps)*(1/AWGFreq))
#    for offRes in freqs:
#        tmpPulseSeq = PulseSequence()
#        tmpPulseSeq.add_control_line(freq=-offRes, initialPhase=0)
#        tmpPulseSeq.controlAmps = (controlScale*pulseAmps).reshape((1,numPoints))
#        tmpPulseSeq.timeSteps = (1/AWGFreq)*np.ones(numPoints)
#        tmpPulseSeq.maxTimeStep = np.Inf
#        tmpPulseSeq.H_int = None
#        
#        pulseSeqs.append(tmpPulseSeq)
#        
#    systemParams.measurement = np.array([[0,1],[1,0]])
#    resultsX = simulate_sequence_stack(pulseSeqs, systemParams, np.array([[1,0],[0,0]]), simType='unitary')
#    systemParams.measurement = np.array([[0,-1j],[1j,0]])
#    resultsY = simulate_sequence_stack(pulseSeqs, systemParams, np.array([[1,0],[0,0]]), simType='unitary')
#    
#    
#    plt.figure()
#    plt.plot(freqs/1e6,np.sqrt(resultsX**2 + resultsY**2))
#    plt.show()
    
    '''  Try to recreate the Bell-Rabi spectroscopy '''
    
    #Setup the system
    systemParams = SystemParams()
    
    #First the two qubits
    Q1 = SCQubit(numLevels=3, omega=4.86359e9-1e6, delta=-300, name='Q1', T1=5.2e-6)
    systemParams.add_sub_system(Q1)
    Q2 = SCQubit(numLevels=3, omega=5.19344e9-1e6, delta=-313.656e6, name='Q2', T1=4.4e-6)
    systemParams.add_sub_system(Q2)
 
    #Add a 2MHz ZZ interaction 
    systemParams.add_interaction('Q1', 'Q2', 'ZZ', 2e6)
   
    #Create the full Hamiltonian   
    systemParams.create_full_Ham()
 
    #Some Pauli operators for the controls
    X = 0.5*(Q1.loweringOp + Q1.raisingOp)
    Y = -0.5*(-1j*Q1.loweringOp + 1j*Q2.raisingOp)
    #The cross-coupling from Q1 drive to Q2
    crossCoupling = 0.5
    
    #Add the Q1 drive Hamiltonians
    systemParams.add_control_ham(inphase = Hamiltonian(systemParams.expand_operator('Q1', X) + crossCoupling*systemParams.expand_operator('Q2', X)),
                                  quadrature = Hamiltonian(systemParams.expand_operator('Q1', Y) + crossCoupling*systemParams.expand_operator('Q2', Y)))
    
    #Setup the measurement operator
#    systemParams.measurement = -systemParams.expand_operator('Q1', Q1.pauliZ) - systemParams.expand_operator('Q2', Q2.pauliZ)
    systemParams.measurement = 0.6057*np.eye(9) + 0.0176*systemParams.expand_operator('Q1', Q1.pauliZ) + 0.0155*systemParams.expand_operator('Q2', Q2.pauliZ) - 0.0074*np.kron(Q1.pauliZ, Q2.pauliZ)

    #Add the T1 dissipators
    systemParams.dissipators.append(Dissipator(systemParams.expand_operator('Q1', Q1.T1Dissipator)))
    systemParams.dissipators.append(Dissipator(systemParams.expand_operator('Q2', Q2.T1Dissipator)))
    
    #Setup the initial state as the ground state
    rhoIn = np.zeros((systemParams.dim, systemParams.dim))
    rhoIn[0,0] = 1

    #First run 1D spectroscopy around the Bell-Rabi drive frequency
    freqSweep = 1e9*np.linspace(5.02, 5.040, 20)
#    freqSweep = [5.023e9]
    ampSweep = np.linspace(-1,1,80)
    x = np.linspace(-2,2,120)
    pulseAmps = (np.exp(-x**2)).reshape((1,x.size))
#    ampSweep = [1]
    
    rabiFreq = 320e6
    
    #Setup the pulseSequences as a series of 10us low-power pulses at different frequencies
    pulseSeqs = []
    for freq in freqSweep:
        for controlAmp in ampSweep:
            tmpPulseSeq = PulseSequence()
            tmpPulseSeq.add_control_line(freq=freq, initialPhase=0)
            tmpPulseSeq.controlAmps = rabiFreq*controlAmp*pulseAmps
            tmpPulseSeq.timeSteps = 5e-9*np.ones(x.size)
            tmpPulseSeq.maxTimeStep = 1e-6
            tmpMat = np.diag(freq*np.arange(3, dtype=np.complex128))
            tmpPulseSeq.H_int = Hamiltonian(systemParams.expand_operator('Q1', tmpMat) + systemParams.expand_operator('Q2', tmpMat))
        
            pulseSeqs.append(tmpPulseSeq)
    
    results = simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='unitary')
    results.resize((freqSweep.size, ampSweep.size))
    
    plt.figure()
#    plt.plot(ampSweep, results)
#    plt.xlabel('Frequency')
#    plt.ylabel('Measurement Voltage')
#    plt.title('Two Qubit Bell-Rabi Spectroscopy')
    plt.imshow(results, extent = [-1, 1, freqSweep[-1], freqSweep[0]], aspect='auto')
    plt.show()

