'''
Created on Nov 6, 2011

@author: cryan

Rough script for testing the python simulator
'''

import numpy as np
import matplotlib.pyplot as plt

from PySim.SystemParams import SystemParams
from PySim.QuantumSystems import Hamiltonian, Dissipator
from PySim.PulseSequence import PulseSequence
from PySim.Simulation import simulate_sequence_stack
from PySim.QuantumSystems import SCQubit

if __name__ == '__main__':
    
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
    
    '''  Try to recreate the Bell-Rabi spectroscopy '''
    
    #Setup the system
    systemParams = SystemParams()
    
    #First the two qubits
    Q1 = SCQubit(numLevels=3, omega=4.863e9-1e6, delta=-300e6, name='Q1', T1=5.2e-6)
    systemParams.add_sub_system(Q1)
    Q2 = SCQubit(numLevels=3, omega=5.193e9-1e6, delta=-313.656e6, name='Q2', T1=4.4e-6)
    systemParams.add_sub_system(Q2)
 
    #Add a 2MHz ZZ interaction 
    systemParams.add_interaction('Q1', 'Q2', 'ZZ', -2e6)
   
    #Create the full Hamiltonian   
    systemParams.create_full_Ham()
 
    #Some Pauli operators for the controls
    X = 0.5*(Q1.loweringOp + Q1.raisingOp)
    Y = 0.5*(-1j*Q1.loweringOp + 1j*Q2.raisingOp)
    #The cross-coupling from Q1 drive to Q2
    crossCoupling = 1
    
    #Add the Q1 drive Hamiltonians
    systemParams.add_control_ham(inphase = Hamiltonian(systemParams.expand_operator('Q1', X) + crossCoupling*systemParams.expand_operator('Q2', X)),
                                  quadrature = Hamiltonian(systemParams.expand_operator('Q1', Y) + crossCoupling*systemParams.expand_operator('Q2', Y)))
    
    #Setup the measurement operator
    systemParams.measurement = np.kron(Q1.levelProjector(1), Q2.levelProjector(1))
#    systemParams.measurement = 0.5*np.kron(Q1.levelProjector(0), Q2.levelProjector(0)) + 0.67*np.kron(Q1.levelProjector(1), Q2.levelProjector(0)) + \
#                                0.64*np.kron(Q1.levelProjector(0), Q2.levelProjector(1)) + 0.72*np.kron(Q1.levelProjector(0), Q2.levelProjector(2)) + \
#                                0.75*np.kron(Q1.levelProjector(1), Q2.levelProjector(1))

    #Add the T1 dissipators
    systemParams.dissipators.append(Dissipator(systemParams.expand_operator('Q1', Q1.T1Dissipator)))
    systemParams.dissipators.append(Dissipator(systemParams.expand_operator('Q2', Q2.T1Dissipator)))
    
    #Setup the initial state as the ground state
    rhoIn = np.zeros((systemParams.dim, systemParams.dim))
    rhoIn[0,0] = 1

    #First run 1D spectroscopy around the Bell-Rabi drive frequency
    freqSweep = 1e9*np.linspace(5.01, 5.040, 1000)
#    freqSweep = [5.023e9]
    ampSweep = np.linspace(-1,1,80)
    x = np.linspace(-2,2,20)
    pulseAmps = (np.exp(-x**2)).reshape((1,x.size))
#    pulseAmps = np.ones((1,1))
    ampSweep = [0.1]
    
    rabiFreq = 200e6
    
    #Setup the pulseSequences as a series of 10us low-power pulses at different frequencies
    pulseSeqs = []
    for freq in freqSweep:
        for controlAmp in ampSweep:
            tmpPulseSeq = PulseSequence()
            tmpPulseSeq.add_control_line(freq=-freq, initialPhase=0)
            tmpPulseSeq.controlAmps = rabiFreq*controlAmp*pulseAmps
            tmpPulseSeq.timeSteps = 5e-9*np.ones(x.size)
            tmpMat = freq*Q1.numberOp
            tmpPulseSeq.H_int = Hamiltonian(systemParams.expand_operator('Q1', tmpMat) + systemParams.expand_operator('Q2', tmpMat))
        
            pulseSeqs.append(tmpPulseSeq)
    
    results = simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='lindblad')[0]
#    results.resize((freqSweep.size, ampSweep.size))
    plt.plot(freqSweep,results)
    plt.show()
#    plt.figure()
##    plt.plot(ampSweep, results)
##    plt.xlabel('Frequency')
##    plt.ylabel('Measurement Voltage')
##    plt.title('Two Qubit Bell-Rabi Spectroscopy')
#    plt.imshow(results, extent = [-1, 1, freqSweep[-1], freqSweep[0]], aspect='auto')
#    plt.show()

