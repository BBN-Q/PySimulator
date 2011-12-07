'''
Created on Dec 1, 2011

@author: cryan

Script for looking at single qubit control when the 0->1 transisiton of Q1 is close to the 1->2 transition of Q2. 
'''

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from scipy.constants import pi

from PySim.SystemParams import SystemParams
from PySim.QuantumSystems import Hamiltonian, Dissipator
from PySim.PulseSequence import PulseSequence
from PySim.Simulation import simulate_sequence_stack, simulate_sequence
from PySim.QuantumSystems import SCQubit

if __name__ == '__main__':
    #Setup the system
    systemParams = SystemParams()
    
    #First the two qubits
    Q1 = SCQubit(numLevels=3, omega=4.86359e9-1e6, delta=-300e6, name='Q1', T1=5.2e-6)
    systemParams.add_sub_system(Q1)
    Q2 = SCQubit(numLevels=3, omega=5.19344e9-1e6, delta=-313.656e6, name='Q2', T1=4.4e-6)
    systemParams.add_sub_system(Q2)
 
    #Add a 2MHz ZZ interaction 
    systemParams.add_interaction('Q1', 'Q2', 'ZZ', -2e6)
   
    #Create the full Hamiltonian   
    systemParams.create_full_Ham()
 
    #Some Pauli operators for the controls
    X = 0.5*(Q1.loweringOp + Q1.raisingOp)
    Y = 0.5*(-1j*Q1.loweringOp + 1j*Q2.raisingOp)
    #The cross-coupling from Q1 drive to Q2
    crossCoupling12 = 0.6
    crossCoupling21 = 0.6
    
    #Add the Q1 drive Hamiltonians
    systemParams.add_control_ham(inphase = Hamiltonian(systemParams.expand_operator('Q1', X) + crossCoupling12*systemParams.expand_operator('Q2', X)),
                                  quadrature = Hamiltonian(systemParams.expand_operator('Q1', Y) + crossCoupling12*systemParams.expand_operator('Q2', Y)))
    systemParams.add_control_ham(inphase = Hamiltonian(systemParams.expand_operator('Q1', X) + crossCoupling12*systemParams.expand_operator('Q2', X)),
                                  quadrature = Hamiltonian(systemParams.expand_operator('Q1', Y) + crossCoupling12*systemParams.expand_operator('Q2', Y)))
    
    #Add the Q2 drive Hamiltonians
    systemParams.add_control_ham(inphase = Hamiltonian(crossCoupling21*systemParams.expand_operator('Q1', X) + systemParams.expand_operator('Q2', X)),
                                  quadrature = Hamiltonian(crossCoupling21*systemParams.expand_operator('Q1', Y) + systemParams.expand_operator('Q2', Y)))
    systemParams.add_control_ham(inphase = Hamiltonian(crossCoupling21*systemParams.expand_operator('Q1', X) + systemParams.expand_operator('Q2', X)),
                                  quadrature = Hamiltonian(crossCoupling21*systemParams.expand_operator('Q1', Y) + systemParams.expand_operator('Q2', Y)))
    
    
    #Setup the measurement operator
#    systemParams.measurement = -systemParams.expand_operator('Q1', Q1.pauliZ)
    systemParams.measurement = 0.5*np.kron(Q1.levelProjector(0), Q2.levelProjector(0)) + 0.67*np.kron(Q1.levelProjector(1), Q2.levelProjector(0)) + \
                                0.64*np.kron(Q1.levelProjector(0), Q2.levelProjector(1)) + 0.72*np.kron(Q1.levelProjector(0), Q2.levelProjector(2)) + \
                                0.75*np.kron(Q1.levelProjector(1), Q2.levelProjector(1))
    #Add the T1 dissipators
    systemParams.dissipators.append(Dissipator(systemParams.expand_operator('Q1', Q1.T1Dissipator)))
    systemParams.dissipators.append(Dissipator(systemParams.expand_operator('Q2', Q2.T1Dissipator)))
    
    #Setup the initial state as the ground state
    rhoIn = np.zeros((systemParams.dim, systemParams.dim))
    rhoIn[0,0] = 1
    
    
    sampRate = 1.2e9
    timeStep = 1.0/sampRate
    
    nutFreqs = 50e6*np.linspace(-1,1,81)
    drive1Freq = Q1.omega
    drive2Freq = Q2.omega


    #Calibrate a 120ns Gaussian pulse on Q1
    numPoints = 144
    xPts = np.linspace(-2,2,numPoints)
    gaussPulse = np.exp(-(xPts**2))
    tmpControls = np.zeros((4,numPoints))
    tmpControls[0] = gaussPulse
#
#    #Setup the pulseSequences as a series of 10us low-power pulses at different frequencies
#    pulseSeqs = []
#    for nutFreq in nutFreqs:
#        tmpPulseSeq = PulseSequence()
#        tmpPulseSeq.add_control_line(freq=-drive1Freq, initialPhase=0)
#        tmpPulseSeq.add_control_line(freq=-drive1Freq, initialPhase=-pi/2)
#        tmpPulseSeq.add_control_line(freq=-drive2Freq, initialPhase=0)
#        tmpPulseSeq.add_control_line(freq=-drive2Freq, initialPhase=-pi/2)
#        tmpPulseSeq.controlAmps = nutFreq*tmpControls
#        tmpPulseSeq.timeSteps = timeStep*np.ones(tmpControls.shape[1])
#        tmpPulseSeq.maxTimeStep = 5e-10
#        tmpPulseSeq.H_int = Hamiltonian(systemParams.expand_operator('Q1', np.diag(drive1Freq*np.arange(Q1.dim, dtype=np.complex128))) + systemParams.expand_operator('Q2', np.diag(drive2Freq*np.arange(Q2.dim, dtype=np.complex128))))
#        pulseSeqs.append(tmpPulseSeq)
#
#    results = simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='unitary')[0]
#    
    #Calibrates to 9.51MHz as expected
    Q1Cal = 2*0.5/(np.sum(gaussPulse)*timeStep)
    bufferPts = 2
    Q1PiPulse = np.zeros((4,numPoints))
    Q1PiPulse[0] = Q1Cal*gaussPulse
    Q1PiBlock = np.hstack((np.zeros((4,bufferPts)), Q1PiPulse, np.zeros((4,bufferPts))))

    
    #Calibrate the Q2 pi DRAG pulse
    numPoints = 64
    xPts = np.linspace(-2,2,numPoints)
    gaussPulse = np.exp(-(xPts**2))
    dragCorrection = -0.08*(1.0/Q2.delta)*(4.0/(64.0/1.2e9))*(-xPts*np.exp(-0.5*(xPts**2)))
    tmpControls = np.zeros((4,numPoints))
    tmpControls[2] = gaussPulse
    tmpControls[3] = dragCorrection
    
    #Setup the pulseSequences as a series of 10us low-power pulses at different frequencies
#    pulseSeqs = []
#    for nutFreq in nutFreqs:
#        tmpPulseSeq = PulseSequence()
#        tmpPulseSeq.add_control_line(freq=-drive1Freq, initialPhase=0)
#        tmpPulseSeq.add_control_line(freq=-drive1Freq, initialPhase=-pi/2)
#        tmpPulseSeq.add_control_line(freq=-drive2Freq, initialPhase=0)
#        tmpPulseSeq.add_control_line(freq=-drive2Freq, initialPhase=-pi/2)
#        tmpPulseSeq.controlAmps = nutFreq*tmpControls
#        tmpPulseSeq.timeSteps = timeStep*np.ones(tmpControls.shape[1])
#        tmpPulseSeq.maxTimeStep = 5e-10
#        tmpPulseSeq.H_int = Hamiltonian(systemParams.expand_operator('Q1', np.diag(drive1Freq*np.arange(Q1.dim, dtype=np.complex128))) + systemParams.expand_operator('Q2', np.diag(drive2Freq*np.arange(Q2.dim, dtype=np.complex128))))
#        pulseSeqs.append(tmpPulseSeq)
#
#    results = simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='unitary')[0]
#
#    plt.figure()
#    plt.plot(nutFreqs/1e6, results)
#    plt.show()
#    
    #Calibrates to 21.58MHz
    Q2Cal = 2*0.5/(np.sum(gaussPulse)*timeStep)
    Q2PiBlock = np.zeros((4,numPoints+2*bufferPts))
    Q2PiBlock[2,2:-2] = gaussPulse
    Q2PiBlock[3,2:-2] = dragCorrection
    Q2PiBlock *= Q2Cal
    
    #Run the actual pi-pi-pi-pi experiment
    
    #Setup the pulse sequence blocks
    
    pulseSeqs = []
    
    tmpPulseSeq = PulseSequence()
    tmpPulseSeq.add_control_line(freq=-drive1Freq, initialPhase=0, controlType='sinusoidal')
    tmpPulseSeq.add_control_line(freq=-drive1Freq, initialPhase=-pi/2, controlType='sinusoidal')
    tmpPulseSeq.add_control_line(freq=-drive2Freq, initialPhase=0, controlType='sinusoidal')
    tmpPulseSeq.add_control_line(freq=-drive2Freq, initialPhase=-pi/2, controlType='sinusoidal')
    tmpPulseSeq.maxTimeStep = 1e-11
#    tmpPulseSeq.H_int = Hamiltonian(systemParams.expand_operator('Q1', np.diag(drive1Freq*np.arange(Q1.dim, dtype=np.complex128))) + systemParams.expand_operator('Q2', np.diag(drive2Freq*np.arange(Q2.dim, dtype=np.complex128))))

    #Q2Pi
    tmpPulseSeq.controlAmps = Q2PiBlock
    tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
    pulseSeqs.append(deepcopy(tmpPulseSeq))
    
    #Q2Pi Q1Pi
    tmpPulseSeq.controlAmps = np.hstack((Q2PiBlock, Q1PiBlock))
    tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
    pulseSeqs.append(deepcopy(tmpPulseSeq))
    
    #Q2Pi Q1Pi Q1Pi
    tmpPulseSeq.controlAmps = np.hstack((Q2PiBlock, Q1PiBlock, Q1PiBlock))
    tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
    pulseSeqs.append(deepcopy(tmpPulseSeq))
    
    #Q2Pi Q1Pi Q1Pi Q2Pi
    tmpPulseSeq.controlAmps = np.hstack((Q2PiBlock, Q1PiBlock, Q1PiBlock, Q2PiBlock))
    tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
    pulseSeqs.append(deepcopy(tmpPulseSeq))
    
    #No pulse reference
    tmpPulseSeq.controlAmps = np.zeros((4,1))
    tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
    pulseSeqs.append(deepcopy(tmpPulseSeq))
    
    #Q1 pi ref
    tmpPulseSeq.controlAmps = Q1PiBlock
    tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
    pulseSeqs.append(deepcopy(tmpPulseSeq))
    
    #Q2Pi ref
    tmpPulseSeq.controlAmps = Q2PiBlock
    tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
    pulseSeqs.append(deepcopy(tmpPulseSeq))
    
    #Q1 pi + Q2 pi ref
    tmpPulseSeq.controlAmps = np.hstack((Q1PiBlock, Q2PiBlock))
    tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
    pulseSeqs.append(deepcopy(tmpPulseSeq))
    

    results = simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='unitary')[0]
    
    plt.figure()
    plt.plot(results.repeat(2))
    plt.show()
    
    
