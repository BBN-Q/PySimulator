"""
Created on Thu Feb  9 21:30:03 2012

Test script to play around with the single pulse single qubit Clifford gates achieved via phase-ramping.
"""

import numpy as np

from scipy.constants import pi
from scipy.linalg import expm

import matplotlib.pyplot as plt

from PySim.SystemParams import SystemParams
from PySim.PulseSequence import PulseSequence
from PySim.Simulation import simulate_sequence_stack
from PySim.QuantumSystems import SCQubit, Hamiltonian

from copy import deepcopy

 #Setup the system
systemParams = SystemParams()
qubit = SCQubit(3, 0e9, delta=200e6, name='Q1', T1=1e-6)
systemParams.add_sub_system(qubit)
systemParams.add_control_ham(inphase = Hamiltonian(0.5*(qubit.loweringOp + qubit.raisingOp)), quadrature = Hamiltonian(0.5*(-1j*qubit.loweringOp + 1j*qubit.raisingOp)))
systemParams.add_control_ham(inphase = Hamiltonian(0.5*(qubit.loweringOp + qubit.raisingOp)), quadrature = Hamiltonian(0.5*(-1j*qubit.loweringOp + 1j*qubit.raisingOp)))
systemParams.measurement = qubit.pauliZ
systemParams.create_full_Ham()

#Define the initial state as the ground state
rhoIn = qubit.levelProjector(0)


#First the basic sequence
basePulseSeq = PulseSequence()
basePulseSeq.add_control_line(freq=0e9, phase=0)
basePulseSeq.add_control_line(freq=0e9, phase=pi/2)
basePulseSeq.H_int = None

#Some parameters for the pulse
timeStep = 1/1.2e9
#How many discrete timesteps to break it up into
stepsArray = np.arange(3,61)



'''
Test a square Hadamard pulse.  
'''

#Setup the pulseSequences
pulseSeqs = []
#
#for numSteps in stepsArray:
#    tmpPulseSeq = deepcopy(basePulseSeq)
#    timePts = np.arange(0, numSteps*timeStep, timeStep)
#    calScale = 0.5/np.sqrt(2)/np.sum(timeStep*np.ones(numSteps))
#    phaseRamp = 2*pi*calScale*np.cumsum(timeStep*np.ones(numSteps))
#    phaseCorr = phaseRamp[-1]
#    #Choose average phase for each time-step by taking difference between subsequent points
#    phaseRamp -= np.diff(np.hstack((0, phaseRamp)))/2
#    
#    complexPulse = calScale*np.exp(-1j*phaseRamp)
#    tmpPulseSeq.controlAmps = np.vstack((complexPulse.real, complexPulse.imag))
#    tmpPulseSeq.timeSteps = timeStep*np.ones(numSteps)
#    
#    pulseSeqs.append(tmpPulseSeq)



'''
Test a Gaussian ampitude profile (with potential DRAG correction).
'''
pulseSeqs = []
for numSteps in stepsArray:
    tmpPulseSeq = deepcopy(basePulseSeq)
    xPts = np.linspace(-2,2,numSteps)
    gaussPulse = np.exp(-0.5*(xPts**2)) - np.exp(-0.5*2**2)
    DRAGPulse = -0.1*(1.0/qubit.delta)*(4.0/(numSteps*timeStep))*(-xPts*np.exp(-0.5*(xPts**2)))

    timePts = np.arange(0, numSteps*timeStep, timeStep)
    calScale = 0.5/np.sqrt(2)/np.sum(timeStep*gaussPulse)
    phaseRamp = 2*pi*calScale*np.cumsum(timeStep*gaussPulse)  
    #Optional DRAG correction
    delta = 0
    phaseRamp += delta*0.25*(1/qubit.delta)*np.cumsum(timeStep*(2*pi*calScale*gaussPulse)**2)    
    #Final change in frame
    phaseCorr = phaseRamp[-1]
    
    #Choose average phase for each time-step by taking difference between subsequent points
    phaseRamp -= np.diff(np.hstack((0, phaseRamp)))/2

    
    complexPulse = calScale*(gaussPulse+1j*DRAGPulse)*np.exp(-1j*phaseRamp)
    tmpPulseSeq.controlAmps = np.vstack((complexPulse.real, complexPulse.imag))
    tmpPulseSeq.timeSteps = timeStep*np.ones(numSteps)
    
    pulseSeqs.append(tmpPulseSeq)






results = simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='unitary')



#Single qubit paulis
X = np.array([[0, 1],[1, 0]])
Y = np.array([[0, -1j],[1j, 0]])
Z = np.array([[1, 0],[0, -1]]);
I = np.eye(2)

UgoalQ = expm(-1j*(pi/2)*(1/np.sqrt(2))*(X+Z))
Ugoal = np.zeros((qubit.dim,qubit.dim), dtype=np.complex128)
Ugoal[0:2,0:2] = UgoalQ

fidelity = [np.abs(np.trace(np.dot(Ugoal.conj().transpose(), np.dot(expm(-1j*(phaseCorr/2)*qubit.pauliZ),tmpU))))**2/4 for tmpU in results[1]]

plt.figure()
plt.semilogy(stepsArray*timeStep,1-np.array(fidelity))
plt.xlabel('Length of Pulse')
plt.ylabel('Gate Error (after Z-correction)')
#plt.savefig('/home/cryan/Desktop/junk.pdf')
plt.show()
print(fidelity[-1])