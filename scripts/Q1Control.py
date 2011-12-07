'''
Created on Dec 2, 2011

@author: cryan

Find Q1 controls when Q2's 1-2 transition is close by.

'''

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import decimate
from scipy.constants import pi
from scipy.linalg import expm

from PySim.SystemParams import SystemParams
from PySim.QuantumSystems import Hamiltonian, Dissipator
from PySim.PulseSequence import PulseSequence
from PySim.Simulation import simulate_sequence_stack, simulate_sequence
from PySim.QuantumSystems import SCQubit
from PySim.OptimalControl import optimize_pulse, PulseParams



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
Y = 0.5*(-1j*Q1.loweringOp + 1j*Q1.raisingOp)
#The cross-coupling from Q1 drive to Q2
crossCoupling12 = 0.7
crossCoupling21 = 0.7

#Add the Q1 drive Hamiltonians
systemParams.add_control_ham(inphase = Hamiltonian(systemParams.expand_operator('Q1', X) + crossCoupling12*systemParams.expand_operator('Q2', X)),
                              quadrature = Hamiltonian(systemParams.expand_operator('Q1', Y) + crossCoupling12*systemParams.expand_operator('Q2', Y)))
systemParams.add_control_ham(inphase = Hamiltonian(systemParams.expand_operator('Q1', X) + crossCoupling12*systemParams.expand_operator('Q2', X)),
                              quadrature = Hamiltonian(systemParams.expand_operator('Q1', Y) + crossCoupling12*systemParams.expand_operator('Q2', Y)))

#Add the Q2 drive Hamiltonians
#systemParams.add_control_ham(inphase = Hamiltonian(crossCoupling21*systemParams.expand_operator('Q1', X) + systemParams.expand_operator('Q2', X)),
#                              quadrature = Hamiltonian(crossCoupling21*systemParams.expand_operator('Q1', Y) + systemParams.expand_operator('Q2', Y)))
#systemParams.add_control_ham(inphase = Hamiltonian(crossCoupling21*systemParams.expand_operator('Q1', X) + systemParams.expand_operator('Q2', X)),
#                              quadrature = Hamiltonian(crossCoupling21*systemParams.expand_operator('Q1', Y) + systemParams.expand_operator('Q2', Y)))

#Setup the measurement operator
#    systemParams.measurement = -systemParams.expand_operator('Q1', Q1.pauliZ)
#systemParams.measurement = 0.6*np.eye(9) - 0.07*systemParams.expand_operator('Q1', Q1.pauliZ) - 0.05*systemParams.expand_operator('Q2', Q2.pauliZ) - 0.04*np.kron(Q1.pauliZ, Q2.pauliZ)
#
##Add the T1 dissipators
#systemParams.dissipators.append(Dissipator(systemParams.expand_operator('Q1', Q1.T1Dissipator)))
#systemParams.dissipators.append(Dissipator(systemParams.expand_operator('Q2', Q2.T1Dissipator)))
#
##Setup the initial state as the ground state
#rhoIn = np.zeros((systemParams.dim, systemParams.dim))
#rhoIn[0,0] = 1


sampRate = 10*1.2e9
timeStep = 1.0/sampRate

drive1Freq = Q1.omega
drive2Freq = Q2.omega

pulseParams = PulseParams()
pulseParams.timeSteps = timeStep*np.ones(1440)
pulseParams.add_control_line(freq=-drive1Freq, initialPhase=0, bandwidth=300e6, maxAmp=100e6)
pulseParams.add_control_line(freq=-drive1Freq, initialPhase=-pi/2, bandwidth=300e6, maxAmp=100e6)
#pulseParams.add_control_line(freq=-drive2Freq, initialPhase=0, bandwidth=300e6, maxAmp=100e6)
#pulseParams.add_control_line(freq=-drive2Freq, initialPhase=-pi/2, bandwidth=300e6, maxAmp=100e6)
pulseParams.H_int = Hamiltonian(systemParams.expand_operator('Q1', np.diag(drive1Freq*np.arange(Q1.dim, dtype=np.complex128))) + systemParams.expand_operator('Q2', np.diag(drive2Freq*np.arange(Q2.dim, dtype=np.complex128))))
#pulseParams.H_int = Hamiltonian(Q1.omega*np.diag(np.arange(Q1.dim)))
pulseParams.type = 'unitary'

Q2Goal = np.eye(3, dtype=np.complex128)
Q2Goal[2,2] = 0
pulseParams.Ugoal = np.kron(Q1.pauliX, Q2Goal)

#Rotate Q2's desired unitary by the frame rotation
#UFrameShift = expm(1j*72e-9*systemParams.expand_operator('Q2', np.diag((Q2.omega-Q1.omega)*np.arange(Q2.dim, dtype=np.complex128))))
#pulseParams.Ugoal = np.dot(UFrameShift, pulseParams.Ugoal)


#State to state goals
pulseParams.rhoStart = np.zeros((9,9), dtype=np.complex128)
pulseParams.rhoStart[0,0] = 1
pulseParams.rhoGoal = np.zeros((9,9), dtype=np.complex128)
pulseParams.rhoGoal[3,3] = 1

#Call the optimization    
optimize_pulse(pulseParams, systemParams)

#Decimate the pulse down to the AWG sampling rate
pulseParams.controlAmps = decimate(pulseParams.controlAmps, 10, n=5, axis=1)
pulseParams.timeSteps = 1e-9*np.ones(60)


'''
Look at the fidelity of some more traditional pulses
'''

#First a simple Gaussian 
#xPts = np.linspace(-2.5,2.5,1000)
#gaussPulse = np.exp(-0.5*(xPts**2)) - np.exp(-0.5*2.5**2)
#pulseInt = np.sum(gaussPulse)
#pulseTimes = 1e-9*np.arange(20,400,10)
#results = np.zeros(pulseTimes.size)
#for ct,pulseTime in enumerate(pulseTimes):
#    #Work out the scaling 
#    pulseScale = (0.5*gaussPulse.size)/(pulseTime*pulseInt)
#    pulseParams.controlAmps = pulseScale*np.vstack((gaussPulse, np.zeros(gaussPulse.size)))
#    pulseParams.timeSteps = (pulseTime/gaussPulse.size)*np.ones(gaussPulse.size)
#    tmpResult = simulate_sequence(pulseParams, systemParams, pulseParams.rhoStart, simType='unitary')
#    results[ct] = np.abs(np.trace(np.dot(pulseParams.Ugoal.conj().T, tmpResult[1])))**2/16
#
#plt.figure()
#plt.plot(pulseTimes*1e9, results)
#
##Now a Hermite 180
#xPts = np.linspace(-2,2,1000)
#Hermite180 = (1-0.956*xPts**2)*np.exp(-(xPts**2))
#
