'''
A script for confirming experimental results on the IBM Phase I device with 3 qubits and 3 cavities.

Created on Jan 4, 2012

@author: cryan
'''


import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from scipy.constants import pi
from scipy.linalg import eigh

from PySim.SystemParams import SystemParams
from PySim.QuantumSystems import Hamiltonian, Dissipator
from PySim.PulseSequence import PulseSequence
from PySim.Simulation import simulate_sequence_stack, simulate_sequence
from PySim.QuantumSystems import SCQubit
from PySim.OptimalControl import optimize_pulse, PulseParams


#Setup the system
systemParams = SystemParams()

#First the two qubits defined in the lab frame
Q1 = SCQubit(numLevels=3, omega=4.76093e9, delta=-244e6, name='Q1', T1=10e-6)
systemParams.add_sub_system(Q1)

Q2 = SCQubit(numLevels=3, omega=5.34012e9, delta=-224e6, name='Q2', T1=10e-6)
systemParams.add_sub_system(Q2)

#Add an interaction between the qubits to get the the cross-resonance effect 
systemParams.add_interaction('Q1', 'Q2', 'FlipFlop', 2e6)

#Create the full Hamiltonian   
systemParams.create_full_Ham()

#Calculate the eigenframe for the natural Hamiltonian
d,v = eigh(systemParams.Hnat.matrix)

#Reorder the transformation matrix to maintain the computational basis ordering
sortOrder = np.argsort(np.argmax(np.abs(v),axis=0))
v = v[:, sortOrder]
systemParams.Hnat.matrix = np.complex128(np.diag(d[sortOrder]))

#Some operators for the controls
X = 0.5*(Q1.loweringOp + Q1.raisingOp)
Y = 0.5*(-1j*Q1.loweringOp + 1j*Q1.raisingOp)

#The cross-coupling between the drives
crossCoupling12 = 12.0/24
crossCoupling21 = 0.5/12

#Add the Q1 drive Hamiltonians
inPhaseHam = Hamiltonian(np.dot(v.conj().T, np.dot(systemParams.expand_operator('Q1', X) + crossCoupling12*systemParams.expand_operator('Q2', X), v)))
quadratureHam = Hamiltonian(np.dot(v.conj().T, np.dot(systemParams.expand_operator('Q1', Y) + crossCoupling12*systemParams.expand_operator('Q2', Y), v)))
systemParams.add_control_ham(inphase = inPhaseHam, quadrature = quadratureHam)
systemParams.add_control_ham(inphase = inPhaseHam, quadrature = quadratureHam)

#Add the Q2 drive Hamiltonians
inPhaseHam = Hamiltonian(np.dot(v.conj().T, np.dot(systemParams.expand_operator('Q2', X) + crossCoupling21*systemParams.expand_operator('Q1', X), v)))
quadratureHam = Hamiltonian(np.dot(v.conj().T, np.dot(systemParams.expand_operator('Q2', Y) + crossCoupling21*systemParams.expand_operator('Q1', Y), v)))
systemParams.add_control_ham(inphase = inPhaseHam, quadrature = quadratureHam)
systemParams.add_control_ham(inphase = inPhaseHam, quadrature = quadratureHam)

#Add the cross-drive Hamiltonians (same drive line as Q2)
inPhaseHam = Hamiltonian(np.dot(v.conj().T, np.dot(systemParams.expand_operator('Q2', X) + crossCoupling21*systemParams.expand_operator('Q1', X), v)))
quadratureHam = Hamiltonian(np.dot(v.conj().T, np.dot(systemParams.expand_operator('Q2', Y) + crossCoupling21*systemParams.expand_operator('Q1', Y), v)))
systemParams.add_control_ham(inphase = inPhaseHam, quadrature = quadratureHam)
systemParams.add_control_ham(inphase = inPhaseHam, quadrature = quadratureHam)

#Setup the measurement operator
systemParams.measurement = np.diag(np.array([0.862, 0.855, 0.850, 0.850, 0.845, 0.840, 0.845, 0.840, 0.835]))

##Add the T1 dissipators
#systemParams.dissipators.append(Dissipator(systemParams.expand_operator('Q1', Q1.T1Dissipator)))
#systemParams.dissipators.append(Dissipator(systemParams.expand_operator('Q2', Q2.T1Dissipator)))
#


#Setup the initial state as the ground state
rhoIn = np.zeros((systemParams.dim, systemParams.dim))
rhoIn[0,0] = 1

#Setup the control pulses
sampRate = 1.2e9
timeStep = 1.0/sampRate
    
drive1Freq = Q1.omega
drive2Freq = Q2.omega
drive3Freq = Q1.omega

numChannels = 3
bufferPts = 2

#Calibrate some DRAG pulses
numPoints = 40
xPts = np.linspace(-2,2,numPoints)
gaussPulse = np.exp(-0.5*(xPts**2)) - np.exp(-0.5*2**2)
dragCorrection = 0.095*(1.0/Q1.delta)*(4.0/(numPoints*1.0/sampRate))*(-xPts*np.exp(-0.5*(xPts**2)))

PiCal = 0.5/(np.sum(gaussPulse)*timeStep)
Pi2Cal = 0.25/(np.sum(gaussPulse)*timeStep)
DragBlock = np.zeros((2,numPoints+2*bufferPts))
DragBlock[0,bufferPts:-bufferPts] = gaussPulse
DragBlock[1,bufferPts:-bufferPts] = dragCorrection

Q1X180Block =  np.zeros((2*numChannels,numPoints+2*bufferPts))
Q1X180Block[0:2] = PiCal*DragBlock
Q1Y180Block =  np.zeros((2*numChannels,numPoints+2*bufferPts))
Q1Y180Block[0] = -PiCal*DragBlock[1]
Q1Y180Block[1] = PiCal*DragBlock[0]

Q1X90pBlock =  np.zeros((2*numChannels,numPoints+2*bufferPts))
Q1X90pBlock[0:2] = Pi2Cal*DragBlock
Q1X90mBlock =  np.zeros((2*numChannels,numPoints+2*bufferPts))
Q1X90mBlock[0:2] = -Pi2Cal*DragBlock
Q1Y90pBlock =  np.zeros((2*numChannels,numPoints+2*bufferPts))
Q1Y90pBlock[0] = -Pi2Cal*DragBlock[1]
Q1Y90pBlock[1] = Pi2Cal*DragBlock[0]
Q1Y90mBlock = -Q1Y90pBlock

Q2X180Block =  np.zeros((2*numChannels,numPoints+2*bufferPts))
Q2X180Block[2:4] = PiCal*DragBlock

numPoints = 48
xPts = np.linspace(0,3,numPoints)
CRAmp = 200e6
halfGauss = CRAmp*np.exp(-0.5*xPts**2)

CRGaussOnBlock =  np.zeros((2*numChannels,numPoints));
CRGaussOnBlock[4] = halfGauss[-1::-1]

CRGaussOffBlock =  np.zeros((2*numChannels,numPoints));
CRGaussOffBlock[4] = halfGauss

CRBlock = np.array([0,0,0,0,1,0]).reshape(2*numChannels,1)
CRBlock *= CRAmp;

baseSeq = PulseSequence()
baseSeq.add_control_line(freq=-drive1Freq, phase=0)
baseSeq.add_control_line(freq=-drive1Freq, phase=-pi/2)
baseSeq.add_control_line(freq=-drive2Freq, phase=0)
baseSeq.add_control_line(freq=-drive2Freq, phase=-pi/2)
baseSeq.add_control_line(freq=-drive1Freq, phase=0)
baseSeq.add_control_line(freq=-drive1Freq, phase=-pi/2)
baseSeq.maxTimeStep = timeStep/4
baseSeq.H_int = Hamiltonian(systemParams.expand_operator('Q1', drive1Freq*Q1.numberOp) + systemParams.expand_operator('Q2', drive2Freq*Q2.numberOp))


'''
#Look at the all XY set of experiments


pulseSeqs = []

#Nothing
tmpPulseSeq = deepcopy(baseSeq)
tmpPulseSeq.controlAmps = np.zeros((4,1))
tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
pulseSeqs.append(tmpPulseSeq)

#X180
tmpPulseSeq = deepcopy(baseSeq)
tmpPulseSeq.controlAmps = Q1X180Block
tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
pulseSeqs.append(tmpPulseSeq)

#X90
tmpPulseSeq = deepcopy(baseSeq)
tmpPulseSeq.controlAmps = Q1X90pBlock
tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
pulseSeqs.append(tmpPulseSeq)

#Y90
tmpPulseSeq = deepcopy(baseSeq)
tmpPulseSeq.controlAmps = Q1Y90pBlock
tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
pulseSeqs.append(tmpPulseSeq)

#X180-X90
tmpPulseSeq = deepcopy(baseSeq)
tmpPulseSeq.controlAmps = np.hstack((Q1X180Block, Q1X90pBlock)).copy(order='C')
tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
pulseSeqs.append(tmpPulseSeq)

#X180-Y90
tmpPulseSeq = deepcopy(baseSeq)
tmpPulseSeq.controlAmps = np.hstack((Q1X180Block, Q1Y90pBlock)).copy(order='C')
tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
pulseSeqs.append(tmpPulseSeq)

'''


#Setup the pulseSequences to calibrate the cross-resonance gate
pulseSeqs = []

pulseLengths = np.arange(10e-9,500e-9,50e-9)

for pulseLength in pulseLengths:
    tmpPulseSeq = deepcopy(baseSeq)
    tmpPulseSeq.controlAmps = np.hstack((CRGaussOnBlock, CRBlock, CRGaussOffBlock)).copy(order='C')
    tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
    tmpPulseSeq.timeSteps[48] = pulseLength
    pulseSeqs.append(tmpPulseSeq)


for pulseLength in pulseLengths:
    tmpPulseSeq = deepcopy(baseSeq)
    tmpPulseSeq.controlAmps = np.hstack((Q2X180Block, CRGaussOnBlock, CRBlock, CRGaussOffBlock, Q2X180Block)).copy(order='C')
    tmpPulseSeq.timeSteps = timeStep*np.ones(tmpPulseSeq.controlAmps.shape[1])
    tmpPulseSeq.timeSteps[92] = pulseLength
    pulseSeqs.append(tmpPulseSeq)


#result = simulate_sequence(pulseSeqs[0], systemParams, rhoIn, simType='unitary')
results = simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='unitary')



