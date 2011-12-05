'''
Created on Nov 29, 2011

A simple script for looking at excited state preparation in qutrits with an eye towards the slow-light/CPT experiment.

@author: cryan
'''

from PySim.SystemParams import SystemParams
from PySim.QuantumSystems import SCQubit, Hamiltonian, Dissipator
from PySim.PulseSequence import PulseSequence
from PySim.Simulation import simulate_sequence_stack, simulate_sequence
from PySim.OptimalControl import optimize_pulse, PulseParams

import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import pi

'''System Setup'''

systemParams = SystemParams()
qubit = SCQubit(3, 0e9, -100e6, name='Q1', T1=200e-9)
systemParams.add_sub_system(qubit)
systemParams.add_control_ham(inphase = Hamiltonian(0.5*(qubit.loweringOp + qubit.raisingOp)), quadrature = Hamiltonian(-0.5*(-1j*qubit.loweringOp + 1j*qubit.raisingOp)))
systemParams.add_control_ham(inphase = Hamiltonian(0.5*(qubit.loweringOp + qubit.raisingOp)), quadrature = Hamiltonian(-0.5*(-1j*qubit.loweringOp + 1j*qubit.raisingOp)))
systemParams.measurement = qubit.levelProjector(1)
systemParams.create_full_Ham()

#Add the T1 dissipator
systemParams.dissipators = [Dissipator(qubit.T1Dissipator)]


'''
Simple Rabi Driving

We'll vary the Rabi power (but always keep the time to a calibrated pi pulse.  We expect to see a maximum at some intermediate regime
where we have a balance between selectivity and T1 decay
'''

pulseSeqs = []
pulseTimes = 1e-9*np.arange(4,300, 2)
rhoIn = qubit.levelProjector(0)
for pulseTime in pulseTimes:
    tmpPulseSeq = PulseSequence()
    tmpPulseSeq.add_control_line(freq=0e9, initialPhase=0)
    tmpPulseSeq.add_control_line(freq=0e9, initialPhase=-pi/2)
    pulseAmp = 0.5/pulseTime
    tmpPulseSeq.controlAmps = np.vstack((pulseAmp*np.array([[1]], dtype=np.float64), np.zeros(1)))
    tmpPulseSeq.timeSteps = pulseTime*np.ones(1)
    tmpPulseSeq.H_int = None
    
    pulseSeqs.append(tmpPulseSeq)

SquareResults = simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='lindblad')[0]

plt.figure()
plt.plot(pulseTimes*1e9,SquareResults)


'''
Gaussian Pulse
'''
xPts = np.linspace(-2.5,2.5,100)
gaussPulse = np.exp(-0.5*(xPts**2)) - np.exp(-0.5*2.5**2)
gaussPulse.resize((1,gaussPulse.size))
pulseInt = np.sum(gaussPulse)
pulseSeqs = []
for ct,pulseTime in enumerate(pulseTimes):
    tmpPulseSeq = PulseSequence()
    tmpPulseSeq.add_control_line(freq=0e9, initialPhase=0)
    tmpPulseSeq.add_control_line(freq=0e9, initialPhase=-pi/2)
    pulseAmp = (0.5*gaussPulse.size)/(pulseTime*pulseInt)
    tmpPulseSeq.controlAmps = np.vstack((pulseAmp*gaussPulse, np.zeros(gaussPulse.size)))
    tmpPulseSeq.timeSteps = (pulseTime/gaussPulse.size)*np.ones(gaussPulse.size)
    tmpPulseSeq.H_int = None

    pulseSeqs.append(tmpPulseSeq)

GaussResults = simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='lindblad')[0]

plt.plot(pulseTimes*1e9,GaussResults)

'''
DRAG Pulse
'''
xPts = np.linspace(-2.5,2.5,100)
gaussPulse = np.exp(-0.5*(xPts**2)) - np.exp(-0.5*2.5**2)
gaussPulse.resize((1,gaussPulse.size))
pulseInt = np.sum(gaussPulse)
pulseSeqs = []
for ct,pulseTime in enumerate(pulseTimes):
    tmpPulseSeq = PulseSequence()
    tmpPulseSeq.add_control_line(freq=0e9, initialPhase=0)
    tmpPulseSeq.add_control_line(freq=0e9, initialPhase=-pi/2)
    pulseAmp = (0.5*gaussPulse.size)/(pulseTime*pulseInt)
    #Have to muck around with DRAG scaling for some reason
    dragPulse = -0.08*(1.0/qubit.delta)*(4.0/pulseTime)*pulseAmp*(-xPts*np.exp(-0.5*(xPts**2)))
    timeStep = pulseTime/gaussPulse.size
    tmpPulseSeq.timeSteps = timeStep*np.ones(gaussPulse.size)
    tmpPulseSeq.controlAmps = np.vstack((pulseAmp*gaussPulse, dragPulse))

    pulseSeqs.append(tmpPulseSeq)

DRAGResults = simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='lindblad')[0]

plt.plot(pulseTimes*1e9,DRAGResults)


'''State to State Optimal Control'''
numSteps = 100
pulseTimes = 1e-9*np.arange(4,25,3)
results = np.zeros_like(pulseTimes)
for ct, pulseTime in enumerate(pulseTimes): 
    pulseParams = PulseParams()
    pulseParams.timeSteps = (pulseTime/numSteps)*np.ones(numSteps)
    pulseParams.rhoStart = qubit.levelProjector(0)
    pulseParams.rhoGoal = qubit.levelProjector(1)
    pulseParams.add_control_line(freq=0, initialPhase=0)
    pulseParams.add_control_line(freq=0, initialPhase=-pi/2)
    pulseParams.type = 'state2state'

    #Call the optimization    
    optimize_pulse(pulseParams, systemParams)

    #Now test the optimized pulse and make sure it puts all the population in the excited state
    results[ct] = simulate_sequence(pulseParams, systemParams, pulseParams.rhoStart, simType='lindblad')[0]
    
plt.plot(pulseTimes*1e9,results,'*')    
plt.xlabel('Pulse Time (ns)')
plt.ylabel('Population in First Excited State')
plt.title('Excited State Preparation')
plt.legend(('Square','Gaussian','DRAG','Optimal Control'))
plt.show()




