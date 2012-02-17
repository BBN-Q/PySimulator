"""
Created on Thu Feb  9 21:30:03 2012

Test script to play around with 2D Rabi vs drive frequency experiments
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PySim.SystemParams import SystemParams
from PySim.PulseSequence import PulseSequence
from PySim.Simulation import simulate_sequence_stack
from PySim.QuantumSystems import SCQubit, Hamiltonian


 #Setup the system
systemParams = SystemParams()
qubit = SCQubit(6, 4.76e9, delta=-200e6, name='Q1', T1=1e-6)
systemParams.add_sub_system(qubit)
systemParams.add_control_ham(inphase = Hamiltonian(0.5*(qubit.loweringOp + qubit.raisingOp)), quadrature = Hamiltonian(0.5*(-1j*qubit.loweringOp + 1j*qubit.raisingOp)))
systemParams.create_full_Ham()
systemParams.measurement = np.diag([0.72, 0.70, 0.69, 0.685, 0.6825, 0.68125])

#Define the initial state as the ground state
rhoIn = qubit.levelProjector(0)


#Some parameters for the pulse
timeStep = 1/1.2e9
pulseAmps = 250e6*np.linspace(-1,1,81)
freqs = np.linspace(4.3e9, 4.9e9,450)

numSteps = 160
xPts = np.linspace(-2,2,numSteps)
gaussPulse = np.exp(-0.5*(xPts**2)) - np.exp(-0.5*2**2)


pulseSeqs = []
for tmpFreq in freqs:
    for pulseAmp in pulseAmps:
        tmpPulseSeq = PulseSequence()
        tmpPulseSeq.add_control_line(freq=-tmpFreq, phase=0)
        tmpPulseSeq.H_int = Hamiltonian(tmpFreq*qubit.numberOp)
        tmpPulseSeq.timeSteps = timeStep*np.ones(numSteps)
        tmpPulseSeq.controlAmps = pulseAmp*gaussPulse
        pulseSeqs.append(tmpPulseSeq)


allResults = simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='unitary')

measResults = np.reshape(allResults[0], (pulseAmps.size, freqs.size), order='F')

measResults -= np.tile(np.mean(measResults, axis=0), (pulseAmps.size,1))
plt.imshow(measResults, cmap=cm.gray, aspect='auto', interpolation='none', extent=[freqs[0]/1e9, freqs[1]/1e9, pulseAmps[-1], pulseAmps[0]], vmin=-0.02, vmax=0.02)
plt.xlabel('Drive Frequency (GHz)')
plt.ylabel('Drive Amplitude')
plt.show()

