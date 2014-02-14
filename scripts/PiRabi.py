'''
Created on Aug 21, 2012

@author: cryan
'''

import numpy as np
import matplotlib.pyplot as plt

import cPickle

from copy import deepcopy

from scipy.constants import pi

from PySim.QuantumSystems import Hamiltonian, Dissipator
from PySim.SystemParams import SystemParams

from PySim.PulseSequence import PulseSequence
from PySim.Simulation import simulate_sequence_stack, simulate_sequence

from SystemSetup import setup_system


if __name__ == '__main__':
    
    #Setup the system
    systemParams, qubits = setup_system()
    
    #Setup the initial state as the ground state
    rhoIn = np.zeros((systemParams.dim, systemParams.dim))
    rhoIn[0,0] = 1
    
    #Setup the control pulses
    sampRate = 1.2e9
    timeStep = 1.0/sampRate
        
    drive1Freq = qubits[0].omega
    drive2Freq = qubits[1].omega
    drive3Freq = qubits[0].omega

    #Setup the sequences 
    
    #First a base sequence
    baseSeq = PulseSequence()
    baseSeq.add_control_line(freq=-drive1Freq, phase=0)
    baseSeq.add_control_line(freq=-drive1Freq, phase=-pi/2)
    baseSeq.add_control_line(freq=-drive2Freq, phase=0)
    baseSeq.add_control_line(freq=-drive2Freq, phase=-pi/2)
    baseSeq.add_control_line(freq=-drive3Freq, phase=0)
    baseSeq.add_control_line(freq=-drive3Freq, phase=-pi/2)
    baseSeq.maxTimeStep = timeStep/10
    baseSeq.H_int = Hamiltonian(systemParams.expand_operator('Q1', drive1Freq*qubits[0].numberOp) + systemParams.expand_operator('Q2', drive2Freq*qubits[1].numberOp))

    #Now, load the sequences from the pickle
    with open('/home/cryan/Programming/Repos/PyQLab/PulseSequencer/pulseSeq.pkl') as FID:
        AWGWFs = cPickle.load(FID)
        
    #Now create sequence matrices
    numSeqs = len(AWGWFs['TekAWG1']['ch1'])
    calScale = 34855558.146774516
    startPt = 180
    endPt = -200
    pulseSeqs = []
    for ct in range(numSeqs):
        #Stack together the appropriate waveforms
        tmpSeq = deepcopy(baseSeq)
        tmpSeq.controlAmps = calScale*np.vstack((AWGWFs['TekAWG2']['ch1'][ct][startPt:endPt], AWGWFs['TekAWG2']['ch2'][ct][startPt:endPt], \
                                        AWGWFs['TekAWG2']['ch3'][ct][startPt:endPt], AWGWFs['TekAWG2']['ch4'][ct][startPt:endPt], \
                                        AWGWFs['TekAWG1']['ch1'][ct][startPt:endPt], AWGWFs['TekAWG1']['ch2'][ct][startPt:endPt]))
        tmpSeq.timeSteps =  timeStep*np.ones(tmpSeq.controlAmps.shape[1])
        pulseSeqs.append(tmpSeq)
    
    results = simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='unitary')
