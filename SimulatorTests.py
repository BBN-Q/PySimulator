'''
Created on Nov 7, 2011

@author: cryan
'''
import unittest

import numpy as np

from scipy.constants import pi

import matplotlib.pyplot as plt

from SystemParams import SystemParams
from PulseSequence import PulseSequence
from Simulation import simulate_sequence_stack, simulate_sequence
from QuantumSystems import SCQubit, Hamiltonian, Dissipator


class SingleQubitRabi(unittest.TestCase):


    def setUp(self):
        #Setup the system
        self.systemParams = SystemParams()
        self.qubit = SCQubit(2,0e9)
        self.systemParams.Hnat = Hamiltonian(2*pi*self.qubit.Hnat())
        self.systemParams.add_control_ham(inphase = Hamiltonian(pi*(self.qubit.loweringOp() + self.qubit.raisingOp())), quadrature = Hamiltonian(-pi*(-1j*self.qubit.loweringOp() + 1j*self.qubit.raisingOp())))
        self.systemParams.dim = 2
        self.systemParams.measurement = -self.qubit.pauliZ()
        
        #Define Rabi frequency and pulse lengths
        self.rabiFreq = 10e6
        self.pulseLengths = np.linspace(0,100e-9,40)
        
        #Define the initial state as the ground state
        self.rhoIn = self.qubit.levelProjector(0)

    def tearDown(self):
        pass


    def testRabiRotatingFrame(self):
        '''
        Test Rabi oscillations in the rotating frame, i.e. with zero drift Hamiltonian drive frequency of zero.
        '''
        
        #Setup the pulseSequences
        pulseSeqs = []
        for pulseLength in self.pulseLengths:
            tmpPulseSeq = PulseSequence()
            tmpPulseSeq.add_control_line(freq=0e9, initialPhase=0)
            tmpPulseSeq.controlAmps = self.rabiFreq*np.array([[1]], dtype=np.float64)
            tmpPulseSeq.timeSteps = np.array([pulseLength])
            tmpPulseSeq.maxTimeStep = pulseLength
            tmpPulseSeq.H_int = None
            
            pulseSeqs.append(tmpPulseSeq)
            
        results = simulate_sequence_stack(pulseSeqs, self.systemParams, self.rhoIn, simType='unitary')
    
        if plotResults:
            plt.figure()
            plt.plot(self.pulseLengths,results)
            plt.title('10MHz Rabi Oscillations in Rotating Frame')
            plt.show()

        np.testing.assert_allclose(results, np.cos(2*pi*self.rabiFreq*self.pulseLengths), atol = 1e-4)

    def testRabiInteractionFrame(self):
        '''
        Test Rabi oscillations after moving into an interaction frame that is different to the pulsing frame and with an irrational timestep for good measure.
        '''
        
        #Setup the system
        self.systemParams.Hnat = Hamiltonian(2*pi*np.array([[0,0], [0, 5e9]], dtype = np.complex128))
        
        #Setup the pulseSequences
        pulseSeqs = []
        for pulseLength in self.pulseLengths:
        
            tmpPulseSeq = PulseSequence()
            tmpPulseSeq.add_control_line(freq=5.0e9, initialPhase=0)
            tmpPulseSeq.controlAmps = self.rabiFreq*np.array([[1]], dtype=np.float64)
            tmpPulseSeq.timeSteps = np.array([pulseLength])
            tmpPulseSeq.maxTimeStep = pi/2*1e-10
            tmpPulseSeq.H_int = Hamiltonian(2*pi*np.array([[0,0], [0, 5.005e9]], dtype = np.complex128))
            
            pulseSeqs.append(tmpPulseSeq)
            
        results = simulate_sequence_stack(pulseSeqs, self.systemParams, self.rhoIn, simType='unitary')
    
        if plotResults:
            plt.figure()
            plt.plot(self.pulseLengths,results)
            plt.title('10MHz Rabi Oscillations using Interaction Frame')
            plt.show()

        np.testing.assert_allclose(results, np.cos(2*pi*self.rabiFreq*self.pulseLengths), atol = 1e-4)

    def testT1Recovery(self):
        '''
        Test a simple T1 recovery without any pulses.  Start in the first excited state and watch recovery down to ground state.
        '''
        self.systemParams.dissipators = [Dissipator(self.qubit.T1Dissipator(1e-6))]
        
        #Just setup a series of delays
        delays = np.linspace(0,5e-6,40)
        pulseSeqs = []
        for tmpDelay in delays:
            tmpPulseSeq = PulseSequence()
            tmpPulseSeq.timeSteps = np.array([tmpDelay])
            tmpPulseSeq.maxTimeStep = tmpDelay
            tmpPulseSeq.H_int = None
            
            pulseSeqs.append(tmpPulseSeq)
        
        results = simulate_sequence_stack(pulseSeqs, self.systemParams, np.array([[0,0],[0,1]], dtype=np.complex128), simType='lindblad')
    
        if plotResults:
            plt.figure()
            plt.plot(1e6*delays,results)
            plt.xlabel(r'Recovery Time ($\mu$s)')
            plt.ylabel(r'Expectation Value of $\sigma_z$')
            plt.title(r'$T_1$ Recovery to the Ground State')
            plt.show()
        
        np.testing.assert_allclose(results, 1-2*np.exp(-delays/1e-6), atol=1e-4)

if __name__ == "__main__":
    
    plotResults = False
    
    unittest.main()