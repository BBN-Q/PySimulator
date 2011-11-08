'''
Created on Nov 7, 2011

@author: cryan
'''
import unittest

import numpy as np

from scipy.constants import pi

import matplotlib.pyplot as plt

from SystemParams import SystemParams
from Hamiltonian import Hamiltonian
from PulseSequence import PulseSequence
from Simulation import simulate_sequence_stack



class SingleQubitRabi(unittest.TestCase):


    def setUp(self):
        #Setup the system
        self.systemParams = SystemParams()
        self.systemParams.Hnat = Hamiltonian(2*pi*np.array([[0,0], [0, 0e9]], dtype = np.complex128))
        self.systemParams.add_control_ham(inphase = Hamiltonian(2*pi*0.5*np.array([[0,1],[1,0]], dtype = np.complex128)), quadrature = Hamiltonian(-2*pi*0.5*np.array([[0,-1j],[1j,0]], dtype = np.complex128)))
        self.systemParams.dim = 2
        self.systemParams.measurement = np.array([[1,0],[0,-1]])
        
        #Define Rabi frequency and pulse lengths
        self.rabiFreq = 10e6
        self.pulseLengths = np.linspace(0,100e-9,40)

 

    def tearDown(self):
        pass


    def testRabiRotatingFrame(self):
        
        
        #Setup the pulseSequences
        pulseSeqs = []
        for pulseLength in self.pulseLengths:
        
            tmpPulseSeq = PulseSequence()
            tmpPulseSeq.add_control_line(freq=0e9, initialPhase=0)
            tmpPulseSeq.controlAmps = self.rabiFreq*np.array([[1]], dtype=np.float64)
            tmpPulseSeq.timeSteps = np.array([pulseLength])
            tmpPulseSeq.maxTimeStep = pulseLength
            tmpPulseSeq.H_int = Hamiltonian(2*pi*np.array([[0,0], [0, 0e9]], dtype = np.complex128))
            
            pulseSeqs.append(tmpPulseSeq)
            
        results = simulate_sequence_stack(pulseSeqs, self.systemParams, np.array([[1,0],[0,0]]), simType='unitary')
    
        plt.figure()
        plt.plot(self.pulseLengths,results)
        plt.title('10MHz Rabi Oscillations in Rotating Frame')
        plt.show()

        np.testing.assert_allclose(results, np.cos(2*pi*self.rabiFreq*self.pulseLengths), atol = 1e-4)

    def testRabiInteractionFrame(self):
        
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
            
        results = simulate_sequence_stack(pulseSeqs, self.systemParams, np.array([[1,0],[0,0]], dtype=np.complex128), simType='unitary')
    
        plt.figure()
        plt.plot(self.pulseLengths,results)
        plt.title('10MHz Rabi Oscillations using Interaction Frame')
        plt.show()

        np.testing.assert_allclose(results, np.cos(2*pi*self.rabiFreq*self.pulseLengths), atol = 1e-4)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()