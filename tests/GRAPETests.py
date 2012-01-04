'''
Created on Nov 27, 2011

@author: cryan
'''
import unittest

from PySim.SystemParams import SystemParams
from PySim.Simulation import simulate_sequence
from PySim.QuantumSystems import SCQubit, Hamiltonian
from PySim.OptimalControl import optimize_pulse, PulseParams

import numpy as np
import matplotlib.pyplot as plt


class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testInversion(self):
        '''
        Try a simple three level SC qubit system and see if can prepare the excited state. 
        '''
        
        #Setup a three level qubit and a 100MHz delta 
        Q1 = SCQubit(3, 4.987456e9, -100e6, name='Q1')
        systemParams = SystemParams()
        systemParams.add_sub_system(Q1)
        systemParams.add_control_ham(inphase = Hamiltonian(0.5*(Q1.loweringOp + Q1.raisingOp)), quadrature = Hamiltonian(0.5*(-1j*Q1.loweringOp + 1j*Q1.raisingOp)))
        systemParams.create_full_Ham()
        systemParams.measurement = Q1.levelProjector(1)
        
        
        #Setup the pulse parameters for the optimization
        pulseParams = PulseParams()
        pulseParams.timeSteps = 1e-9*np.ones(30)
        pulseParams.rhoStart = Q1.levelProjector(0)
        pulseParams.rhoGoal = Q1.levelProjector(1)
        pulseParams.add_control_line(freq=-Q1.omega)
        pulseParams.H_int = Hamiltonian(Q1.omega*np.diag(np.arange(Q1.dim)))
        pulseParams.optimType = 'state2state'
        
        #Call the optimization    
        optimize_pulse(pulseParams, systemParams)

        #Now test the optimized pulse and make sure it puts all the population in the excited state
        result = simulate_sequence(pulseParams, systemParams, pulseParams.rhoStart, simType='unitary')[0]
        assert result > 0.99
        
        
    def testDRAG(self):
        '''
        Try a unitary inversion pulse on a three level SCQuibt and see if we get something close to DRAG
        '''
        #Setup a three level qubit and a 100MHz delta 
        Q1 = SCQubit(3, 4.987456e9, -150e6, name='Q1')
        systemParams = SystemParams()
        systemParams.add_sub_system(Q1)
        systemParams.add_control_ham(inphase = Hamiltonian(0.5*(Q1.loweringOp + Q1.raisingOp)), quadrature = Hamiltonian(0.5*(-1j*Q1.loweringOp + 1j*Q1.raisingOp)))
        systemParams.add_control_ham(inphase = Hamiltonian(0.5*(Q1.loweringOp + Q1.raisingOp)), quadrature = Hamiltonian(0.5*(-1j*Q1.loweringOp + 1j*Q1.raisingOp)))
        systemParams.create_full_Ham()
        systemParams.measurement = Q1.levelProjector(1)
        
        #Setup the pulse parameters for the optimization
        numPoints = 30
        pulseTime = 15e-9
        pulseParams = PulseParams()
        pulseParams.timeSteps = (pulseTime/numPoints)*np.ones(numPoints)
        pulseParams.rhoStart = Q1.levelProjector(0)
        pulseParams.rhoGoal = Q1.levelProjector(1)
        pulseParams.Ugoal = Q1.pauliX
        pulseParams.add_control_line(freq=-Q1.omega, bandwidth=300e6, maxAmp=200e6)
        pulseParams.add_control_line(freq=-Q1.omega, phase=-np.pi/2, bandwidth=300e6, maxAmp=200e6)
        pulseParams.H_int = Hamiltonian((Q1.omega)*np.diag(np.arange(Q1.dim)))
        pulseParams.optimType = 'unitary'
        pulseParams.derivType = 'finiteDiff'
        
        #Start with a Gaussian
        tmpGauss = np.exp(-np.linspace(-2,2,numPoints)**2)
        tmpScale = 0.5/(np.sum(pulseParams.timeSteps*tmpGauss))
        pulseParams.startControlAmps = np.vstack((tmpScale*tmpGauss, np.zeros(numPoints)))
        
        #Call the optimization    
        optimize_pulse(pulseParams, systemParams)
        
        if plotResults:
            plt.plot(np.cumsum(pulseParams.timeSteps)*1e9,pulseParams.controlAmps.T/1e6);
            plt.ylabel('Pulse Amplitude (MHz)')
            plt.xlabel('Time (ns)')
            plt.legend(('X Quadrature', 'Y Quadrature'))
            plt.title('DRAG Pulse from Optimal Control')
            plt.show()
            

        #Now test the optimized pulse and make sure it does give us the desired unitary
        result = simulate_sequence(pulseParams, systemParams, pulseParams.rhoStart, simType='unitary')
        assert np.abs(np.trace(np.dot(result[1].conj().T, pulseParams.Ugoal)))**2/np.abs(np.trace(np.dot(pulseParams.Ugoal.conj().T, pulseParams.Ugoal)))**2 > 0.99



if __name__ == "__main__":
    
    plotResults = True
    
    #import sys;sys.argv = ['', 'Test.test']
    unittest.main()