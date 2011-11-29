'''
Created on Nov 27, 2011

@author: cryan
'''
import unittest

from SystemParams import SystemParams
from Simulation import simulate_sequence
from QuantumSystems import SCQubit, Hamiltonian, Dissipator
import OptimalControl

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
        
        #Set the carrier to the qubit frequency and a 100MHz delta 
        Q1 = SCQubit(3, 0, -100e6, name='Q1')
        systemParams = SystemParams()
        systemParams.add_sub_system(Q1)
        systemParams.add_control_ham(inphase = Hamiltonian(0.5*(Q1.loweringOp + Q1.raisingOp)), quadrature = Hamiltonian(-0.5*(-1j*Q1.loweringOp + 1j*Q1.raisingOp)))
        systemParams.create_full_Ham()
        systemParams.measurement = Q1.levelProjector(1)
        
        
        #Setup the pulse parameters for the optimization
        pulseParams = OptimalControl.PulseOptimParams()
        pulseParams.timeSteps = 1e-9*np.ones(30)
        pulseParams.rhoStart = Q1.levelProjector(0)
        pulseParams.rhoGoal = Q1.levelProjector(1)
        pulseParams.add_control_line()
        pulseParams.H_int = None
        pulseParams.type = 'state2state'
        
        #Call the optimization    
        OptimalControl.optimize_pulse(pulseParams, systemParams)

        #Now test the optimized pulse and make sure it puts all the population in the excited state
        result = simulate_sequence(pulseParams, systemParams, pulseParams.rhoStart, simType='unitary')
        assert result > 0.99
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test']
    unittest.main()