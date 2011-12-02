'''
Created on Nov 7, 2011

@author: cryan
'''

import unittest

import numpy as np

from scipy.constants import pi

import matplotlib.pyplot as plt

from PySim.SystemParams import SystemParams
from PySim.PulseSequence import PulseSequence
from PySim.Simulation import simulate_sequence_stack, simulate_sequence
from PySim.QuantumSystems import SCQubit, Hamiltonian, Dissipator


class SingleQubit(unittest.TestCase):

    def setUp(self):
        #Setup the system
        self.systemParams = SystemParams()
        self.qubit = SCQubit(2,0e9, name='Q1', T1=1e-6)
        self.systemParams.add_sub_system(self.qubit)
        #self.systemParams.add_control_ham(inphase = Hamiltonian(0.5*(self.qubit.loweringOp + self.qubit.raisingOp)), quadrature = Hamiltonian(0.5*(-1j*self.qubit.loweringOp + 1j*self.qubit.raisingOp)))
        self.systemParams.add_control_ham(inphase = Hamiltonian(0.5*self.qubit.pauliX), quadrature = Hamiltonian(0.5*self.qubit.pauliY))
        self.systemParams.measurement = self.qubit.pauliZ
        self.systemParams.create_full_Ham()
        
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
        expectedResults = np.cos(2*pi*self.rabiFreq*self.pulseLengths)
        if plotResults:
            plt.figure()
            plt.plot(self.pulseLengths,results)
            plt.plot(self.pulseLengths, expectedResults, color='r', linestyle='--', linewidth=2)
            plt.title('10MHz Rabi Oscillations in Rotating Frame')
            plt.xlabel('Pulse Length')
            plt.ylabel(r'$\sigma_z$')
            plt.legend(('Simulated Results', '10MHz Cosine'))
            plt.show()

        np.testing.assert_allclose(results, expectedResults , atol = 1e-4)

    def testRabiInteractionFrame(self):
        '''
        Test Rabi oscillations after moving into an interaction frame that is different to the pulsing frame and with an irrational timestep for good measure.
        '''
        
        #Setup the system
        self.systemParams.subSystems[0] = SCQubit(2,5e9, 'Q1')
        self.systemParams.create_full_Ham()
        
        #Setup the pulseSequences
        pulseSeqs = []
        for pulseLength in self.pulseLengths:
        
            tmpPulseSeq = PulseSequence()
            tmpPulseSeq.add_control_line(freq=-5.0e9, initialPhase=0)
            tmpPulseSeq.controlAmps = self.rabiFreq*np.array([[1]], dtype=np.float64)
            tmpPulseSeq.timeSteps = np.array([pulseLength])
            tmpPulseSeq.maxTimeStep = pi/2*1e-10
            tmpPulseSeq.H_int = Hamiltonian(np.array([[0,0], [0, 5.005e9]], dtype = np.complex128))
            
            pulseSeqs.append(tmpPulseSeq)
            
        results = simulate_sequence_stack(pulseSeqs, self.systemParams, self.rhoIn, simType='unitary')
        expectedResults = np.cos(2*pi*self.rabiFreq*self.pulseLengths)
        if plotResults:
            plt.figure()
            plt.plot(self.pulseLengths,results)
            plt.plot(self.pulseLengths, expectedResults, color='r', linestyle='--', linewidth=2)
            plt.title('10MHz Rabi Oscillations in Rotating Frame')
            plt.xlabel('Pulse Length')
            plt.ylabel(r'$\sigma_z$')
            plt.legend(('Simulated Results', '10MHz Cosine'))
            plt.show()

        np.testing.assert_allclose(results, expectedResults , atol = 1e-4)
        
    def testRamsey(self):
        '''
        Just look at Ramsey decay to make sure we get the off-resonance right.
        '''
        
        #Setup the system
        self.systemParams.subSystems[0] = SCQubit(2,5e9, 'Q1')
        self.systemParams.create_full_Ham()
        self.systemParams.dissipators = [Dissipator(self.qubit.T1Dissipator)]
        
        #Setup the pulseSequences
        delays = np.linspace(0,8e-6,200)
        t90 = 0.25*(1/self.rabiFreq)
        offRes = 0.56789e6
        pulseSeqs = []
        for delay in delays:
        
            tmpPulseSeq = PulseSequence()
            #Shift the pulsing frequency down by offRes
            tmpPulseSeq.add_control_line(freq=-(5.0e9-offRes), initialPhase=0)
            #Pulse sequence is X90, delay, X90
            tmpPulseSeq.controlAmps = self.rabiFreq*np.array([[1, 0, 1]], dtype=np.float64)
            tmpPulseSeq.timeSteps = np.array([t90, delay, t90])
            #Interaction frame with some odd frequency
            tmpPulseSeq.H_int = Hamiltonian(np.array([[0,0], [0, 5.00e9]], dtype = np.complex128))
            
            pulseSeqs.append(tmpPulseSeq)
            
        results = simulate_sequence_stack(pulseSeqs, self.systemParams, self.rhoIn, simType='lindblad')
        expectedResults = -np.cos(2*pi*offRes*(delays+t90))*np.exp(-delays/(2*self.qubit.T1))
        if plotResults:
            plt.figure()
            plt.plot(1e6*delays,results)
            plt.plot(1e6*delays, expectedResults, color='r', linestyle='--', linewidth=2)
            plt.title('Ramsey Fringes 0.56789MHz Off-Resonance')
            plt.xlabel('Pulse Spacing (us)')
            plt.ylabel(r'$\sigma_z$')
            plt.legend(('Simulated Results', '0.57MHz Cosine with T1 limited decay.'))
            plt.show()

    def testYPhase(self):
        
        '''
        Make sure the frame-handedness matches what we expect: i.e. if the qubit frequency is 
        greater than the driver frequency this corresponds to a positive rotation.
        '''        
        #Setup the system
        self.systemParams.subSystems[0] = SCQubit(2,5e9, 'Q1')
        self.systemParams.create_full_Ham()
        
        #Add a Y control Hamiltonian 
        self.systemParams.add_control_ham(inphase = Hamiltonian(0.5*self.qubit.pauliX), quadrature = Hamiltonian(0.5*self.qubit.pauliY))
        
        #Setup the pulseSequences
        delays = np.linspace(0,8e-6,200)
        t90 = 0.25*(1/self.rabiFreq)
        offRes = 1.2345e6
        pulseSeqs = []
        for delay in delays:
        
            tmpPulseSeq = PulseSequence()
            #Shift the pulsing frequency down by offRes
            tmpPulseSeq.add_control_line(freq=-(5.0e9-offRes), initialPhase=0)
            tmpPulseSeq.add_control_line(freq=-(5.0e9-offRes), initialPhase=-pi/2)
            #Pulse sequence is X90, delay, Y90
            tmpPulseSeq.controlAmps = self.rabiFreq*np.array([[1, 0, 0],[0,0,1]], dtype=np.float64)
            tmpPulseSeq.timeSteps = np.array([t90, delay, t90])
            #Interaction frame with some odd frequency
            tmpPulseSeq.H_int = Hamiltonian(np.array([[0,0], [0, 5.00e9]], dtype = np.complex128))
            
            pulseSeqs.append(tmpPulseSeq)
            
        results = simulate_sequence_stack(pulseSeqs, self.systemParams, self.rhoIn, simType='lindblad')
        expectedResults = -np.sin(2*pi*offRes*(delays+t90))
        if plotResults:
            plt.figure()
            plt.plot(1e6*delays,results)
            plt.plot(1e6*delays, expectedResults, color='r', linestyle='--', linewidth=2)
            plt.title('Ramsey Fringes %.2f MHz Off-Resonance' % (offRes/1e6))
            plt.xlabel('Pulse Spacing (us)')
            plt.ylabel(r'$\sigma_z$')
            plt.legend(('Simulated Results', '%.2f MHz Cosine with T1 limited decay.' % (offRes/1e6) ))
            plt.show()
        

        
    def testT1Recovery(self):
        '''
        Test a simple T1 recovery without any pulses.  Start in the first excited state and watch recovery down to ground state.
        '''
        self.systemParams.dissipators = [Dissipator(self.qubit.T1Dissipator)]
        
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
        expectedResults = 1-2*np.exp(-delays/self.qubit.T1)
        if plotResults:
            plt.figure()
            plt.plot(1e6*delays,results)
            plt.plot(1e6*delays, expectedResults, color='r', linestyle='--', linewidth=2)
            plt.xlabel(r'Recovery Time ($\mu$s)')
            plt.ylabel(r'Expectation Value of $\sigma_z$')
            plt.title(r'$T_1$ Recovery to the Ground State')
            plt.legend(('Simulated Results', 'Exponential T1 Recovery'))
            plt.show()
        
        np.testing.assert_allclose(results, expectedResults, atol=1e-4)
        
class SingleQutrit(unittest.TestCase):

    def setUp(self):
        #Setup the system
        self.systemParams = SystemParams()
        self.qubit = SCQubit(3, 5e9, -100e6, name='Q1', T1=2e-6)
        self.systemParams.add_sub_system(self.qubit)
        self.systemParams.add_control_ham(inphase = Hamiltonian(0.5*(self.qubit.loweringOp + self.qubit.raisingOp)), quadrature = Hamiltonian(-0.5*(-1j*self.qubit.loweringOp + 1j*self.qubit.raisingOp)))
        self.systemParams.measurement = self.qubit.pauliZ
        self.systemParams.create_full_Ham()
        
        #Add the 2us T1 dissipator
        self.systemParams.dissipators = [Dissipator(self.qubit.T1Dissipator)]

        #Define the initial state as the ground state
        self.rhoIn = self.qubit.levelProjector(0)

    def tearDown(self):
        pass


    def testTwoPhoton(self):
        
        '''
        Test spectroscopy and the two photon transition from the ground to the second excited state.
        '''
        
        freqSweep = 1e9*np.linspace(4.9, 5.1, 1000)
        rabiFreq = 1e6
        
        #Setup the pulseSequences as a series of 10us low-power pulses
        pulseSeqs = []
        for freq in freqSweep:
            tmpPulseSeq = PulseSequence()
            tmpPulseSeq.add_control_line(freq=freq, initialPhase=0)
            tmpPulseSeq.controlAmps = np.array([[rabiFreq]], dtype=np.float64)
            tmpPulseSeq.timeSteps = np.array([10e-6])
            tmpPulseSeq.maxTimeStep = np.Inf
            tmpPulseSeq.H_int = Hamiltonian(np.diag(freq*np.arange(3, dtype=np.complex128)))
            
            pulseSeqs.append(tmpPulseSeq)
        
        results = simulate_sequence_stack(pulseSeqs, self.systemParams, self.rhoIn, simType='lindblad')

        if plotResults:        
            plt.figure()
            plt.plot(freqSweep/1e9,results)
            plt.xlabel('Frequency')
            plt.ylabel(r'$\sigma_z$')
            plt.title('Qutrit Spectroscopy')
            plt.show()


class TwoQubit(unittest.TestCase):


    def setUp(self):
        #Setup a simple non-coupled two qubit system
        self.systemParams = SystemParams()
        self.Q1 = SCQubit(2, 5e9, name='Q1')
        self.systemParams.add_sub_system(self.Q1)
        self.Q2 = SCQubit(2, 6e9, name='Q2')
        self.systemParams.add_sub_system(self.Q2)
        X = 0.5*(self.Q1.loweringOp + self.Q1.raisingOp)
        Y = 0.5*(-1j*self.Q1.loweringOp + 1j*self.Q2.raisingOp)
        self.systemParams.add_control_ham(inphase = Hamiltonian(self.systemParams.expand_operator('Q1', X)), quadrature = Hamiltonian(self.systemParams.expand_operator('Q1', Y)))
        self.systemParams.add_control_ham(inphase = Hamiltonian(self.systemParams.expand_operator('Q2', X)), quadrature = Hamiltonian(self.systemParams.expand_operator('Q2', Y)))
        self.systemParams.measurement = self.systemParams.expand_operator('Q1', self.Q1.pauliZ) + self.systemParams.expand_operator('Q2', self.Q2.pauliZ)
        self.systemParams.create_full_Ham()
        
        #Define Rabi frequency and pulse lengths
        self.rabiFreq = 10e6
        
        #Define the initial state as the ground state
        self.rhoIn = np.zeros((self.systemParams.dim, self.systemParams.dim))
        self.rhoIn[0,0] = 1

    def tearDown(self):
        pass

    def testZZGate(self):
        '''Test whether the ZZ interaction performs as expected'''
        
        #Take the bare Hamiltonian as the interaction Hamiltonian
        H_int = Hamiltonian(self.systemParams.Hnat.matrix)
        
        #First add a 2MHz ZZ interaction and add it to the system
        self.systemParams.add_interaction('Q1', 'Q2', 'ZZ', 2e6)
        self.systemParams.create_full_Ham()
        
        #Setup the pulseSequences
        delays = np.linspace(0,4e-6,100)
        pulseSeqs = []
        for delay in delays:
            tmpPulseSeq = PulseSequence()
            tmpPulseSeq.add_control_line(freq=-5.0e9, initialPhase=0)
            tmpPulseSeq.controlAmps = self.rabiFreq*np.array([[1, 0], [0,0]], dtype=np.float64)
            tmpPulseSeq.timeSteps = np.array([25e-9, delay])
            tmpPulseSeq.maxTimeStep = np.Inf
            tmpPulseSeq.H_int = H_int
 
            pulseSeqs.append(tmpPulseSeq)
        
        self.systemParams.measurement = np.kron(self.Q1.pauliX, self.Q2.pauliZ)
        resultsXZ = simulate_sequence_stack(pulseSeqs, self.systemParams, self.rhoIn, simType='unitary')
        self.systemParams.measurement = np.kron(self.Q1.pauliY, self.Q2.pauliZ)
        resultsYZ = simulate_sequence_stack(pulseSeqs, self.systemParams, self.rhoIn, simType='unitary')

        if plotResults:
            plt.figure()
            plt.plot(delays*1e6, resultsXZ)
            plt.plot(delays*1e6, resultsYZ, 'g')
            plt.legend(('XZ','YZ'))
            plt.xlabel('Coupling Time')
            plt.ylabel('Operator Expectation Value')
            plt.title('ZZ Coupling Evolution After a X90 on Q1')
            plt.show()

if __name__ == "__main__":
    
    plotResults = True
    
#    unittest.main()
    singleTest = unittest.TestSuite()
    singleTest.addTest(SingleQubit("testYPhase"))
    unittest.TextTestRunner(verbosity=2).run(singleTest)