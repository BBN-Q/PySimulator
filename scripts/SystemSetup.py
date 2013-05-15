'''
Created on Aug 21, 2012

@author: cryan
'''

import numpy as np


from PySim.SystemParams import SystemParams
from PySim.QuantumSystems import SCQubit
from PySim.QuantumSystems import Hamiltonian, Dissipator

from scipy.linalg import eigh

def setup_system():
    '''
    Should probably take in some parameters, but for now hard-code things.
    '''
    #Setup the system
    systemParams = SystemParams()
    
    #First the two qubits defined in the lab frame
    Q1 = SCQubit(numLevels=3, omega=4.279e9, delta=-239e6, name='Q1', T1=10e-6)
    systemParams.add_sub_system(Q1)
    
    Q2 = SCQubit(numLevels=3, omega=4.06e9, delta=-224e6, name='Q2', T1=10e-6)
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
    crossCoupling12 = 5.0/20
    crossCoupling21 = 2.0/30
    
    #Add the Q1 drive Hamiltonians
    inPhaseHam = Hamiltonian(np.dot(v.conj().T, np.dot(systemParams.expand_operator('Q1', X) + crossCoupling12*systemParams.expand_operator('Q2', X), v)))
    quadratureHam = Hamiltonian(np.dot(v.conj().T, np.dot(systemParams.expand_operator('Q1', Y) + crossCoupling12*systemParams.expand_operator('Q2', Y), v)))
    systemParams.add_control_ham(inphase = inPhaseHam, quadrature = quadratureHam)
    systemParams.add_control_ham(inphase = inPhaseHam, quadrature = quadratureHam)
    
    #Add the cross-drive Hamiltonians (same drive line as Q1)
    inPhaseHam = Hamiltonian(np.dot(v.conj().T, np.dot(systemParams.expand_operator('Q1', X) + crossCoupling12*systemParams.expand_operator('Q2', X), v)))
    quadratureHam = Hamiltonian(np.dot(v.conj().T, np.dot(systemParams.expand_operator('Q1', Y) + crossCoupling12*systemParams.expand_operator('Q2', Y), v)))
    systemParams.add_control_ham(inphase = inPhaseHam, quadrature = quadratureHam)
    systemParams.add_control_ham(inphase = inPhaseHam, quadrature = quadratureHam)
 
    #Add the Q2 drive Hamiltonians
    inPhaseHam = Hamiltonian(np.dot(v.conj().T, np.dot(systemParams.expand_operator('Q2', X) + crossCoupling21*systemParams.expand_operator('Q1', X), v)))
    quadratureHam = Hamiltonian(np.dot(v.conj().T, np.dot(systemParams.expand_operator('Q2', Y) + crossCoupling21*systemParams.expand_operator('Q1', Y), v)))
    systemParams.add_control_ham(inphase = inPhaseHam, quadrature = quadratureHam)
    systemParams.add_control_ham(inphase = inPhaseHam, quadrature = quadratureHam)
    
    
    #Setup the measurement operator
    systemParams.measurement = np.diag(np.array([0.088, 0.082, 0.080, 0.080, 0.78, 0.75, 0.078, 0.074, 0.072]))
    
    ##Add the T1 dissipators
    #systemParams.dissipators.append(Dissipator(systemParams.expand_operator('Q1', Q1.T1Dissipator)))
    #systemParams.dissipators.append(Dissipator(systemParams.expand_operator('Q2', Q2.T1Dissipator)))
    #

    return systemParams, (Q1,Q2)
    

if __name__ == '__main__':
    testSys = setup_system()