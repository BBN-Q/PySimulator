'''
Created on Nov 8, 2011

@author: cryan
'''

import numpy as np

class QuantumSystem(object):
    '''
    A generic quantum system container
    '''
    def __init__(self, dim=0):
        self.dim = dim
        
    def zeros(self):
        return np.zeros((self.dim, self.dim), dtype=np.complex128) 

class SNO(QuantumSystem):
    '''
    A class for a Standard Non-Linear Oscillator (see doi:10.1103/PhysRevA.83.012308)
    '''
    
    def __init__(self, numLevels = 0, omega = 0, delta = 0):
        '''
        Constructor
        '''
        super(SNO, self).__init__(dim=numLevels)
        self.omega = omega
        self.delta = delta
        
    def Hnat(self):
        '''
        Create the diagonal natural Hamiltonian from the level splitting an anharmonicity.
        '''
        
        #We should probably memoize this
        Hnat = self.zeros()
        
        for ct in range(1,self.dim):
            if ct == 1:
                Hnat[ct,ct] = ct*self.omega
            elif ct == 2:
                Hnat[ct,ct] = ct*self.omega + self.delta
            else:
                Hnat[ct,ct] = ct*self.omega + self.delta*(ct-1)*ct/2.0
    
        return Hnat
    
    def raisingOp(self):
        '''
        Create the raising operator under the linear harmonic oscillator function. This may not be true with a cavity.
        '''
        return np.diag(np.sqrt(np.arange(1,self.dim, dtype=np.complex128)), -1)
                             
    def loweringOp(self):
        '''
        Create the lowering operator under the linear harmonic oscillator function. This may not be true with a cavity.
        '''
        return np.diag(np.sqrt(np.arange(1,self.dim, dtype=np.complex128)), 1)

    def levelProjector(self, level):
        '''
        Creates a rank-1 projector on to an energy eigenstate
        '''
        tmpMat = self.zeros()
        tmpMat[level,level] = 1
        return tmpMat
    

class SCQubit(SNO):
    '''
    A class for superconducting qubits based on non-linear oscillators
    '''
    def __init__(self, numLevels = 0, omega = 0, delta = 0):
        super(SCQubit, self).__init__(numLevels, omega, delta)

    #Define some effective spin operators in the qubit manifold (lowest two levels)
    def pauliZ(self):
        assert self.dim > 1, 'Oops! Defining a Z operator requires two more more dimensions.'
        tmpMat = self.zeros()
        tmpMat[0,0] = -1
        tmpMat[1,1] = 1
        return tmpMat
        
    def pauliX(self):
        assert self.dim > 1, 'Oops! Defining a X operator requires two more more dimensions.'
        tmpMat = self.zeros()
        tmpMat[0,1] = 1
        tmpMat[1,0] = 1
        return tmpMat
    
    def pauliY(self):
        assert self.dim > 1, 'Oops! Defining a Y operator requires two more more dimensions.'
        tmpMat = self.zeros()
        tmpMat[0,1] = -1j
        tmpMat[1,0] = 1j
        return tmpMat
        
        
    
    
    
    