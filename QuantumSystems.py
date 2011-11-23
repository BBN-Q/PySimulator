'''
Created on Nov 8, 2011

@author: cryan
'''

import numpy as np

class QuantumSystem(object):
    '''
    A generic quantum system container
    '''
    def __init__(self, dim=0, name=None):
        self.dim = dim
        self.name = name
        
    def zeros(self):
        return np.zeros((self.dim, self.dim), dtype=np.complex128) 

class SNO(QuantumSystem):
    '''
    A class for a Standard Non-Linear Oscillator (see doi:10.1103/PhysRevA.83.012308)
    '''
    
    def __init__(self, numLevels = 0, omega = 0, delta = 0, name=None):
        '''
        Constructor
        '''
        super(SNO, self).__init__(dim=numLevels, name=name)
        self.omega = omega
        self.delta = delta
    
    @property    
    def Hnat(self):
        '''
        Create the diagonal natural Hamiltonian from the level splitting and anharmonicity.
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
    def __init__(self, numLevels = 0, omega = 0, delta = 0, name=None):
        super(SCQubit, self).__init__(numLevels, omega, delta, name=name)

    #Define some effective spin operators in the qubit manifold (lowest two levels)
    @property
    def pauliZ(self):
        assert self.dim > 1, 'Oops! Defining a Z operator requires two more more dimensions.'
        tmpMat = self.zeros()
        tmpMat[0,0] = -1
        tmpMat[1,1] = 1
        return tmpMat
        
    @property
    def pauliX(self):
        assert self.dim > 1, 'Oops! Defining a X operator requires two more more dimensions.'
        tmpMat = self.zeros()
        tmpMat[0,1] = 1
        tmpMat[1,0] = 1
        return tmpMat

    @property
    def pauliY(self):
        assert self.dim > 1, 'Oops! Defining a Y operator requires two more more dimensions.'
        tmpMat = self.zeros()
        tmpMat[0,1] = -1j
        tmpMat[1,0] = 1j
        return tmpMat
    
    def T1Dissipator(self, T1=0):
        ''' Create a T1 dissipator given a T1 value '''
        return (1/np.sqrt(T1))*self.loweringOp()
        
        
        
class Hamiltonian(object):
    '''
    A generic Hamiltonian class including utilities for changing representations
    '''


    def __init__(self, mat = None):
        '''
        Constructor
        '''
        self.mat = mat
        self.dim = mat.shape[0] if mat is not None else 0
        
        
    def move2interaction_frame(self, transformMat):
        '''
        Helper function to move into an interaction frame 
        '''
        return np.dot(np.dot(transformMat,self.Hlab),transformMat.conj().transpose())
    
    def superOpColStack(self):
        '''
        Return the super-operator for Lindbladian dyanamics column-stacked
        '''
        tmpEye = np.eye(self.dim)
        return np.kron(self.mat.conj(), tmpEye) - np.kron(tmpEye, self.mat)
    
class Dissipator(object):
    '''
    A generic Lindladian dissipator class
    '''
    def __init__(self, mat = None):
        '''
        Constructor
        '''
        self.mat = mat
        self.dim = mat.shape[0] if mat is not None else 0
        
    def superOpColStack(self):
        '''
        Return the super-operator for Lindbladian dynamics column-stacked.
        '''
        tmpEye = np.eye(self.dim)
        return np.kron(self.mat.conj(), self.mat) -0.5*np.kron(tmpEye, np.dot(self.mat.conj().transpose(), self.mat)) - 0.5*np.kron(np.dot(np.transpose(self.mat), self.mat.conj()), tmpEye)
     
    
class Interaction(object):
    '''
    A class for interactions between quantum system
    '''
    def __init__(self, system1 = None, system2 = None, interactionType = None, interactionStrength = None, interactionMat = None):
        self.system1 = system1
        self.system2 = system2
        self.interactionType = interactionType
        self.interactionStrength = interactionStrength
        self.matrix = interactionMat
        self.createMat()
        
    
    def createMat(self):
        ''' Create the matrix representation of the interaction. '''
        if self.matrix is None:
            #Work it out for different interaction types
            if self.interactionType == 'ZZ':
                self.matrix = self.interactionStrength*np.kron(self.system1.pauliZ, self.system2.pauliZ)
            else:
                raise NameError('Unknown interaction type.')
    
            

def expand_hilbert_space(operator, operatorSubSystems, eyeSubSystems, dimensions):
    ''' 
    Helper function for expanding an subsystem operator into the full Hilbert space.
    
    For example if we want a CNOT between qubits 1 and 3 and we have the CNOT matrix there is no easy way to create 
    the full matrix. We have to create np.kron(CNOT,np.eye(2)) and then swap qubits 2 and 3 before and after.
    We can achieve this with:
    CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    expand_hilbert_space(CNOT, [1, 3], 2)
    
    Based on work from Jonathan Hodges. 
    '''
    
    #Turn potential lists into numpy arrays
    operatorSubSystems = np.array(operatorSubSystems)
    eyeSubSystems = np.array(eyeSubSystems) if eyeSubSystems is not None else np.array([])
    dimensions = np.array(dimensions)
    
    #Calculate some dimensions
    dimEye = np.prod(dimensions[eyeSubSystems]) if eyeSubSystems.size > 0 else 1
    
    #Create the full matrix in the wrong order
    tmpMat = np.kron(operator, np.eye(dimEye) )

    #List this wrong order 
    curIndices = np.hstack((operatorSubSystems, eyeSubSystems))
    
    #The total permuation matrix
    totP = np.eye(tmpMat.shape[0])
    
    #Helper function to create permutation matrix for swapping two systems
    def calc_perm_mat(dim1, dim2):
        totDim = dim1*dim2
        tmpP = np.zeros((totDim, totDim))
        for ct1 in range(dim1):
            for ct2 in range(dim2):
                tmpP[(ct1*dim2+ct2, ct2*dim1+ct1)] = 1
        return tmpP
        
    #Now step through each sub system and swap until it is in lexicographical order
    for ct in range(curIndices.size):
        while ct < curIndices[ct]:
            
            #Find out where it currently is
            curInd = np.nonzero(curIndices==ct)[0][0]
            
            #Calculate the permutation matrix for swapping kron(I,B,A,I) to kron(I,A,B,I)
            tmpP = calc_perm_mat(dimensions[curInd], dimensions[curInd-1])
            #Calculate the identity dimensions before and after
            preDim = np.prod(dimensions[curIndices[:curInd-1]]) if curInd > 1 else 1
            postDim = np.prod(dimensions[curIndices[curInd+1:]]) if curInd < dimensions.size-1 else 1
            
            totP = np.dot(np.kron(np.kron(np.eye(preDim), tmpP), np.eye(postDim)), totP)
            
            #Recorder the indices to reflect the swap
            curIndices[[curInd-1,curInd]] = curIndices[[curInd, curInd-1]]
            
    #Apply the permutation matrix and return it
    return np.dot(np.dot(totP, tmpMat), totP.transpose())
    
    
    
    