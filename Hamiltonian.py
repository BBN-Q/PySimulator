'''
Created on Nov 6, 2011

@author: cryan
'''

import numpy as np


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
        
        
    def calc_interaction_frame(self, transformMat):
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
        
        
    
        
        
    
    
    
    
    
    
    