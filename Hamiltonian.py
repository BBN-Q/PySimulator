'''
Created on Nov 6, 2011

@author: cryan
'''

import numpy as np


class Hamiltonian(object):
    '''
    A generic Hamiltonian class including utilities for handling interaction frames 
    '''


    def __init__(self, mat = None):
        '''
        Constructor
        '''
        self.mat = mat
        self.dim = mat.shape[0]
        
        
    def move2interaction_frame(self, transformMat):
        '''
        Helper function to move into an interaction frame 
        '''
        return np.dot(np.dot(transformMat,self.Hlab),transformMat.conj().transpose())
        