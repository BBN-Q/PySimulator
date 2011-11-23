'''
Created on Nov 6, 2011

@author: cryan
'''

import numpy as np

from QuantumSystems import Interaction, Hamiltonian, expand_hilbert_space

class SystemParams(object):
    '''
    A class containing all the system parameters - mainly Hamiltonians.
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.controlHams = []
        self.dissipators = []
        self.subSystems = []
        self.interactions = []
        self.Hnat = None
        
    def add_control_ham(self, inphase = None, quadrature = None):
        ''' Add a control Hamiltonian.  Should  be added in the same order that they are listed as control amplitdues.'''
        tmpControlHam = {}
        tmpControlHam['inphase'] = inphase
        tmpControlHam['quadrature'] = quadrature
        self.controlHams.append(tmpControlHam)
        
    @property
    def numControlHams(self):
        return len(self.controlHams)
        
    def add_sub_system(self, systemIn):
        '''  Add a quantum sub-system for automatic natural Hamiltonian generation.'''
        self.subSystems.append(systemIn)
        
    @property
    def numSubSystems(self):
        return len(self.subSystems)
    
    @property
    def subSystemDims(self):
        dims = np.zeros(self.numSubSystems, dtype=np.int)
        for ct,tmpSys in enumerate(self.subSystems):
            dims[ct] = tmpSys.dim
        return dims

    @property 
    def dim(self):
        return np.prod(self.subSystemDims)
    
    @property
    def subSystemNames(self):
        return [tmpSys.name for tmpSys in self.subSystems]
    
    def add_interaction(self, system1 = None, system2 = None, interactionType = None, interactionStrength = None, interactionMat = None ):
        '''Add an interaction between two sub-systems.'''
        self.interactions.append(Interaction(self.get_subsystem_by_name(system1), self.get_subsystem_by_name(system2), interactionType, interactionStrength, interactionMat))
    
    def find_subsystem_pos(self, systemName):
        ''' Find the position of a particular system'''
        return self.subSystemNames.index(systemName)
    
    def get_subsystem_by_name(self, systemName):
        return self.subSystems[self.find_subsystem_pos(systemName)]
        
    def expand_operator(self, systemName, operator):
        ''' Expand a single system operator over the full Hilbert space.  '''
        #Find what position the single system is 
        sysPos = self.find_subsystem_pos(systemName)
        
        return expand_hilbert_space(operator,sysPos,np.setxor1d(np.array(sysPos),np.arange(self.numSubSystems)),self.subSystemDims);    
    
    def create_full_Ham(self):
        ''' Create the full Hamiltonian with all the interactions'''
        Hnat = np.zeros((self.dim,self.dim), dtype=np.complex128)
        
        #Loop over all the sub-system self Hamiltonians
        for tmpNam, tmpSys in zip(self.subSystemNames, self.subSystems):
            Hnat += self.expand_operator(tmpNam, tmpSys.Hnat)
        
        #Loop over all inter-system interactions
        for tmpInteraction in self.interactions:
            sys1Pos = self.find_subsystem_pos(tmpInteraction.system1.name)
            sys2Pos = self.find_subsystem_pos(tmpInteraction.system2.name)
            Hnat += expand_hilbert_space(tmpInteraction.matrix, np.array([sys1Pos, sys2Pos]), np.setxor1d([sys1Pos, sys2Pos], np.arange(self.numSubSystems)), self.subSystemDims)
        
        self.Hnat = Hamiltonian(Hnat)
        
if __name__ == '__main__':
    
    '''Some hack testing.'''
    from QuantumSystems import *
    Q1 = SCQubit(2, 5e9, name='Q1')
    Q2 = SCQubit(2, 6e9, name='Q2')
    tmpSystem = SystemParams()
    tmpSystem.add_sub_system(Q1)
    tmpSystem.add_sub_system(Q2)
    tmpSystem.add_interaction('Q1', 'Q2', 'ZZ', 1e9)
    print(tmpSystem.subSystemDims)
    print(tmpSystem.expand_operator('Q1', Q1.Hnat))
    tmpSystem.create_full_Ham()
    print(tmpSystem.Hnat.matrix)
        
    
        