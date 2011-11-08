'''
Created on Nov 6, 2011

@author: cryan
'''

class SystemParams(object):
    '''
    A class containing all the system parameters - mainly Hamiltonians.
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.controlHams = []
        self.numControlHams = 0
        
        
    def add_control_ham(self, inphase = None, quadrature = None):
        tmpControlHam = {}
        tmpControlHam['inphase'] = inphase
        tmpControlHam['quadrature'] = quadrature
        self.controlHams.append(tmpControlHam)
        self.numControlHams += 1