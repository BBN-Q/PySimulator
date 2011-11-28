'''
Created on Nov 6, 2011

@author: cryan
'''
import numpy as np

class ControlLine(object):
    def __init__(self, freq = 0, initialPhase = 0, controlType = None):
        self.freq = freq
        self.initialPhase = initialPhase
        self.controlType = 'rotating' if controlType == None else controlType
        
class PulseSequence(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Empty Constructor
        '''
        self.controlLines = []
        self.numControls = 0
        self.timeSteps = np.zeros(0,dtype=np.float64)
        self.controlAmps = None
        self.H_int = None
        self.maxTimeStep = np.Inf
    
    def add_control_line(self, freq = 0, initialPhase = 0, controlType = None):
        self.controlLines.append(ControlLine(freq, initialPhase, controlType))
        self.numControls += 1
        
    @property
    def numTimeSteps(self):
        return self.timeSteps.size
        
    
    

    
    
    