'''
Created on Nov 6, 2011

@author: cryan
'''
import numpy as np

class ControlLine(object):
    def __init__(self, freq = 0, initialPhase = 0, controlType = None, bandwidth=np.inf, maxAmp=np.inf):
        self.freq = freq
        self.initialPhase = initialPhase
        self.controlType = 'rotating' if controlType == None else controlType
        self.bandwidth = bandwidth
        self.maxAmp = maxAmp
        
class PulseSequence(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Empty Constructor
        '''
        self.controlLines = []
        self.timeSteps = np.zeros(0,dtype=np.float64)
        self.controlAmps = None
        self.H_int = None
        self.maxTimeStep = np.Inf
    
    def add_control_line(self, *args, **kwargs):
        self.controlLines.append(ControlLine(*args, **kwargs))

    @property
    def numControlLines(self):
        return len(self.controlLines)
            
    @property
    def numTimeSteps(self):
        return self.timeSteps.size
        
    
    

    
    
    