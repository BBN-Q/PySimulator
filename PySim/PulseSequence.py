'''
Created on Nov 6, 2011

@author: cryan
'''
import numpy as np

class ControlLine(object):
    '''
    A class for control line: basically a modulated microwave carrier. 
    '''
    def __init__(self, freq = 0, phase = 0, controlType = None, bandwidth=np.inf, maxAmp=np.inf):
        self.freq = freq
        self.phase = phase
        #Whether we are taking a rotational or linearly polarized component
        self.controlType = 'rotating' if controlType == None else controlType
        self.bandwidth = bandwidth
        self.maxAmp = maxAmp
        
class PulseSequence(object):
    '''
    A class for describing pulse sequences in terms of control lines, amplitudes of those control lines and the timesteps. 
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
        
    
    

    
    
    