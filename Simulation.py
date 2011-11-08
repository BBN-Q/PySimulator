'''
Created on Nov 6, 2011

@author: cryan
'''

import numpy as np

from Evolution import evolution_unitary

def simulate_sequence(pulseSeq, systemParams, rhoIn, simType='unitary'):
    
    tmpU = evolution_unitary(pulseSeq, systemParams)
    
    rhoOut = np.dot(np.dot(tmpU,rhoIn), tmpU.conj().transpose())
    
    return np.real(np.trace(np.dot(systemParams.measurement, rhoOut)))

def simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='unitary'):
    
    results = np.zeros(len(pulseSeqs), dtype=np.float64)

    for seqct, tmpSeq in enumerate(pulseSeqs):
        results[seqct] = simulate_sequence(tmpSeq, systemParams, rhoIn, simType)
        
    return results
        