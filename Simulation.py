'''
Created on Nov 6, 2011

@author: cryan
'''

import numpy as np

import multiprocessing
from functools import partial 

from Evolution import evolution_unitary, evolution_lindblad

def simulate_sequence(pulseSeq=None, systemParams=None, rhoIn=None, simType='unitary'):
    
    if simType == 'unitary':
        tmpU = evolution_unitary(pulseSeq, systemParams)
        rhoOut = np.dot(np.dot(tmpU,rhoIn), tmpU.conj().transpose())
    elif simType == 'lindblad':
        rhoOut = evolution_lindblad(pulseSeq, systemParams, rhoIn)
    else:
        raise NameError('Unknown simulation type.')
    
    #Return the expectation value of the measurement operator    
    return np.real(np.trace(np.dot(systemParams.measurement, rhoOut)))
    
def simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='unitary'):
    
    #Setup a partial function that only takes the sequence
    partial_simulate_sequence = partial(simulate_sequence, systemParams=systemParams, rhoIn=rhoIn, simType=simType)

    #Setup a pool of worker threads
    pool = multiprocessing.Pool()
    
    #Map all the pulse sequences into a results list in parallel
    results = pool.map(partial_simulate_sequence, pulseSeqs, 1)
    
    #Bring the threads back together 
    pool.close()
    pool.join()
    
    #Convert back from a list to a numpy array. 
    results = np.array(results, dtype=np.float64)
        
    return results
        