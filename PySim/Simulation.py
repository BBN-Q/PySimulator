'''
Created on Nov 6, 2011

@author: cryan
'''

import numpy as np

import multiprocessing
from functools import partial 

import time

from progressbar import Percentage, Bar, ProgressBar, ETA

from Evolution import evolution_unitary, evolution_lindblad

def simulate_sequence(pulseSeq=None, systemParams=None, rhoIn=None, simType='unitary'):
    '''
    Simulate a single pulse sequence and return the expectation value of the measurement.
    '''
    if simType == 'unitary':
        totProp = evolution_unitary(pulseSeq, systemParams)
        if rhoIn is not None:
            rhoOut = np.dot(np.dot(totProp,rhoIn), totProp.conj().transpose())
        else:
            rhoOut = None
    elif simType == 'lindblad':
        totProp = evolution_lindblad(pulseSeq, systemParams, rhoIn)
        #Reshape, propagate and reshape again the density matrix
        rhoOut = (np.dot(totProp, rhoIn.reshape((systemParams.dim**2,1), order='F'))).reshape((systemParams.dim,systemParams.dim), order='F')
    else:
        raise NameError('Unknown simulation type.')
    
    #Return the expectation value of the measurement operator the unitary and the rhouut
    if systemParams.measurement is not None:    
        measOut =  np.real(np.trace(np.dot(systemParams.measurement, rhoOut)))
    else:
        measOut = None
    
    #Return everything
    return measOut, totProp, rhoOut
    
def simulate_sequence_stack(pulseSeqs, systemParams, rhoIn, simType='unitary'):
    '''
    Helper function to simulate a series of pusle sequences with parallelization over multiple cores and progress bar output.
    '''
    
    #Setup a partial function that only takes the sequence
    partial_simulate_sequence = partial(simulate_sequence, systemParams=systemParams, rhoIn=rhoIn, simType=simType)

    #Setup a pool of worker threads
    pool = multiprocessing.Pool()
    
    #Map all the pulse sequences into a results list in parallel
    results = pool.map_async(partial_simulate_sequence, pulseSeqs, 1)
    
    #Close the pool to new jobs 
    pool.close()
    
    numSeqs = len(pulseSeqs)
    pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=numSeqs).start()
  
    while True:
        if not results.ready():
            pbar.update(numSeqs-results._number_left)
            time.sleep(0.1)
        else:
            break
    
    pbar.finish()
    
    tmpResults = results.get()
            
    #Extract the measurement results into a numpy array. 
    measResults = np.array([tmpResult[0] for tmpResult in tmpResults], dtype=np.float64)
    
    props = [tmpResult[1] for tmpResult in tmpResults]
        
    rhos = [tmpResult[2] for tmpResult in tmpResults]
    
    return measResults, props, rhos