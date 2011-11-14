'''
Created on Nov 6, 2011

@author: cryan

Functions for evolving the pulse sequence

'''

import numpy as np
from numpy import sin, cos,pi

from scipy.linalg import expm

def evolution_unitary(pulseSequence, systemParams):
    '''
    Main function for evolving a state under unitary conditions
    '''
    
    totU = np.eye(systemParams.dim)
    
    #Loop over each timestep in the sequence
    curTime = 0.0
    for timect, timeStep in enumerate(pulseSequence.timeSteps):
        tmpTime = 0.0 
        #Loop over the sub-pixels if we have a finer discretization
        while tmpTime < timeStep:
            #Choose the minimum of the time left or the sub pixel timestep
            subTimeStep = np.minimum(timeStep-tmpTime, pulseSequence.maxTimeStep)

            #Initialize the Hamiltonian to the drift Hamiltonian
            Htot = np.copy(systemParams.Hnat.mat)
            
            #Add each of the control Hamiltonians
            for controlct, tmpControl in enumerate(pulseSequence.controlLines):
                tmpPhase = 2*pi*tmpControl.freq*curTime + tmpControl.initialPhase
                if tmpControl.controlType == 'rotating':
                    tmpMat = cos(tmpPhase)*systemParams.controlHams[controlct]['inphase'].mat + sin(tmpPhase)*systemParams.controlHams[controlct]['quadrature'].mat
                else:
                    tmpMat = cos(tmpPhase)*systemParams.controlHams[controlct]['inphase'].mat
                Htot += pulseSequence.controlAmps[controlct,timect]*tmpMat
            
            if pulseSequence.H_int is not None:
                #Calculate the current interaction frame transformation
                transformMat = expm((1j*curTime)*pulseSequence.H_int.mat); 
            
                #Move the total Hamiltonian into the interaction frame
                Htot = np.dot(np.dot(transformMat,Htot),transformMat.conj().transpose()) - pulseSequence.H_int.mat
                
            #Propagate the unitary
            totU = np.dot(expm(-1j*subTimeStep*Htot),totU)
            
            tmpTime += subTimeStep
            curTime += subTimeStep
            
    return totU

    
def evolution_lindblad(pulseSequence, systemParams, rhoIn):
    '''
    Main function for evolving a state with Lindladian dissipators conditions
    '''
    
    pass
    
    

