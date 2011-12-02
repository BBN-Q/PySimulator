'''
Created on Nov 6, 2011

@author: cryan

Functions for evolving the pulse sequence

'''

import numpy as np
from numpy import sin, cos

from scipy.constants import pi
from scipy.linalg import expm

from copy import deepcopy

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
            Htot = deepcopy(systemParams.Hnat)
            
            #Add each of the control Hamiltonians
            for controlct, tmpControl in enumerate(pulseSequence.controlLines):
                tmpPhase = 2*pi*tmpControl.freq*curTime + tmpControl.initialPhase
                if tmpControl.controlType == 'rotating':
                    tmpMat = cos(tmpPhase)*systemParams.controlHams[controlct]['inphase'].matrix + sin(tmpPhase)*systemParams.controlHams[controlct]['quadrature'].matrix
                elif tmpControl.controlType == 'sinusoidal':
                    tmpMat = cos(tmpPhase)*systemParams.controlHams[controlct]['inphase'].matrix
                else:
                    raise TypeError('Unknown control type.')
                tmpMat *= pulseSequence.controlAmps[controlct,timect]
                Htot += tmpMat

            if pulseSequence.H_int is not None:
                #Move the total Hamiltonian into the interaction frame
                Htot.calc_interaction_frame(pulseSequence.H_int, curTime)
                #Propagate the unitary
                totU = np.dot(expm(-1j*2*pi*subTimeStep*Htot.interactionMatrix),totU)
            else:
                #Propagate the unitary
                totU = np.dot(expm(-1j*2*pi*subTimeStep*Htot.matrix),totU)
            
            #Update the times
            tmpTime += subTimeStep
            curTime += subTimeStep
            
    return totU

    
def evolution_lindblad(pulseSequence, systemParams, rhoIn):
    '''
    Main function for evolving a state with Lindladian dissipators conditions.
    
    Currently does not currently properly handle transformation of dissipators into interaction frame. 
    '''
    
    '''
    Main function for evolving a state under unitary conditions
    '''
    
    #Setup the super operators for the dissipators
    
    supDis = np.zeros((systemParams.dim**2, systemParams.dim**2), dtype=np.complex128)
    for tmpDis in systemParams.dissipators:
        supDis += tmpDis.superOpColStack()
        
    #Initialize the propagator
    totF = np.eye(systemParams.dim**2)
    
    #Loop over each timestep in the sequence
    curTime = 0.0
    for timect, timeStep in enumerate(pulseSequence.timeSteps):
        tmpTime = 0.0 
        #Loop over the sub-pixels if we have a finer discretization
        while tmpTime < timeStep:
            #Choose the minimum of the time left or the sub pixel timestep
            subTimeStep = np.minimum(timeStep-tmpTime, pulseSequence.maxTimeStep)

            #Initialize the Hamiltonian to the drift Hamiltonian
            Htot = deepcopy(systemParams.Hnat)
            
            #Add each of the control Hamiltonians
            for controlct, tmpControl in enumerate(pulseSequence.controlLines):
                tmpPhase = 2*pi*tmpControl.freq*curTime + tmpControl.initialPhase
                if tmpControl.controlType == 'rotating':
                    tmpMat = cos(tmpPhase)*systemParams.controlHams[controlct]['inphase'].matrix + sin(tmpPhase)*systemParams.controlHams[controlct]['quadrature'].matrix
                elif tmpControl.controlType == 'sinusoidal':
                    tmpMat = cos(tmpPhase)*systemParams.controlHams[controlct]['inphase'].matrix
                else:
                    raise TypeError('Unknown control type.')
                tmpMat *= pulseSequence.controlAmps[controlct,timect]
                Htot += tmpMat
               
            if pulseSequence.H_int is not None:
                #Move the total Hamiltonian into the interaction frame
                Htot.calc_interaction_frame(pulseSequence.H_int, curTime)
                supHtot = Htot.superOpColStack(interactionMatrix=True)
            else:
                supHtot = Htot.superOpColStack()
            
            
            #Propagate the unitary
            totF = np.dot(expm(subTimeStep*(-1j*2*pi*supHtot + supDis)),totF)
            
            tmpTime += subTimeStep
            curTime += subTimeStep
            
    #Reshape, propagate and reshape again the density matrix
    rhoOut = (np.dot(totF, rhoIn.reshape((systemParams.dim**2,1), order='F'))).reshape((systemParams.dim,systemParams.dim), order='F')
    
            
    return rhoOut

    
    
    
    

