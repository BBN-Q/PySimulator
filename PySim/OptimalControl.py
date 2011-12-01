'''

Created on Nov 25, 2011

@author: cryan

Code for numerical optimal control. 

'''

import numpy as np
from numpy import sin,cos
from copy import deepcopy

from scipy.constants import pi
from scipy.linalg import expm

from scipy.optimize import fmin_tnc, tnc

import scipy.signal

from PulseSequence import PulseSequence
from QuantumSystems import Hamiltonian

class PulseOptimParams(PulseSequence):
    '''
    For now just a container for pulse optimization parameters.  Subclasses a optimParamsuence as it has to define similar things.
    '''
    def __init__(self):
        super(PulseOptimParams, self).__init__()
        self.numChannels = 0
        self.numPoints = 0
        self.startPulse = None
        self.ramps = None
        self.maxControlAmps = None

def create_random_pulse(numChannels, numPoints):
    '''
    Helper function to create smooth pulse starting point.
    '''
    #TODO: return something besides ones
    return 5e6*np.ones((numChannels, numPoints))


def calc_control_Hams(optimParamsuence, systemParams):
    '''
    A helper function to calculate the control Hamiltonians in the interaction frame.  This only needs to be done once per opimization. 
    '''
    #We'll store them in a numControlHamsxnumTimeSteps array
    controlHams = np.zeros((systemParams.numControlHams, optimParamsuence.numTimeSteps, systemParams.dim, systemParams.dim), dtype = np.complex128)
    
    #Now loop over each timestep
    curTime = 0.0
    for timect, timeStep in enumerate(optimParamsuence.timeSteps):
        
        #Loop over each of the control Hamiltonians
        for controlct, tmpControl in enumerate(optimParamsuence.controlLines):
            tmpPhase = 2*pi*tmpControl.freq*curTime + tmpControl.initialPhase
            if tmpControl.controlType == 'rotating':
                tmpHam = Hamiltonian(cos(tmpPhase)*systemParams.controlHams[controlct]['inphase'].matrix + sin(tmpPhase)*systemParams.controlHams[controlct]['quadrature'].matrix)
            elif tmpControl.controlType == 'sinusoidal':
                tmpHam = Hamiltonian(cos(tmpPhase)*systemParams.controlHams[controlct]['inphase'])
            else:
                raise KeyError('Unknown control type.')

        if optimParamsuence.H_int is not None:
            #Move the total Hamiltonian into the interaction frame
            tmpHam.calc_interaction_frame(optimParamsuence.H_int, curTime)
            controlHams[controlct, timect] = tmpHam.interaction_matrix
        else:
            #Propagate the unitary
            controlHams[controlct, timect] = tmpHam.matrix

        #Update the times
        curTime += timeStep
            
    return controlHams

    


def evolution_unitary(optimParams, systemParams, controlHams):
    '''
    Main function for evolving a state under unitary conditions
    '''
    
    totU = np.eye(systemParams.dim)
    timeStepUs = np.zeros((optimParams.timeSteps.size, systemParams.dim, systemParams.dim), dtype=np.complex128)
    
    #Loop over each timestep in the sequence
    curTime = 0.0
    for timect, timeStep in enumerate(optimParams.timeSteps):
        #Initialize the Hamiltonian to the drift Hamiltonian
        Htot = deepcopy(systemParams.Hnat)
        
        #Add each of the control Hamiltonians
        for controlct in range(optimParams.numControlLines):
            Htot += optimParams.controlAmps[controlct, timect]*controlHams[controlct, timect]

        if optimParams.H_int is not None:
            #Move the total Hamiltonian into the interaction frame
            Htot.calc_interaction_frame(optimParams.H_int, curTime)
            #Propagate the unitary
            timeStepUs[timect] = expm(-1j*2*pi*timeStep*Htot.interactionMatrix) 
        else:
            #Propagate the unitary
            timeStepUs[timect] = expm(-1j*2*pi*timeStep*Htot.matrix)

        totU = np.dot(timeStepUs[timect],totU)
        
        #Update the times
        curTime += timeStep
            
    return totU, timeStepUs


def eval_pulse(optimParams, systemParams, controlHams):
    '''
    Evaluate the fidelity of a pulse with respect to the goal unitary or state-to-state transformation
    '''
    
    #TODO: incorporate buffer times
    
    #TODO: allow incoherent distributions
    
    #Calculate the unitary associated with the pulse
    
    Usim = evolution_unitary(optimParams, systemParams, controlHams)[0]
    
    #Use the trace fidelity to evaluate it
    if optimParams.type == 'unitary':
        return -(np.abs(np.trace(np.dot(Usim.conj().T, optimParams.Ugoal)))/systemParams.dim)**2
    elif optimParams.type == 'state2state':
        rhoOut = np.dot(np.dot(Usim, optimParams.rhoStart), Usim.conj().T)
        return -(np.abs(np.trace(np.dot(rhoOut, optimParams.rhoGoal))))**2
    else:
        raise KeyError('Unknown optimization type.  Currently handle "unitary" or "state2state"')
    
    
def eval_derivs(optimParams, systemParams, controlHams):
    '''
    Evaluate the derivatives of each control parameter with respect to the goal unitary or state.
    '''
    #TODO: incorporate buffer times
    
    #TODO: allow incoherent distributions
    
    #Calculate the unitaries associated with the pulse
    Usteps = evolution_unitary(optimParams, systemParams, controlHams)[1]
    
    #Calculate the forward evolution up to each time step
    numSteps = Usteps.shape[0]
    Uforward = np.zeros_like(Usteps, dtype=np.complex128)
    Uforward[0] = Usteps[0]
    for ct in range(1,numSteps):
        Uforward[ct] = np.dot(Usteps[ct], Uforward[ct-1])
        
    #And now the backwards evolution
    Uback = np.zeros_like(Usteps, dtype=np.complex128)
    if optimParams.type == 'unitary':
        Uback[-1] = -optimParams.Ugoal
    elif optimParams.type == 'state2state':
        Uback[-1] = np.eye(systemParams.dim, dtype = np.complex128)
    else:
        raise KeyError('Unknown optimization type.  Currently handle "unitary" or "state2state"')
    for ct in range(1,numSteps):
        Uback[-(ct+1)] = np.dot(Usteps[-ct].conj().T, Uback[-ct])

    #Now calculate the derivatives
    derivs = np.zeros((systemParams.numControlHams, optimParams.numTimeSteps), dtype=np.float64)
    if optimParams.type == 'unitary':
        for timect in range(numSteps):
            for controlct in range(systemParams.numControlHams):
                derivs[controlct, timect] = 2*optimParams.timeSteps[timect]*np.imag(np.trace(np.dot(Uback[timect].conj().T, 
                                            np.dot(controlHams[controlct,timect], Uforward[timect]))) *
                                               np.trace(np.dot(Uforward[timect].conj().T, Uback[timect])))
    elif optimParams.type == 'state2state':
        rhoSim = np.dot(np.dot(Uforward[-1], optimParams.rhoStart), Uforward[-1].conj().T)
        tmpMult = np.trace(np.dot(rhoSim, optimParams.rhoGoal))
        for timect in range(numSteps):
            rhoj = np.dot(np.dot(Uforward[timect], optimParams.rhoStart), Uforward[timect].conj().T)
            lambdaj = np.dot(np.dot(Uback[timect], optimParams.rhoGoal), Uback[timect].conj().T)
            for controlct in range(systemParams.numControlHams):
                derivs[controlct, timect] = 2*optimParams.timeSteps[timect]*np.imag(np.trace(np.dot(lambdaj.conj().T, 
                                            np.dot(controlHams[controlct,timect], rhoj) - np.dot(rhoj, controlHams[controlct,timect])))*tmpMult)
    
    return -derivs.flatten()
                    
        
def optimize_pulse(optimParams, systemParams):
    '''
    Main entry point for pulse optimization. 
    '''
    
    #Create the initial pulse
    if optimParams.controlAmps is None:
        curPulse = create_random_pulse(optimParams.numControlLines, optimParams.numTimeSteps)
    else:
        curPulse = optimParams.controlAmps
    
    #Calculate the interaction frame Hamiltonians
    controlHams_int = calc_control_Hams(optimParams, systemParams)

    #Create some helper functions for the goodness and derivative evaluation
    def tmpEvalPulse(pulseIn):
        optimParams.controlAmps = pulseIn.reshape((optimParams.numControlLines, optimParams.numTimeSteps))
        return eval_pulse(optimParams, systemParams, controlHams_int)
        
    def tmpEvalDerivs(pulseIn):
        optimParams.controlAmps = pulseIn.reshape((optimParams.numControlLines, optimParams.numTimeSteps))
        return eval_derivs(optimParams, systemParams, controlHams_int)
    
    #We can use these to take into account power limits and to squeeze the pulse down to zero and the start and finish for finite bandwidth concerns
    #We'll use a Gaussian filter to achieve a ramp up and ramp down on the pulse edges 
    #Setup bounds at the maximum drive frequency
    timeStep = optimParams.timeSteps[0]
    tmpBounds = np.inf*np.ones_like(curPulse, dtype=np.float64)
    for controlct, tmpControl in enumerate(optimParams.controlLines):
        if tmpControl.bandwidth < np.inf:
            #If the bandwidth is defined as the -3dB point and the frequency response is defined as exp(-(pi*f)**2/alpha then alpha = (pi*f_3dB)**2/log2
            alpha = (np.pi*tmpControl.bandwidth)**2/np.log(2)
            #Then in the impulse response in the time domain is exp(-t^2*alpha) and we want to go out to 2.5sigma to ensure we start small
            tmax = 2.5/np.sqrt(alpha)
            #Number of points we need (assuming equal spacing)
            numPts = np.ceil(tmax/timeStep)
            #Make sure we have enough points in the pulse (this could be handled more gracefully)
            assert optimParams.numTimeSteps > 2*numPts, 'Error: unable to handle such a short pulse with the channel bandwidth.  Need at least {0} points for filtering.'.format(2*numPts+1)
            #Define the Gaussian impulse response and normalize
            impulseResponse = np.exp(-alpha*(timeStep*np.linspace(-numPts, numPts, 2*numPts+1))**2)
            impulseResponse /= np.sum(impulseResponse)
        else:
            numPts = 0
            impulseResponse = np.ones(1, dtype=np.float64)

        tmpBounds[controlct] = tmpControl.maxAmp*np.convolve(impulseResponse, np.ones(optimParams.numTimeSteps-2*numPts))

    bounds = [(-x, x) for x in tmpBounds.flatten()]
        
    #Call the scipy minimizer
    #Look at fmin_tnc.func_globals to see how some variables are defined, they are also stored in scipy.optimize.tnc
    optimResults = fmin_tnc(tmpEvalPulse, curPulse.flatten(), fprime=tmpEvalDerivs, messages=tnc.MSG_ITER, bounds=bounds, fmin=-1)
    
    #Print out the search result
    print(tnc.RCSTRINGS[optimResults[2]])
   
    optimParams.startControlAmps = curPulse
    optimParams.controlAmps = optimResults[0].reshape((optimParams.numControlLines, optimParams.numTimeSteps))
        
    
    
    