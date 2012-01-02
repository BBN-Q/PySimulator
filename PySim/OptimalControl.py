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
from scipy.linalg import eigh
from scipy.optimize import fmin_l_bfgs_b

import matplotlib.pyplot as plt

from PulseSequence import PulseSequence
from QuantumSystems import Hamiltonian

from Evolution import expm_eigen

#Try to load the CPPBackEnd
try:
    import PySim.CySim
    CPPBackEnd = True
except ImportError:
    CPPBackEnd = False

class PulseParams(PulseSequence):
    '''
    For now just a container for pulse optimization parameters.  Subclasses a PulseSequence as it has to define similar things.
    '''
    def __init__(self):
        super(PulseParams, self).__init__()
        self.numChannels = 0
        self.numPoints = 0
        self.startControlAmps = None  #Initial guess for the pulse
        self.fTol = 1e-4    #optimization paramter: will exit when difference in fidelity is less than this. 
        self.maxfun = 15000
        self.derivType = 'approx'

def create_random_pulse(numChannels, numPoints):
    '''
    Helper function to create smooth pulse starting point.
    '''
    #TODO: return something besides ones
    return 2e6*np.ones((numChannels, numPoints))


def calc_control_Hams(optimParams, systemParams):
    '''
    A helper function to calculate the control Hamiltonians in the interaction frame.  This only needs to be done once per opimization. 
    '''
    #We'll store them in a numControlHamsxnumTimeSteps array
    controlHams = np.zeros((systemParams.numControlHams, optimParams.numTimeSteps, systemParams.dim, systemParams.dim), dtype = np.complex128)
    
    #Now loop over each timestep
    curTime = 0.0
    for timect, timeStep in enumerate(optimParams.timeSteps):
        
        #Loop over each of the control Hamiltonians
        for controlct, tmpControl in enumerate(optimParams.controlLines):
            tmpPhase = 2*pi*tmpControl.freq*curTime + tmpControl.phase
            if tmpControl.controlType == 'rotating':
                tmpHam = Hamiltonian(cos(tmpPhase)*systemParams.controlHams[controlct]['inphase'].matrix + sin(tmpPhase)*systemParams.controlHams[controlct]['quadrature'].matrix)
            elif tmpControl.controlType == 'sinusoidal':
                tmpHam = Hamiltonian(cos(tmpPhase)*systemParams.controlHams[controlct]['inphase'])
            else:
                raise KeyError('Unknown control type.')

            if optimParams.H_int is not None:
                #Move the total Hamiltonian into the interaction frame
                tmpHam.calc_interaction_frame(optimParams.H_int, curTime)
                controlHams[controlct, timect] = tmpHam.interactionMatrix + optimParams.H_int.matrix
            else:
                #Just store the matrix
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
    Vs = np.zeros((optimParams.timeSteps.size, systemParams.dim, systemParams.dim), dtype=np.complex128)
    Ds = np.zeros((optimParams.timeSteps.size, systemParams.dim), dtype=np.float64)
    totHams = np.zeros_like(timeStepUs)
    
    #Loop over each timestep in the sequence
    curTime = 0.0
    for timect, timeStep in enumerate(optimParams.timeSteps):
        #Initialize the Hamiltonian to the drift Hamiltonian
        Htot = deepcopy(systemParams.Hnat)

        if optimParams.H_int is not None:
            #Move the total Hamiltonian into the interaction frame
            Htot.calc_interaction_frame(optimParams.H_int, curTime)
            Htot.matrix = np.copy(Htot.interactionMatrix)
        
        #Add each of the control Hamiltonians
        for controlct in range(optimParams.numControlLines):
            Htot += optimParams.controlAmps[controlct, timect]*controlHams[controlct, timect]

        #Propagate the unitary
        totHams[timect] = Htot.matrix
        timeStepUs[timect], Ds[timect], Vs[timect] = expm_eigen(Htot.matrix, -1j*2*pi*timeStep)
        totU = np.dot(timeStepUs[timect],totU)
        
        #Update the times
        curTime += timeStep
            
    return totU, timeStepUs, Vs, Ds, totHams


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
        return -(np.abs(np.trace(np.dot(Usim.conj().T, optimParams.Ugoal)))**2)/optimParams.dimC2
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

    #Shorten some expressions
    dim = systemParams.dim
    
    #Calculate the unitaries associated with the pulse and the diagonalization 
    Usteps, Vs, Ds, totHams = evolution_unitary(optimParams, systemParams, controlHams)[1:]
    
    #Calculate the forward evolution up to each time step
    numSteps = Usteps.shape[0]
#    Uforward = np.zeros((optimParams.numTimeSteps+1, dim, dim), dtype=np.complex128)
#    Uforward[0] = np.eye(dim, dtype=np.complex128)
#    for ct in range(numSteps):
#        Uforward[ct+1] = np.dot(Usteps[ct], Uforward[ct])
        
    Uforward = np.zeros_like(Usteps, dtype=np.complex128)
    Uforward[0] = Usteps[0]
    for ct in range(1,numSteps):
        Uforward[ct] = np.dot(Usteps[ct], Uforward[ct-1])
    
    #And now the backwards evolution
    Uback = np.zeros_like(Usteps, dtype=np.complex128)
    if optimParams.type == 'unitary':
        Uback[-1] = optimParams.Ugoal
    elif optimParams.type == 'state2state':
        Uback[-1] = np.eye(systemParams.dim, dtype = np.complex128)
    else:
        raise KeyError('Unknown optimization type.  Currently handle "unitary" or "state2state"')
    for ct in range(1,numSteps):
        Uback[-(ct+1)] = np.dot(Usteps[-ct].conj().T, Uback[-ct])

    #Now calculate the derivatives
    #We often use the identity that trace(A^\dagger*B) = np.sum(A.conj()*B) but it doesn't seem to be any faster
    derivs = np.zeros((systemParams.numControlHams, numSteps), dtype=np.float64)
    if optimParams.type == 'unitary':
        curOverlap = np.sum(Uforward[-1].conj()*optimParams.Ugoal)
        for timect in range(numSteps):
            #Put the Hz to rad conversion in the timestep
            tmpTimeStep = 2*pi*optimParams.timeSteps[timect]
            for controlct in range(systemParams.numControlHams):
                #See Machnes, S., Sander, U., Glaser, S. J., Fouquieres, P., Gruslys, A., Schirmer, S., & Schulte-Herbrueggen, T. (2010). Comparing, Optimising and Benchmarking Quantum Control Algorithms in a  Unifying Programming Framework. arXiv, quant-ph. Retrieved from http://arxiv.org/abs/1011.4874v2
                if optimParams.derivType == 'exact':
                    #Exact method
                    eigenFrameControlHam = np.dot(Vs[timect].conj().T, np.dot(controlHams[controlct,timect], Vs[timect]))
                    eigenFrameDeriv = np.zeros_like(eigenFrameControlHam)
                    for rowct in range(dim):
                        for colct in range(dim):
                            diff = Ds[timect][rowct] - Ds[timect][colct]
                            if diff < 1e-12:
                                eigenFrameDeriv[rowct, colct] = -1j*tmpTimeStep*eigenFrameControlHam[rowct,colct]*np.exp(-1j*tmpTimeStep*Ds[timect][rowct])
                            else:
                                eigenFrameDeriv[rowct, colct] = eigenFrameControlHam[rowct,colct]*((np.exp(-1j*tmpTimeStep*Ds[timect][rowct]) - np.exp(-1j*tmpTimeStep*Ds[timect][colct]))/diff)
                    propFrameDeriv = np.dot(Vs[timect], np.dot(eigenFrameDeriv, Vs[timect].conj().T))
                    dUjduk = propFrameDeriv
                    if timect == 0:
                        derivs[controlct, timect] =  (2.0/optimParams.dimC2)*np.real(np.sum(Uback[timect].conj()*dUjduk) * curOverlap)
                    else:
                        derivs[controlct, timect] =  (2.0/optimParams.dimC2)*np.real(np.sum(Uback[timect].conj()*np.dot(dUjduk, Uforward[timect-1])) * curOverlap)
    
                elif optimParams.derivType == 'approx':
                    #Approximate method
                    derivs[controlct, timect] = (2/optimParams.dimC2)*tmpTimeStep*np.imag(np.trace(np.dot(Uback[timect].conj().T, \
                            np.dot(controlHams[controlct,timect], Uforward[timect]))) * curOverlap)

                elif optimParams.derivType == 'finiteDiff':
                    #Finite difference approach
                    tmpU1 = expm_eigen(totHams[timect] + 1e-6*controlHams[controlct,timect], -1j*tmpTimeStep)[0]
                    tmpU2 = expm_eigen(totHams[timect] - 1e-6*controlHams[controlct,timect], -1j*tmpTimeStep)[0]
                    dUjduk = (tmpU1-tmpU2)/2e-6
                    if timect == 0:
                        derivs[controlct, timect] =  (2.0/optimParams.dimC2)*np.real(np.sum(Uback[timect].conj()*dUjduk) * curOverlap)
                    else:
                        derivs[controlct, timect] =  (2.0/optimParams.dimC2)*np.real(np.sum(Uback[timect].conj()*np.dot(dUjduk, Uforward[timect-1])) * curOverlap)
                    
                else:
                    raise NameError('Unknown derivative type for unitary search.')
    
    elif optimParams.type == 'state2state':
        rhoSim = np.dot(np.dot(Uforward[-1], optimParams.rhoStart), Uforward[-1].conj().T)
        tmpMult = np.sum(rhoSim.T*optimParams.rhoGoal)
        if optimParams.derivType == 'approx':
            for timect in range(numSteps):
                tmpTimeStep = 2*pi*optimParams.timeSteps[timect]
                rhoj = np.dot(np.dot(Uforward[timect], optimParams.rhoStart), Uforward[timect].conj().T)
                lambdaj = np.dot(np.dot(Uback[timect], optimParams.rhoGoal), Uback[timect].conj().T)
                for controlct in range(systemParams.numControlHams):
                    derivs[controlct, timect] = 2*tmpTimeStep*np.imag(np.sum(lambdaj.conj()*(np.dot(controlHams[controlct,timect], rhoj) - np.dot(rhoj, controlHams[controlct,timect])))*tmpMult)
        else:
            raise NameError('Unknown derivative type for state to state.')
                    
    return -derivs.flatten()
                    
        
def optimize_pulse(optimParams, systemParams):
    '''
    Main entry point for pulse optimization. 
    '''
    
    #Create the initial pulse
    if optimParams.startControlAmps is None:
        curPulse = create_random_pulse(optimParams.numControlLines, optimParams.numTimeSteps)
    else:
        curPulse = np.copy(optimParams.startControlAmps)
        
    #Figure out the dimension (squared) of the computational space from the desired unitary
    #We use this for normalizing the results
    optimParams.dimC2 = np.abs(np.trace(np.dot(optimParams.Ugoal.conj().T, optimParams.Ugoal)))**2 if optimParams.type == 'unitary' else 0
    
    #Calculate the interaction frame Hamiltonians
    controlHams_int = calc_control_Hams(optimParams, systemParams)

    #Rescale time to ensure the derivatives aren't limited by numerical accuracy
    pulseTime = np.sum(optimParams.timeSteps)
    optimParams.timeSteps /= pulseTime
    systemParams.Hnat.matrix *= pulseTime
    if optimParams.H_int is not None:
        optimParams.H_int.matrix *= pulseTime
    curPulse *= pulseTime
    
    '''
    Create some helper functions for the goodness and derivative evaluation
    If we are using the C++ backend then we define some C classes to store C pointers to the data and control Hamiltonians and temporary propagator results
    which we can then pass to the evaluator functions.
    '''
    if CPPBackEnd:
        controlHams_int_CPP = PySim.CySim.PyControlHams_int(controlHams_int)
        optimParams_CPP = PySim.CySim.PyOptimParams(optimParams)
        systemParams_CPP = PySim.CySim.PySystemParams(systemParams)
        propResults_CPP = PySim.CySim.PyPropResults(optimParams.numTimeSteps, systemParams.dim)
        
        def tmpEvalPulse(pulseIn):
            optimParams_CPP.controlAmps = pulseIn.reshape((optimParams.numControlLines, optimParams.numTimeSteps))
            return PySim.CySim.Cy_eval_pulse(optimParams_CPP, systemParams_CPP, controlHams_int_CPP, propResults_CPP)
        
        def tmpEvalDerivs(pulseIn):
            optimParams.controlAmps = pulseIn.reshape((optimParams.numControlLines, optimParams.numTimeSteps))
            return PySim.CySim.Cy_eval_derivs(optimParams_CPP, systemParams_CPP, controlHams_int_CPP, propResults_CPP)
    else:
    
        def tmpEvalPulse(pulseIn):
            optimParams.controlAmps = pulseIn.reshape((optimParams.numControlLines, optimParams.numTimeSteps))
            return eval_pulse(optimParams, systemParams, controlHams_int)
            
        def tmpEvalDerivs(pulseIn):
            optimParams.controlAmps = pulseIn.reshape((optimParams.numControlLines, optimParams.numTimeSteps))
        
        #Code for evaluating goodness of derivatives. 
#        origGoodness = eval_pulse(optimParams, systemParams, controlHams_int)
#        if origGoodness < -0.5:
#            tmpderivs = eval_derivs(optimParams, systemParams, controlHams_int)
#            finiteDerivs = np.zeros(144)
#            for timect in range(144):
#                optimParams.controlAmps[1,timect] += 1e-6
#                tmpGoodness = eval_pulse(optimParams, systemParams, controlHams_int)
#                optimParams.controlAmps[1,timect] -= 1e-6
#                finiteDerivs[timect] = 1e6*(tmpGoodness-origGoodness)
#            
#            plt.figure()
#            plt.plot(tmpderivs[144:],'b')
#            plt.plot(finiteDerivs,'g--')
#            plt.show()
            
            return eval_derivs(optimParams, systemParams, controlHams_int)
    
    #We can use these to take into account power limits and to squeeze the pulse down to zero and the start and finish for finite bandwidth concerns
    #We'll use a Gaussian filter to achieve a ramp up and ramp down on the pulse edges 
    #Setup bounds at the maximum drive frequency
    timeStep = pulseTime*optimParams.timeSteps[0]
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

    tmpBounds *= pulseTime
    bounds = [(-x, x) for x in tmpBounds.flatten()]
        
    #Call the scipy minimizer
    optimResults = fmin_l_bfgs_b(tmpEvalPulse, curPulse.flatten(), fprime=tmpEvalDerivs, bounds=bounds, iprint=0, maxfun=optimParams.maxfun)
    
    #Reshape the optimized pulse from a 1D vector
    foundPulse = optimResults[0].reshape((optimParams.numControlLines, optimParams.numTimeSteps))
   
#    #Rescale time
    optimParams.timeSteps *= pulseTime
    systemParams.Hnat.matrix /= pulseTime
    if optimParams.H_int is not None:
        optimParams.H_int.matrix /= pulseTime
    curPulse /= pulseTime
    foundPulse /= pulseTime
   
    optimParams.startControlAmps = curPulse
    optimParams.controlAmps = foundPulse
        
    
    
    