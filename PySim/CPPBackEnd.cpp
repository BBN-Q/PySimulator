/*
 * C++ backend for the state/unitary/superoperator propagation for the PySim package.
 *
 * Generically we take in pointers to the numpy input data and preallocated output data.
 * We then use Eigen::Map to access, manipulate the data and calculate the output data.
 *
 */

#include "CPPBackEnd.h"

class ControlHamMap
{
public:
	Mapcd inphase;
	Mapcd quadrature;
	ControlHamMap() : inphase(NULL,0,0), quadrature(NULL,0,0) {};
};

void evolve_propagator_CPP(const PulseSequence & pulseSeq, const SystemParams & systemParams, const int & simType, cdouble * totPropPtr){
	/*
	 * Propagate evolution through a pulse sequence.
	 * It is assumed that the totPropPtr points to memory initialized to the initial condition
	 * The simType defines whether we do unitary (0) or lindbladian (1) evolution
	 */

	//Some shorthand for the system dimension and dimension squared
	size_t dim = systemParams.dim;
	size_t dim2 = dim*dim;

	//Map the output array which we'll evolve
	Mapcd totProp(NULL,0,0);
	if(simType == 0){
		new (&totProp) Mapcd(totPropPtr, dim, dim);
	}
	else{
		new (&totProp) Mapcd(totPropPtr, dim2, dim2);
	}

	//If necessary setup the super operators for the dissipators
	MatrixXcd supDis;
	if(simType==1){
		supDis = MatrixXcd::Zero(dim2, dim2);
		for (size_t ct = 0; ct < systemParams.dissipatorPtrs.size() ; ++ct) {
			supDis += superOp_colStack_dissipator(Mapcd(systemParams.dissipatorPtrs[ct], dim, dim));
		}
	}

	//Map the control Hamiltonians
	//We use a std::vector of controlHam classes
	std::vector<ControlHamMap> controlHams(pulseSeq.numControlLines);
	for (int controlct = 0; controlct < pulseSeq.numControlLines; ++controlct) {
		new (&controlHams[controlct].inphase) Mapcd(systemParams.controlHams[controlct].inphasePtr, dim, dim);
		if (systemParams.controlHams[controlct].quadraturePtr != NULL){
			new (&controlHams[controlct].quadrature) Mapcd(systemParams.controlHams[controlct].quadraturePtr, dim, dim);
		}
	}

	//Map the timesteps vector
	Map<VectorXd> timeSteps(pulseSeq.timeStepsPtr, pulseSeq.numTimeSteps);

	//Map the drift Hamiltonian
	Mapcd Hnat(systemParams.HnatPtr, dim, dim);

	//Map the control amplitudes
	Map<MatrixXd> controlAmps(pulseSeq.controlAmpsPtr, pulseSeq.numControlLines, pulseSeq.numTimeSteps);

	//Map the interaction Hamiltonian if we have one
	Mapcd H_int(NULL, 0, 0);
	if (pulseSeq.H_intPtr != NULL){
		new (&H_int) Mapcd(pulseSeq.H_intPtr, dim, dim);
	}

	//The total time through the pulse sequence
	double curTime = 0.0;

	for (size_t timect = 0; timect < pulseSeq.numTimeSteps; ++timect) {
		//Time in this timestep
		double tmpTime = 0.0;
		while (tmpTime + 1e-15 < timeSteps(timect)) {
			//Choose the minimum of the time left or the sub pixel timestep
			double subTimeStep = std::min(timeSteps(timect)-tmpTime, pulseSeq.maxTimeStep);

			//Initialize the Hamiltonian to the drift Hamitlonian
			MatrixXcd Htot = Hnat;

			//Add each of the control Hamiltonians
			for (size_t controlct = 0; controlct < pulseSeq.numControlLines; ++controlct) {
				double tmpPhase = TWOPI*pulseSeq.controlLines[controlct].freq*curTime + pulseSeq.controlLines[controlct].phase;
				//Linearly polarized r.f. field.
				if (pulseSeq.controlLines[controlct].controlType == 0){
					Htot += controlAmps(controlct, timect)*cos(tmpPhase)*controlHams[controlct].inphase;
				}
				//Rotating field
				else{
					Htot += controlAmps(controlct, timect)*(cos(tmpPhase)*controlHams[controlct].inphase + sin(tmpPhase)*controlHams[controlct].quadrature);
				}

			}

			//If necessary move into the interaction frame
			if (pulseSeq.H_intPtr != NULL) 	Htot = move2interaction_frame(H_int, curTime, Htot);

			if(simType == 0){
				//Propagate the unitary
				//Using Pade approximant
	//			Htot *= -i*TWOPI*subTimeStep;
	//			totU = Htot.exp()*totU;

				//Using eigenvalue decomp.
				totProp = expm_eigen(Htot, -i*TWOPI*subTimeStep)*totProp;
			}
			else{
				//Create the column-stack representation
				MatrixXcd supHtot = superOp_colStack_hamiltonian(Htot);

				//Propagate the propagator
				//Using Pade approximant
				totProp = (subTimeStep*(i*TWOPI*supHtot + supDis)).exp()*totProp;
			}
			//Update the times
			tmpTime += subTimeStep;
			curTime += subTimeStep;
		}
	}
}

//Helper function to evolve the unitary propagator for optimal control.
//This stores all intermediate results in propResults structure

void opt_evolve_propagator_CPP(const OptimParams & optimParams, const SystemParams & systemParams, cdouble *** controlHams_int, PropResults & propResults){

	size_t dim = systemParams.dim;

	//Map the timesteps vector
	Map<VectorXd> timeSteps(optimParams.timeStepsPtr, optimParams.numTimeSteps);

	//Map the drift Hamiltonian
	Mapcd Hnat(systemParams.HnatPtr, dim, dim);

	//Map the control amplitudes
	Map<MatrixXd> controlAmps(optimParams.controlAmpsPtr, optimParams.numControlLines, optimParams.numTimeSteps);

	//Map the interaction Hamiltonian if we have one
	Mapcd H_int(NULL, 0, 0);
	if (optimParams.H_intPtr != NULL){
		new (&H_int) Mapcd(optimParams.H_intPtr, dim, dim);
	}

	//Initialize the total unitary to the identity
	propResults.Uforward[0].setIdentity();

	//Loop over each timestep in the sequence
    double curTime = 0.0;
    MatrixXcd Htot(dim,dim);
    for (int timect = 0; timect < optimParams.numTimeSteps; ++timect) {
    	//Initialize the Hamiltonian to the drift Hamiltonian in the appropriate frame
    	Htot = (optimParams.H_intPtr != NULL) ? move2interaction_frame(H_int, curTime, Hnat) : Hnat;

    	//Add each of the control Hamiltonians
    	for (size_t controlct = 0; controlct < optimParams.numControlLines; ++controlct) {
    		Htot += controlAmps(controlct,timect)*Map<MatrixXcd>(controlHams_int[controlct][timect], dim, dim);
    	}

    	//Propagate the unitary
    	propResults.totHams[timect] = Htot;
		SelfAdjointEigenSolver<MatrixXcd> es(Htot);
   		propResults.Ds[timect] = es.eigenvalues();
   		propResults.Vs[timect] = es.eigenvectors();
    	propResults.Us[timect] = propResults.Vs[timect]*((-i*TWOPI*timeSteps[timect]*propResults.Ds[timect]).array().exp().matrix().replicate(1,dim).cwiseProduct(propResults.Vs[timect].adjoint()));
    	propResults.Uforward[timect+1] = propResults.Us[timect]*propResults.Uforward[timect-1];
    }
    propResults.totU = propResults.Uforward[optimParams.numTimeSteps];
}

void eval_derivs(const OptimParams & optimParams, const SystemParams & systemParams, cdouble *** controlHams_int, PropResults & propResults, double * derivsPtr){

	size_t dim = systemParams.dim;

	//Map the timesteps vector
	Map<VectorXd> timeSteps(optimParams.timeStepsPtr, optimParams.numTimeSteps);

	//Calculate the unitaries associated with the pulse and the diagonalization
	opt_evolve_propagator_CPP(optimParams, systemParams, controlHams_int, propResults);

	//Calculate the backward evolution
	//TODO: state2state
	propResults.Uback[optimParams.numTimeSteps-1] = optimParams.Ugoal;
	for(size_t timect=optimParams.numTimeSteps-2; timect >= 0; --timect){
		propResults.Uback[timect] = propResults.Us[timect+1].adjoint()*propResults.Uback[timect+1];
	}

	//Now calculate the derivatives
    //We often use the identity that trace(np.dot(A^\dagger,B)) = np.sum(A.conj()*B)
	Map<MatrixXd> derivsMat(derivsPtr, optimParams.numControlLines, optimParams.numTimeSteps);

	//Current trace overlap
	cdouble curOverlap = (propResults.totU.conjugate().cwiseProduct(optimParams.Ugoal)).sum();
	for (size_t timect = 0; timect < optimParams.numTimeSteps; ++timect) {
		//Put the Hz to rad conversion in the timestep
        double tmpTimeStep = TWOPI*timeSteps(timect);
		for (size_t controlct = 0; controlct < optimParams.numControlLines; ++controlct) {
			//Finite difference approach
			MatrixXcd tmpU1 = expm_eigen(propResults.totHams[timect] + 1e-6*Map<MatrixXcd>(controlHams_int[controlct][timect],dim,dim), -1j*tmpTimeStep);
			MatrixXcd tmpU2 = expm_eigen(propResults.totHams[timect] - 1e-6*Map<MatrixXcd>(controlHams_int[controlct][timect],dim,dim), -1j*tmpTimeStep);
            MatrixXcd dUjdUk = (tmpU1-tmpU2)/2e-6;
            derivsMat(controlct, timect) = (2.0/optimParams.dimC2)*(propResults.Uback[timect].conjugate().cwiseProduct(dUjdUk*propResults.Uforward[timect]).sum()*curOverlap).real();
		}
	}
}

/*



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

    */

