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


//Helper function to calculate the fitness of a simulated unitary
double eval_unitary_fitness(const OptimParams & optimParams, const PropResults & propResults){
	double tmpResult = abs(optimParams.Ugoal.conjugate().cwiseProduct(propResults.totU).sum());
	return -(tmpResult*tmpResult)/optimParams.dimC2;
}

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
   		propResults.Uforward[timect+1] = propResults.Us[timect]*propResults.Uforward[timect];
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
	for(int timect=optimParams.numTimeSteps-2; timect >= 0; --timect){
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
			MatrixXcd dUjdUk;
			switch (optimParams.derivType) {
				//Finite difference approach
				case 0:{
					MatrixXcd tmpU1 = expm_eigen(propResults.totHams[timect] + 1e-6*Map<MatrixXcd>(controlHams_int[controlct][timect],dim,dim), -1j*tmpTimeStep);
					MatrixXcd tmpU2 = expm_eigen(propResults.totHams[timect] - 1e-6*Map<MatrixXcd>(controlHams_int[controlct][timect],dim,dim), -1j*tmpTimeStep);
					dUjdUk = (tmpU1-tmpU2)/2e-6;
				}break;
				//Approximate gradients
				case 1:
					dUjdUk = -i*tmpTimeStep*Map<MatrixXcd>(controlHams_int[controlct][timect],dim,dim)*propResults.Us[timect];
					break;

				//Exact gradients
				case 2:
					//Move the control Hamiltonian into the eigenbasis
					MatrixXcd eigenFrameControlHam = propResults.Vs[timect].adjoint()*Map<MatrixXcd>(controlHams_int[controlct][timect],dim,dim)*propResults.Vs[timect];
					//Initialize the derivative of the unitary step in the eigenbasis
					MatrixXcd eigenFrameDeriv = MatrixXcd::Zero(dim,dim);
					for (size_t rowct = 0; rowct < dim; ++rowct) {
						for (size_t colct = 0; colct < dim; ++colct) {
							//Calculate the difference in eigenvalues
							double diff = propResults.Ds[timect](rowct) - propResults.Ds[timect](colct);
							//If it is close to zero
							if (diff < 1e-12) {
								//For some bizarre reason I have to cast everything
								eigenFrameDeriv(rowct,colct) = static_cast<cdouble>(-1i*tmpTimeStep)*static_cast<cdouble>(eigenFrameControlHam(rowct,colct))*exp(-i*tmpTimeStep*propResults.Ds[timect](rowct));
							}
							else{
								eigenFrameDeriv(rowct,colct) = eigenFrameControlHam(rowct,colct)*(exp(-i*tmpTimeStep*propResults.Ds[timect](rowct)) - exp(-i*tmpTimeStep*propResults.Ds[timect](colct)))/diff;
							}
						}
					}
					//Convert back to the standard basis
					dUjdUk = propResults.Vs[timect]*eigenFrameDeriv*propResults.Vs[timect].adjoint();
					break;
			}

			derivsMat(controlct, timect) = (2.0/optimParams.dimC2)*(propResults.Uback[timect].conjugate().cwiseProduct(dUjdUk*propResults.Uforward[timect]).sum()*curOverlap).real();

		}
	}
}

