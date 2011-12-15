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
			if (pulseSeq.H_intPtr != NULL){
				Htot = move2interaction_frame(H_int, curTime, Htot);
			}

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





