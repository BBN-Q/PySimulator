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


//Propagate unitary evolution.  It is assumed that the totUPtr points to memory initialized to the identity matrix.
void evolution_unitary_CPP(const PulseSequence & pulseSeq, const SystemParams & systemParams, cdouble * totUPtr){

	//Map the output array which we'll update
	Map<MatrixXcd> totU(totUPtr, systemParams.dim, systemParams.dim);

	//Map the control Hamiltonians
	//We use a std::vector of controlHam classes
	std::vector<ControlHamMap> controlHams(pulseSeq.numControlLines);
	for (int controlct = 0; controlct < pulseSeq.numControlLines; ++controlct) {
		new (&controlHams[controlct].inphase) Mapcd(systemParams.controlHams[controlct].inphasePtr, systemParams.dim, systemParams.dim);
		if (systemParams.controlHams[controlct].quadraturePtr != NULL){
			new (&controlHams[controlct].quadrature) Mapcd(systemParams.controlHams[controlct].quadraturePtr, systemParams.dim, systemParams.dim);
		}
	}

	//Map the timesteps vector
	Map<VectorXd> timeSteps(pulseSeq.timeStepsPtr, pulseSeq.numTimeSteps);

	//Map the drift Hamiltonian
	Mapcd Hnat(systemParams.HnatPtr, systemParams.dim, systemParams.dim);

	//Map the control amplitudes
	Map<MatrixXd> controlAmps(pulseSeq.controlAmpsPtr, pulseSeq.numControlLines, pulseSeq.numTimeSteps);

	//Map the interaction Hamiltonian if we have one
	Mapcd H_int(NULL, 0, 0);
	if (pulseSeq.H_intPtr != NULL){
		new (&H_int) Mapcd(pulseSeq.H_intPtr, systemParams.dim, systemParams.dim);
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

			//Propagate the unitary
			//Using Pade approximant
//			Htot *= -i*TWOPI*subTimeStep;
//			totU = Htot.exp()*totU;

			//Using eigenvalue decomp.
			totU = expm_eigen(Htot, -i*TWOPI*subTimeStep)*totU;

			//Update the times
			tmpTime += subTimeStep;
			curTime += subTimeStep;

		}

	}

}

//Propagate unitary evolution.  It is assumed that the totFPtr points to memory initialized to the identity matrix.
//TODO: There is alot of overlapping code with the unitary evolution.  It should be possible to keep in DRYer.
void evolution_lindblad_CPP(const PulseSequence & pulseSeq, const SystemParams & systemParams, cdouble * totFPtr){
	//Setup the super operators for the dissipators
	size_t dim2 = systemParams.dim*systemParams.dim;
	MatrixXcd supDis = MatrixXcd::Zero(dim2, dim2);
	for (size_t ct = 0; ct < systemParams.dissipatorPtrs.size() ; ++ct) {
		MatrixXcd tmpMat = superOp_colStack_dissipator(Mapcd(systemParams.dissipatorPtrs[ct], systemParams.dim, systemParams.dim));
		supDis += superOp_colStack_dissipator(Mapcd(systemParams.dissipatorPtrs[ct], systemParams.dim, systemParams.dim));
	}

	//Map the output array which we'll update
	Map<MatrixXcd> totF(totFPtr, dim2, dim2);

	//Map the control Hamiltonians
	//We use a std::vector of controlHam classes
	std::vector<ControlHamMap> controlHams(pulseSeq.numControlLines);
	for (int controlct = 0; controlct < pulseSeq.numControlLines; ++controlct) {
		new (&controlHams[controlct].inphase) Mapcd(systemParams.controlHams[controlct].inphasePtr, systemParams.dim, systemParams.dim);
		if (systemParams.controlHams[controlct].quadraturePtr != NULL){
			new (&controlHams[controlct].quadrature) Mapcd(systemParams.controlHams[controlct].quadraturePtr, systemParams.dim, systemParams.dim);
		}
	}

	//Map the timesteps vector
	Map<VectorXd> timeSteps(pulseSeq.timeStepsPtr, pulseSeq.numTimeSteps);

	//Map the drift Hamiltonian
	Mapcd Hnat(systemParams.HnatPtr, systemParams.dim, systemParams.dim);

	//Map the control amplitudes
	Map<MatrixXd> controlAmps(pulseSeq.controlAmpsPtr, pulseSeq.numControlLines, pulseSeq.numTimeSteps);


	//Map the interaction Hamiltonian if we have one
	Mapcd H_int(NULL, 0, 0);
	if (pulseSeq.H_intPtr != NULL){
		new (&H_int) Mapcd(pulseSeq.H_intPtr, systemParams.dim, systemParams.dim);
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

			//Create the column-stack representation
			MatrixXcd supHtot = superOp_colStack_hamiltonian(Htot);

			//Propagate the propagator
			//Using Pade approximant
			totF = (subTimeStep*(i*TWOPI*supHtot + supDis)).exp()*totF;

			//Update the times
			tmpTime += subTimeStep;
			curTime += subTimeStep;

		}

	}

}




