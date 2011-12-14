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

void evolution_unitary_CPP(const PulseSequence & pulseSeq, const SystemParams & systemParams, cdouble * totUPtr){

	//Map the output array which we'll update
	Map<MatrixXcd> Utot(totUPtr, systemParams.dim, systemParams.dim);
	cout << Utot << endl;

	//Map the control Hamiltonians
	//We use a std::vector of controlHam classes
	std::vector<ControlHamMap> controlHams(pulseSeq.numControlLines);
	for (int controlct = 0; controlct < pulseSeq.numControlLines; ++controlct) {
		new (&controlHams[controlct].inphase) Mapcd(systemParams.controlHams[controlct].inphase, systemParams.dim, systemParams.dim);
		new (&controlHams[controlct].quadrature) Mapcd(systemParams.controlHams[controlct].quadrature, systemParams.dim, systemParams.dim);
	}

	cout << controlHams.size() << endl;
	cout << controlHams[0].quadrature << endl;

	//The total time through the pulse sequence
	double curTime = 0.0;

	for (size_t timect = 0; timect < pulseSeq.numTimeSteps; ++timect) {
		//Time in this timestep
		double tmpTime = 0;



	}

	cout << "Got here!" << endl;

}
