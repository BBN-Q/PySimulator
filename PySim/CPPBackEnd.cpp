/*
 * C++ backend for the state/unitary/superoperator propagation for the PySim package.
 *
 * Generically we take in pointers to the numpy input data and preallocated output data.
 * We then use Eigen::Map to access manipulate it and calculate the output data.
 *
 */

#include "CPPBackEnd.h"

class controlHam
{
public:
	Mapcd inphase;
	Mapcd quadrature;
	controlHam() : inphase(NULL,0,0), quadrature(NULL,0,0) {};
};

void evolution_unitary_CPP(int numControlLines, int numTimeSteps, int dim, double * Hnat, double * timeSteps, cdouble ** Hcontrols, cdouble * totUPtr){

	//Map the output array which we'll update
	Map<MatrixXcd> Utot(totUPtr, dim, dim);
	cout << Utot << endl;

	//Map the control Hamiltonians
	//This actually involves a copy but it shouldn't be too expensive
	//We use a std::vector of pairs of maps
	std::vector<controlHam> controlHams(numControlLines);
	for (int controlct = 0; controlct < numControlLines; ++controlct) {
		new (&controlHams[controlct].inphase) Mapcd(Hcontrols[2*controlct], dim, dim);
		new (&controlHams[controlct].quadrature) Mapcd(Hcontrols[2*controlct+1], dim, dim);
	}

	cout << controlHams.size() << endl;
	cout << controlHams[0].quadrature << endl;


	cout << "Got here!" << endl;

}
