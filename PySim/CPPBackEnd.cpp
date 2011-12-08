/*
 * C++ backend for the state/unitary/superoperator propagation for the PySim package.
 *
 * Generically we take in pointers to the numpy input data and preallocated output data.
 * We then use Eigen::Map to access manipulate it and calculate the output data.
 *
 */

#include "CPPBackEnd.h"

void evolution_unitary(int numControlLines, int numTimeSteps, int dim, double * Hnat, double * timeSteps, double ** Hcontrols, cdouble * totU){

	cout << "Got here!";
}
