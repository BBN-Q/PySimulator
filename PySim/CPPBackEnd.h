/*
 * Header file exporting functions and importing namespaces
 */

#ifndef CPPBACKEND_H__
#define CPPBACKEND_H__

#include <iostream>
#include <vector>
#include <complex>
#include <math.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <unsupported/Eigen/MatrixFunctions>

using Eigen::MatrixXcd;

using Eigen::VectorXd;
using Eigen::MatrixXd;

using Eigen::MatrixBase;
using Eigen::Map;

using Eigen::SelfAdjointEigenSolver;

const double PI = 2*acos(0.0);
const double TWOPI = 2*PI;
const std::complex<double> i = std::complex<double>(0,1);

using std::cout;
using std::endl;

typedef std::complex<double> cdouble;
typedef Map<MatrixXcd> Mapcd;


//Some classes/structures to nicely store the data
class ControlHam
{
public:
	cdouble * inphasePtr;
	cdouble * quadraturePtr;
};

class ControlLine
{
public:
	double freq;
	double phase;
	int controlType; // 0 for linear 1 for rotating
};

class PulseSequence{
public:
	size_t numControlLines;
	size_t numTimeSteps;
	double * timeStepsPtr;
	double maxTimeStep;
	double * controlAmpsPtr;
	std::vector<ControlLine> controlLines;
	cdouble * H_intPtr;
};

class SystemParams{
public:
	size_t numControlHams;
	size_t dim;
	std::vector<ControlHam> controlHams;
	std::vector<cdouble *> dissipatorPtrs;
	cdouble * HnatPtr;
};

class OptimParams : public PulseSequence {
public:
	cdouble * UgoalPtr;
	MatrixXcd Ugoal;
	size_t dimC2;
};

//Class for holding intermediate propagator evolution results
class PropResults{
public:
	//The total Hamiltonian at each step
	std::vector<MatrixXcd> totHams;
	//The eigenvalues of the Hamiltonian
	std::vector<MatrixXd> Ds;
	//The eigenvectors of the Hamiltonian
	std::vector<MatrixXcd> Vs;
	//The unitary of each time step
	std::vector<MatrixXcd> Us;
	//The unitary after each time step
	std::vector<MatrixXcd> Uforward;
	//The reverse-time unitary up to each time step
	std::vector<MatrixXcd> Uback;
	//The total unitary
	MatrixXcd totU;

	//Constructor initializes all the memory given the number of timesteps and the system dimensions
	PropResults(int numSteps, int dim) : totHams(numSteps), Ds(numSteps), Vs(numSteps), Us(numSteps), Uforward(numSteps+1), Uback(numSteps) {

		for (int ct=0; ct < numSteps; ++ct) {
			totHams[ct] = MatrixXcd::Zero(dim,dim);
			Ds[ct] = MatrixXd::Zero(dim,1);
			Vs[ct] = MatrixXcd::Zero(dim,dim);
			Us[ct] = MatrixXcd::Zero(dim,dim);
			Uforward[ct] = MatrixXcd::Zero(dim,dim);
			Uback[ct] = MatrixXcd::Zero(dim,dim);
			totU = MatrixXcd::Zero(dim,dim);
		}
		Uforward[numSteps] = MatrixXcd::Zero(dim,dim);
	};
};



#include "HelperFunctions.h"

//Forward declarations of the functions
void evolve_propagator_CPP(const PulseSequence &, const SystemParams &, const int &,  cdouble *);



#endif /* CPPBACKEND_H__ */
