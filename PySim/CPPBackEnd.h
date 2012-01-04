/*
 * Header file exporting functions and importing namespaces
 */

#ifndef CPPBACKEND_H__
#define CPPBACKEND_H__

#include <iostream>
#include <vector>
#include <complex>
#include <math.h>
#include <stdio.h>

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
	size_t dimC2;
	int derivType;
	int optimType;
	Mapcd Ugoal;
	Mapcd rhoStart;
	Mapcd rhoGoal;

	OptimParams(cdouble * UgoalPtr, cdouble * rhoStartPtr, cdouble * rhoGoalPtr, size_t dim, size_t dimC2In) : Ugoal(NULL,0,0), rhoStart(NULL,0,0), rhoGoal(NULL,0,0), dimC2(dimC2In) {
		if (UgoalPtr != NULL){
			new (&Ugoal) Mapcd(UgoalPtr,dim,dim);
		}
		if (rhoStartPtr != NULL){
			new (&rhoStart) Mapcd(rhoStartPtr,dim,dim);
		}
		if (rhoGoalPtr != NULL){
			new (&rhoGoal) Mapcd(rhoGoalPtr,dim,dim);
		}
	};
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
	PropResults(size_t numSteps, size_t dim) : totHams(numSteps), Ds(numSteps), Vs(numSteps), Us(numSteps), Uforward(numSteps+1), Uback(numSteps) {

		for (size_t ct=0; ct < numSteps; ++ct) {
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

//Simulation evolution
void evolve_propagator_CPP(const PulseSequence &, const SystemParams &, const int &,  cdouble *);


//Optimization evolution (returns all intermediate steps and has precalculated interaction frame control Hamiltonians)
void opt_evolve_propagator_CPP(const OptimParams & optimParams, const SystemParams & systemParams, cdouble *** controlHams_int, PropResults & propResults);

//Optimization derivative evaluation
void eval_derivs(const OptimParams & optimParams, const SystemParams & systemParams, cdouble *** controlHams_int, PropResults & propResults, double * derivsPtr);

//Helper function to calculate the fitness of a simulated unitary
double eval_pulse_fitness(const OptimParams &, const PropResults &);


#endif /* CPPBACKEND_H__ */
