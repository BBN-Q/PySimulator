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

#include "HelperFunctions.h"

//Forward declarations of the functions
void evolution_unitary_CPP(const PulseSequence &, const SystemParams &, cdouble *);
void evolution_lindblad_CPP(const PulseSequence &, const SystemParams &, cdouble *);



#endif /* CPPBACKEND_H__ */
