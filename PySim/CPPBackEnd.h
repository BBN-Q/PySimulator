/*
 * Header file exporting functions and importing namespaces
 */

#ifndef CPPBACKEND_H__
#define CPPBACKEND_H__

#include <iostream>
#include <vector>
#include <complex>
#include <math.h>
#include <string>

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <unsupported/Eigen/MatrixFunctions>


using Eigen::Matrix;
using Eigen::MatrixXcd;

using Eigen::MatrixBase;
using Eigen::Map;

const double PI = 2*acos(0.0);
const double TWOPI = 2*PI;

using std::cout;
using std::endl;

typedef std::complex<double> cdouble;
typedef Map<MatrixXcd> Mapcd;

class ControlHam
{
public:
	cdouble * inphase;
	cdouble * quadrature;
};

//Some classes/structures to nicely store the data
class PulseSequence{
public:
	size_t numControlLines;
	size_t numTimeSteps;
	double * timeSteps;
};

class SystemParams{
public:
	size_t numControlHams;
	size_t dim;
	std::vector<ControlHam> controlHams;
	cdouble * Hnat;
};

//Forward declarations of the functions
void evolution_unitary_CPP(const PulseSequence &, const SystemParams &, cdouble *);



#endif /* CPPBACKEND_H__ */
