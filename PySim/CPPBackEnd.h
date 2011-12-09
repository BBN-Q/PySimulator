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


//Forward declarations of the functions

void evolution_unitary_CPP(int numControlLines, int numTimeSteps, int dim, double * Hnat, double * timeSteps, cdouble ** Hcontrols, cdouble * totU);


#endif /* CPPBACKEND_H__ */
