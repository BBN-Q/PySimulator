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
using Eigen::MatrixXd;
using Eigen::Matrix4d;
using Eigen::Matrix3d;

using Eigen::Matrix3cd;
using Eigen::Vector3cd;

using Eigen::MatrixXi;
using Eigen::Vector3i;

using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::Vector4d;

using Eigen::MatrixBase;
using Eigen::Map;

using Eigen::AngleAxisd;

const double PI = 2*acos(0.0);
const double TWOPI = 2*PI;


using std::cout;
using std::endl;

//Forward declarations of the functions

void evolution_unitary(int numControlLines, int numTimeSteps, int dim, double * Hnat, double * timeSteps, double ** Hcontrols, complex * totU);


#endif /* CPPBACKEND_H__ */
