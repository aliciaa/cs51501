#ifndef LINEAR_SOLVER
#define LINEAR_SOLVER

#define LINEAR_SOLVER_MAX_ITER 400
#define LINEAR_SOLVER_ABS_TOL 1e-9
#define LINEAR_SOLVER_REL_TOL 1e-5
//#define USE_INTEL_MKL

//#ifdef USE_INTEL_MKL
#include "mkl_types.h"
typedef MKL_INT LINEAR_INT;
//#else
//typedef long long LINEAR_INT;
//#endif

void linear_solver(const LINEAR_INT* A_ia,
                   const LINEAR_INT* A_ja,
                   const double* A_values, // CSR format of full matri A
                   const double* Q1,          // n * s double, column major
                   const double* rhs,         // n * s double, column major
                   double* sol,    // n * s double, column major
                   LINEAR_INT n,
                   LINEAR_INT r);

#endif // LINEAR_SOLVER
