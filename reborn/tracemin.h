#ifndef TRACE_MIN_1
#define TRACE_MIN_1

#include <mkl_types.h>

#define NUM_TIMINGS 5
#define EIGEN_CONVERGENCE_TOL 1.e-4
#define MAX_NUM_ITER 10000
#define PACE 100

static const char NTRANSA = 'N',
                  TRANSA = 'T',
                  INCREASING = 'I';
static const double D_ONE = 1.0,
                    D_ZERO = 0.0;
static const char MAT_GXXF[6] = "GXXF";

/* Trace Minimization for finding the smallest p eigenpairs of the generalized eigenvalue problem:
 * 		AY = BY * diag(S)
 * @input n size of the system
 * @input p number of desired eigenvalues
 * @input A symmetric matrix A
 * @input B s.p.d. matrix B
 * @output Y eigenvectors of the system
 * @output S eigenvalues of the system
 * @output rnorms residual column norms
 * @output timing timing
 */
void TraceMin1(const MKL_INT n,
							 const MKL_INT p,
							 const MKL_INT *AI,
							 const MKL_INT *AJ,
							 const double *AV,
							 const MKL_INT *BI,
							 const MKL_INT *BJ,
							 const double *BV,
               double *&Y,
               double *&S,
               double *&rnorms,
               double *&timing);

#endif // TRACE_MIN
