#ifndef TRACE_MIN_1
#define TRACE_MIN_1

#include <mkl_types.h>

#define NUM_TIMINGS 5

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
