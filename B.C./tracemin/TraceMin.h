#ifndef TRACE_MIN_1
#define TRACE_MIN_1

#include <petscmat.h>

/* Trace Minimization for finding the smallest p eigenpairs of the generalized eigenvalue problem:
 * 		AY = BY * diag(S)
 * @input n size of the system
 * @input p number of desired eigenvalues
 * @input A symmetric matrix A
 * @input B s.p.d. matrix B
 * @output Y eigenvectors of the system
 * @output S eigenvalues of the system
 */
void TraceMin1(const char* fileO,
               const PetscInt n,
							 const PetscInt p,
							 const Mat A,
							 const Mat B,
							 Mat & Y,
							 Vec & S);

#endif // TRACE_MIN
