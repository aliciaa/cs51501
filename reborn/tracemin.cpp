#include "tracemin.h"
#include "view.h"
#include "jacobi.h"
#include "linear_solver.h"
#include <mkl.h>
#include <mkl_scalapack.h>
#include <omp.h>
#include <cstdio>
#include <cstring>

#define EIGEN_CONVERGENCE_TOL 1.e-5
#define MAX_NUM_ITER 1000

static const char NTRANSA = 'N',
                  TRANSA = 'T',
                  INCREASING = 'I';
static const double D_ONE = 1.0,
                    D_ZERO = 0.0;
static const char MAT_GXXF[6] = "GXXF";

/* Trace Minimization for generalized eigenvalue problem: AY = BY * diag(S)
 * @input n size of the system
 * @input p number of desired eigenvalues
 * @input A symmetric matrix A
 * @input B s.p.d. matrix B
 * @output Y eigenvectors of the system
 * @output S eigenvalues of the system
 */
void TraceMin1(const MKL_INT n,
							 const MKL_INT p,
							 const MKL_INT *A_i,
							 const MKL_INT *A_j,
							 const double *A_v,
							 const MKL_INT *B_i,
							 const MKL_INT *B_j,
							 const double *B_v,
               double *&Y,
               double *&S)
{
	/*---------------------------------------------------------------------------
	 * declaration of variables
	 *---------------------------------------------------------------------------*/
	double t_start,
				 t_end;
	double cg_start, cg_end, cg_total = 0.0;
	double j_start, j_end, j_total = 0.0;
	MKL_INT s = 2 * p,      								// dimension of the subspace
          info;                           // return values from LAPACK
	int c = p,    							      			// number of converged columns
	    w;							    	      				// width of the matrix orders
	int *orders = NULL;	    		      			// order of annihilation
	
  /*---------------------------------------------------------------------------
	 * create the matries V, BV, BZ, BY, M, X, XP, U, W, Z, AZ, AY, Y, R, T
   * and the vectors MS, eigenvalues, norms and perm
	 *---------------------------------------------------------------------------*/
	MKL_INT *perm = new MKL_INT[s];		      // permuatation
	double *norms = new double[p], 			  	// column norms
         *V     = new double[n * s](),  	// the matrix V
         *BV    = new double[n * s],      // B * V
         *BZ    = new double[n * s],      // B * Z
         *BY    = new double[n * s],      // B * Y; also Q1 after QR factorization
         *AZ    = new double[n * s],      // A * Z
         *AY    = new double[n * s],      // A * Y; also the RHS after projection
         *M     = new double[s * s],      // V^T * B * V or Z^T * A * Z; also eigenvectors
         *U     = new double[s * s],      // Q1^T * AY
         *W     = new double[n * s],      // Q1 * U
         *Z     = new double[n * s],      // V * M after eigen decomposition
         *R     = new double[n * p],      // AY - BY * S
         *MS    = new double[s],          // the magnitude of S
         *TS    = new double[s],          // Ritz values; approximate eigenvalues
         *tau   = new double[s];          // workspace for QR factorization

  Y = new double[n * s];                  // eigenvectors
  S = new double[s];                      // eigenvalues

	/*---------------------------------------------------------------------------
	 * start the timer
	 *---------------------------------------------------------------------------*/
	t_start = omp_get_wtime();

	/*---------------------------------------------------------------------------
	 * start with V = [I; I; ...]
	 *---------------------------------------------------------------------------*/
	for (int i = 0; i < n; ++i) {
    V[(i % s) * n + i] = 1.0;
	}

	/*---------------------------------------------------------------------------
	 * generate the order of annihiliation
	 *---------------------------------------------------------------------------*/
	GenerateAnnihilationOrder(s, w, orders);
  
  int k;
  for (k = 0; k < MAX_NUM_ITER; ++k) {
		/*---------------------------------------------------------------------------
		 * M = V^T B V
		 *---------------------------------------------------------------------------*/
    mkl_dcsrmm(&NTRANSA, &n, &s, &n, &D_ONE, MAT_GXXF, B_v, B_j, B_i, B_i+1, V, &n, &D_ZERO, BV, &n);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, s, s, n, 1.0, V, n, BV, n, 0.0, M, s);
		/*---------------------------------------------------------------------------
		 * Perform eigen decomposition
		 *---------------------------------------------------------------------------*/
    j_start = omp_get_wtime();
		Jacobi1(orders, s, w, M, S);
    j_end = omp_get_wtime();
    j_total += j_end - j_start;

		/*---------------------------------------------------------------------------
		 * MS = S^{-1/2}
		 * M = M * MS
		 * BY = BV * M
		 * Z = V * M
		 * AZ = A * Z 
		 * M = Z^T * AZ
		 *---------------------------------------------------------------------------*/
    vdInvSqrt(s, S, MS);
    for (int j = 0; j < s; ++j) {
      cblas_dscal(s, MS[j], M + j*s, 1);
    }
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, s, 1.0, V, n, M, s, 0.0, Z, n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, s, 1.0, BV, n, M, s, 0.0, BZ, n);
    mkl_dcsrmm(&NTRANSA, &n, &s, &n, &D_ONE, MAT_GXXF, A_v, A_j, A_i, A_i+1, Z, &n, &D_ZERO, AZ, &n);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, s, s, n, 1.0, Z, n, AZ, n, 0.0, M, s);

    /*---------------------------------------------------------------------------
		 * Perform eigen decomposition
		 *---------------------------------------------------------------------------*/
    j_start = omp_get_wtime();
		//Jacobi1(orders, s, w, M, S, false);
		Jacobi2(orders, s, w, M, S);
    j_end = omp_get_wtime();
    j_total += j_end - j_start;
	
#if 0
    ViewDense("S", 1, s, S);
#endif
		/*---------------------------------------------------------------------------
		 * Sort the eigenvalues and get the permuation
		 *---------------------------------------------------------------------------*/
		vdAbs(s, S, MS);
		for (int j = 0; j < s; ++j) {
			perm[j] = j + 1;
		}
    dlasrt2(&INCREASING, &s, MS, perm, &info);
    
		/*---------------------------------------------------------------------------
		 * Permute X and S, and postmultiply Z, BZ and AZ by X
		 *---------------------------------------------------------------------------*/
    LAPACKE_dlapmr(LAPACK_ROW_MAJOR, 1, s, s, M, s, perm);
    LAPACKE_dlapmr(LAPACK_ROW_MAJOR, 1, s, 1, S, 1, perm);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, s, 1.0, Z, n, M, s, 0.0, Y, n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, s, 1.0, BZ, n, M, s, 0.0, BY, n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, s, 1.0, AZ, n, M, s, 0.0, AY, n);
		
		/*---------------------------------------------------------------------------
		 * R = AY - BY * M
		 *---------------------------------------------------------------------------*/
    memcpy(R, BY, n * p * sizeof(double));

#if 0
    if (k == 2) {
      ViewDense("R", n, s, R);
      return;
    }
#endif

    for (int j = 0; j < p; ++j) {
      cblas_dscal(n, S[j], R + j*n, 1);
    }
    cblas_daxpy(n * p, -1.0, AY, 1, R, 1);

		/*---------------------------------------------------------------------------
		 * Test for convergence
		 *---------------------------------------------------------------------------*/
		c = 0;
    for (int j = 0; j < p; ++j) {
      norms[j] = cblas_dnrm2(n, R + j*n, 1);
			if (norms[j] <= EIGEN_CONVERGENCE_TOL * MS[j]) ++c;
		}
		if (k % 20 == 0) {
		  printf("Iter[%d] : Number of converged columns = %d\n", k, c);
      for (int j = 0; j < p; ++j) {
        printf("norms[%d]=%.10lf, ev[%d]=%.10lf\n", j, norms[j], j, S[j]);
      }
		}
		if (c == p) break;
	
		/*---------------------------------------------------------------------------
		 * Perform QR factorization of BY and get Q1
		 *---------------------------------------------------------------------------*/
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, s, BY, n, tau);
    LAPACKE_dorgqr(LAPACK_COL_MAJOR, n, s, s, BY, n, tau);

		/*---------------------------------------------------------------------------
		 * Compute the right-hand-side of the reduced system
		 *---------------------------------------------------------------------------*/
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, s, s, n, 1.0, BY, n, AY, n, 0.0, U, s);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, s, s, 1.0, BY, n, U, s, 0.0, W, n);
    cblas_daxpy(n * s, -1.0, W, 1, AY, 1);

#if 0
    if (k == 20) {
      ViewDense("Q1", n, s, BY);
      ViewDense("RHS", n, s, AY);
      return;
    }
#endif

		/*---------------------------------------------------------------------------
		 * CG / MINRES
		 *---------------------------------------------------------------------------*/
    cg_start = omp_get_wtime();
    linear_solver(A_i, A_j, A_v, BY, AY, V, n, s);
    cg_end = omp_get_wtime();
    cg_total += cg_end - cg_start;
    
    cblas_daxpy(n * s, -1.0, Y, 1, V, 1);
    cblas_dscal(n * s, -1.0, V, 1);
  }

	/*---------------------------------------------------------------------------
	 * stop the timer
	 *---------------------------------------------------------------------------*/
	t_end = omp_get_wtime();
	printf("Total iter = %d\n", k);
	printf("Total time = %.6lf\n", t_end - t_start);
	printf("Jacobi time = %.6lf (average = %.6lf)\n", j_total, j_total / (2 * k));
	printf("Linear time = %.6lf\n", cg_total);

	/*---------------------------------------------------------------------------
	 * deallocate the matrices and vectors
	 *---------------------------------------------------------------------------*/
  delete [] V;
  delete [] BV;
  delete [] BZ;
  delete [] BY;
  delete [] M;
  delete [] U;
  delete [] W;
  delete [] Z;
  delete [] AZ;
  delete [] AY;
  delete [] R;
  delete [] MS;
  delete [] orders;
  delete [] norms;
  delete [] tau;
  delete [] perm;
}

