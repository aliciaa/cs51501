#include "tracemin.h"
#include "view.h"
#include "jacobi.h"
#include "linear_solver.h"
#include <mkl.h>
#include <mkl_scalapack.h>
#include <omp.h>
#include <cstdio>
#include <cstring>

#define EIGEN_CONVERGENCE_TOL 1.e-4
#define MAX_NUM_ITER 5000
#define PACE         50

static const char NTRANSA = 'N',
                  TRANSA = 'T',
                  INCREASING = 'I';
static const double D_ONE = 1.0,
                    D_ZERO = 0.0;
static const char MAT_GXXF[6] = "GXXF";
static const int RESTART = 20;

/* Trace Minimization for generalized eigenvalue problem: AY = BY * diag(S)
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
							 const MKL_INT *A_i,
							 const MKL_INT *A_j,
							 const double *A_v,
							 const MKL_INT *B_i,
							 const MKL_INT *B_j,
							 const double *B_v,
               double *&Y,
               double *&S,
               double *&norms,
               double *&timing)
{
	/*---------------------------------------------------------------------------
	 * declaration of variables
	 *---------------------------------------------------------------------------*/
	double t_start,
				 t_end;
	double cg_start, cg_end, cg_total = 0.0;
	double j_start, j_end, j_total = 0.0;
	double qr_start, qr_end, qr_total = 0.0;
	MKL_INT c = p, 										      // dimension of the subspace
          s = 0,                          // start of the newly expanded subspace
	        info;                           // return values from LAPACK
  int d = RESTART * p,                    // max dimension of the subspace
      nc,                                 // number of converged columns
	    wA, wB;					    				      	// width of the matrix orders for matrix A and B
	int *ordersA = NULL,	    				      // order of annihilation for Z^T * A * Z
	    *ordersB = NULL;	    				      // order of annihilation for V^T * B * V
	
  /*---------------------------------------------------------------------------
	 * create the matries V, BV, BZ, BY, M, N, X, XP, U, W, Z, AZ, AY, Y, R, T
   * and the vectors MS, eigenvalues, norms and perm
	 *---------------------------------------------------------------------------*/
	MKL_INT *perm = new MKL_INT[d];		      // permuatation
  double *V     = new double[n * p](),  	// the matrix V
         *Q     = new double[n * (d - p)],// projection matrix for V
         *BV    = new double[n * d],      // B * V
         *BZ    = new double[n * d],      // B * Z
         *BY    = new double[n * p],      // B * Y; also Q1 after QR factorization
         *AZ    = new double[n * d],      // A * Z
         *AY    = new double[n * p],      // A * Y; also the RHS after projection
         *ZAZ   = new double[d * d],      // Z^T * A * Z
         *M     = new double[d * d],      // V^T * B * V; also eigenvectors of Z^T * A * Z
         *U     = new double[d * d],      // Q1^T * AY
         *W     = new double[n * d],      // Q1 * U
         *Z     = new double[n * d],      // V * M after eigen decomposition
         *R     = new double[n * p],      // AY - BY * S
         *MS    = new double[d],          // the magnitude of S
         *TS    = new double[d],          // Ritz values; approximate eigenvalues
         *tau   = new double[d];          // workspace for QR factorization

  Y = new double[n * p];                  // eigenvectors
  S = new double[p];                      // eigenvalues
	norms = new double[p]; 			  	        // column norms
  timing = new double[NUM_TIMINGS];       // timing

	/*---------------------------------------------------------------------------
	 * start the timer
	 *---------------------------------------------------------------------------*/
	t_start = omp_get_wtime();

	/*---------------------------------------------------------------------------
	 * start with V = [I; I; ...]
	 *---------------------------------------------------------------------------*/
	for (int i = 0; i < n; ++i) {
    V[(i % p) * n + i] = 1.0;
	}

	/*---------------------------------------------------------------------------
	 * generate the order of annihiliation
	 *---------------------------------------------------------------------------*/
	GenerateAnnihilationOrder(p, wB, ordersB);
  
  int k;
  for (k = 0; k < MAX_NUM_ITER; ++k) {
    /*---------------------------------------------------------------------------
     * if it is not initial iteration, project newly added columns out of BV(k-1)
     *---------------------------------------------------------------------------*/
    if (c != p) {
      /*---------------------------------------------------------------------------
       * Copy the first s = c-p columns of BZ (B-orthonormalized V) to Q
       *---------------------------------------------------------------------------*/
      memcpy(Q, BZ, n * s * sizeof(double));

      /*---------------------------------------------------------------------------
       * Perform QR factorization of first s columns of V and get Q1
       *---------------------------------------------------------------------------*/
      qr_start = omp_get_wtime();
      LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, s, Q, n, tau);
      LAPACKE_dorgqr(LAPACK_COL_MAJOR, n, s, s, Q, n, tau);
      qr_end = omp_get_wtime();
      qr_total += qr_end - qr_start;
      
      /*---------------------------------------------------------------------------
       * Apply I - Q1 * Q1^T to last p columns of V
       *---------------------------------------------------------------------------*/
      cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, s, p, n, 1.0, Q, n, V, n, 0.0, U, s);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, p, s, 1.0, Q, n, U, s, 0.0, W, n);
      cblas_daxpy(n * p, -1.0, W, 1, V, 1);
    }

		/*---------------------------------------------------------------------------
		 * M = V^T B V
		 *---------------------------------------------------------------------------*/
    mkl_dcsrmm(&NTRANSA, &n, &p, &n, &D_ONE, MAT_GXXF, B_v, B_j, B_i, B_i+1, V, &n, &D_ZERO, BV, &n);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, p, p, n, 1.0, V, n, BV, n, 0.0, M, p);
#if 0
    ViewDense("M", p, p, M);
#endif
		/*---------------------------------------------------------------------------
		 * Perform eigen decomposition
		 *---------------------------------------------------------------------------*/
    j_start = omp_get_wtime();
		Jacobi1(ordersB, p, wB, M, TS);
    j_end = omp_get_wtime();
    j_total += j_end - j_start;

		/*---------------------------------------------------------------------------
		 * MS = S^{-1/2}
		 * X = X * M
		 * BY = BV * X
		 * Z = V * X
		 * AZ = A * Z 
		 * M = Z^T * AZ
		 *---------------------------------------------------------------------------*/
    vdInvSqrt(p, TS, MS);
    for (int j = 0; j < p; ++j) {
      cblas_dscal(p, MS[j], M + j*p, 1);
    }
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, p, p, 1.0, V, n, M, p, 0.0, Z + s*n, n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, p, p, 1.0, BV, n, M, p, 0.0, BZ + s*n, n);
    mkl_dcsrmm(&NTRANSA, &n, &p, &n, &D_ONE, MAT_GXXF, A_v, A_j, A_i, A_i+1, Z + s*n, &n, &D_ZERO, AZ + s*n, &n);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, c, p, n, 1.0, Z, n, AZ + s*n, n, 0.0, ZAZ + s*d, d);
    for (int j = 0; j < p; ++j) {
      for (int i = 0; i < c; ++i) {
        ZAZ[i * d + s + j] = ZAZ[(s + j) * d + i];
      }
    }
    for (int j = 0; j < c; ++j) {
      memcpy(M + j*c, ZAZ + j*d, c * sizeof(double));
    }
#if 0
    ViewDense("M", c, c, M);
#endif

    /*---------------------------------------------------------------------------
		 * Perform eigen decomposition
		 *---------------------------------------------------------------------------*/
    if (c == p) {
      wA = wB;
      ordersA = new int[n * wB];
      memcpy(ordersA, ordersB, n * wB * sizeof(int));
    } else {
      /*---------------------------------------------------------------------------
       * generate the order of annihiliation
       *---------------------------------------------------------------------------*/
      GenerateAnnihilationOrder(c, wA, ordersA);
    }
    j_start = omp_get_wtime();
		Jacobi1(ordersA, c, wA, M, TS, false);
		//Jacobi2(ordersA, c, wA, M, TS);
    j_end = omp_get_wtime();
    j_total += j_end - j_start;

    delete [] ordersA;
	
#if 0
    ViewDense("S", 1, s, S);
#endif
		/*---------------------------------------------------------------------------
		 * Sort the eigenvalues and get the permuation
		 *---------------------------------------------------------------------------*/
		vdAbs(c, TS, MS);
		for (int j = 0; j < c; ++j) {
			perm[j] = j + 1;
		}
    dlasrt2(&INCREASING, &c, MS, perm, &info);
    
		/*---------------------------------------------------------------------------
		 * Permute X and S, and postmultiply Z, BZ and AZ by X
		 *---------------------------------------------------------------------------*/
    LAPACKE_dlapmr(LAPACK_ROW_MAJOR, 1, c, c, M, c, perm);
    LAPACKE_dlapmr(LAPACK_ROW_MAJOR, 1, c, 1, TS, 1, perm);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, p, c, 1.0, Z, n, M, c, 0.0, Y, n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, p, c, 1.0, BZ, n, M, c, 0.0, BY, n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, p, c, 1.0, AZ, n, M, c, 0.0, AY, n);
		
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
      cblas_dscal(n, TS[j], R + j*n, 1);
    }
    cblas_daxpy(n * p, -1.0, AY, 1, R, 1);

#if 0
    ViewDense("R", n, p, R);
    ViewDense("M", c, c, M);
    ViewDense("TS", 1, c, TS);
#endif

		/*---------------------------------------------------------------------------
		 * Test for convergence
		 *---------------------------------------------------------------------------*/
    nc = 0;
    for (int j = 0; j < p; ++j) {
      norms[j] = cblas_dnrm2(n, R + j*n, 1);
			if (norms[j] <= EIGEN_CONVERGENCE_TOL * MS[j]) ++nc;
		}
		if (k % PACE == 0) {
		  printf("Iter[%d] : Number of converged columns = %d\n", k, nc);
      for (int j = 0; j < p; ++j) {
        printf("norms[%d]=%.10lf, ev[%d]=%.10lf\n", j, norms[j], j, TS[j]);
      }
		}
		if (nc >= p) break;

		/*---------------------------------------------------------------------------
		 * Perform QR factorization of first p columns of BY and get Q1
		 *---------------------------------------------------------------------------*/
    qr_start = omp_get_wtime();
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, p, BY, n, tau);
    LAPACKE_dorgqr(LAPACK_COL_MAJOR, n, p, p, BY, n, tau);
    qr_end = omp_get_wtime();
    qr_total += qr_end - qr_start;

		/*---------------------------------------------------------------------------
		 * Compute the right-hand-side of the reduced system
		 *---------------------------------------------------------------------------*/
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, p, p, n, 1.0, BY, n, AY, n, 0.0, U, p);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, p, p, 1.0, BY, n, U, p, 0.0, W, n);
    cblas_daxpy(n * p, -1.0, W, 1, AY, 1);

#if 0
    ViewDense("Q1", n, p, BY);
    ViewDense("RHS", n, p, AY);
#endif

		/*---------------------------------------------------------------------------
		 * CG / MINRES
		 *---------------------------------------------------------------------------*/
    cg_start = omp_get_wtime();
    linear_solver(A_i, A_j, A_v, BY, AY, V, n, p, TS);
    cg_end = omp_get_wtime();
    cg_total += cg_end - cg_start;
#if 0    
    ViewDense("V", n, p, V);
#endif

		/*---------------------------------------------------------------------------
		 * if it exceeds the limit, only keep the first p columns of Z
		 *---------------------------------------------------------------------------*/
    if (c + p > d) {
      c = 2 * p;
      s = p;
    } else {
      c += p;
      s += p;
    }
#if 0
    if (k == 1) {
      break;
    }
#endif
  }

	/*---------------------------------------------------------------------------
	 * stop the timer
	 *---------------------------------------------------------------------------*/
	t_end = omp_get_wtime();
	printf("Total iter = %d\n", k);
	printf("Total time = %.6lf\n", t_end - t_start);
	printf("Jacobi time = %.6lf (average = %.6lf)\n", j_total, j_total / (2 * k));
	printf("QR time = %.6lf\n", qr_total);
	printf("Linear time = %.6lf\n", cg_total);

  timing[0] = t_end - t_start;
  timing[1] = j_total;
  timing[2] = j_total / (2 * k);
  timing[3] = qr_total;
  timing[4] = cg_total;

  /*---------------------------------------------------------------------------
   * copy back the eigenvalues
   *---------------------------------------------------------------------------*/
  memcpy(S, TS, p * sizeof(double));

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
  delete [] ordersB;
  delete [] tau;
  delete [] perm;
}

