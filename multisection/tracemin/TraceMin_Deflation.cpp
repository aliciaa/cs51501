#include "TraceMin.h"
#include "JacobiEigenDecomposition.h"
#include "QRFactorization.h"
#include "tracemin_cg_q1.h"
#include <omp.h>

#define EIGEN_CONVERGENCE_TOL 1.e-5
#define MAX_NUM_ITER 1000

/* Trace Minimization for generalized eigenvalue problem: AY = BY * diag(S)
 * @input n size of the system
 * @input p number of desired eigenvalues
 * @input A symmetric matrix A
 * @input B s.p.d. matrix B
 * @output Y eigenvectors of the system
 * @output S eigenvalues of the system
 */
void TraceMin1(int task_id,
              const char* fileO,
              const PetscInt n,
							 const PetscInt p,
							 const Mat A,
							 const Mat B,
							 Mat & Y,
							 Vec & S)
{
	/*---------------------------------------------------------------------------
	 * declaration of variables
	 *---------------------------------------------------------------------------*/
	double t_start,
				 t_end;
	double cg_start, cg_end, cg_total;
	PetscInt s = 2 * p,							    	// dimension of the subspace
					 c = 0,								    		// number of converged columns
           uc,                          // number of unconverged column
					 w;											    	// width of the matrix orders
	PetscInt *orders,							    		// order of annihilation
					 *idx,								    		// array of indices
					 *perm;								    		// permuatation
	PetscReal *eigenvalues,						    // list of eigenvalues
						*norms;									    // column norms
  PetscScalar ev,                       // eigenvalue
              *dataYC,                  // pointer to array of YC
              *dataBYC,                 // pointer to array of BYC
              *dataBC;                  // pointer to array of BC
  PetscBool converged = PETSC_FALSE;    // more converged columns in previous iteration?
	Mat C = NULL,
      V,														    // the matrix V
      BC = NULL,
      PBC = NULL,
      BCU = NULL,
      BCW = NULL,
			BV,
			BZ,
			BY,
			AZ,
			AY,
			M,
			N,
			X,
			XP,
      U,
      T,
      W,
			Z,
      YC,
      BYC,
      PBYC = NULL,
			R;
	Vec TS,
      MS;											       			// the magnitude of S
	IS col;											      			// index set for column

	/*---------------------------------------------------------------------------
	 * create the matries V, BV, BZ, BY, M, N, X, Z, AZ, AY, YC, R, Y
	 *---------------------------------------------------------------------------*/
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &V);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &BV);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &BZ);
  MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &BCW);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, s, s, NULL, &M);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, s, s, NULL, &N);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, s, s, NULL, &X);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, s, s, NULL, &XP);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, s, s, NULL, &U);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &W);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &Z);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &AZ);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &AY);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &R);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &T);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s+p, NULL, &YC);
  MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s+p, NULL, &BYC);

	/*---------------------------------------------------------------------------
	 * the matrices Y is submatrices of the matrix YC and
   * the matrices BYC and PBYC are the submatrices of the matrix BYC
	 *---------------------------------------------------------------------------*/
  MatDenseGetArray(YC, &dataYC);
  MatDenseGetArray(BYC, &dataBYC);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, dataYC, &Y);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, dataBYC, &BY);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, dataBYC, &PBYC);
	
	/*---------------------------------------------------------------------------
	 * create the vectors S, TS and MS
	 *---------------------------------------------------------------------------*/
	VecCreate(MPI_COMM_SELF, &S);
	VecSetSizes(S, PETSC_DECIDE, p);
	VecSetFromOptions(S);
	VecCreate(MPI_COMM_SELF, &TS);
	VecSetSizes(TS, PETSC_DECIDE, s);
	VecSetFromOptions(TS);
	VecCreate(MPI_COMM_SELF, &MS);
	VecSetSizes(MS, PETSC_DECIDE, s);
	VecSetFromOptions(MS);

	/*---------------------------------------------------------------------------
	 * allocate memory for arrays for eigenvalues and permutation
	 *---------------------------------------------------------------------------*/
	PetscMalloc1(s, &eigenvalues);
	PetscMalloc1(s, &norms);
	PetscMalloc1(s, &perm);

	/*---------------------------------------------------------------------------
	 * start the timer
	 *---------------------------------------------------------------------------*/
	t_start = omp_get_wtime();

	/*---------------------------------------------------------------------------
	 * start with V = [I; I; ...]
	 *---------------------------------------------------------------------------*/
	for (PetscInt i = 0; i < n; ++i) {
		MatSetValue(V, i, i%s, 1.0, INSERT_VALUES);
	}
	MatAssemblyBegin(V, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(V, MAT_FINAL_ASSEMBLY);

	//MatView(V, PETSC_VIEWER_STDOUT_SELF);

	/*---------------------------------------------------------------------------
	 * generate the order of annihiliation
	 *---------------------------------------------------------------------------*/
	GenerateAnnihilationOrder(s, w, orders);

	//while (true) {
  PetscInt k;
  for (k = 0; k < MAX_NUM_ITER; ++k) {
		/*---------------------------------------------------------------------------
		 * if some columns have been converged, need to project the subspace orthogonal to BC
		 *---------------------------------------------------------------------------*/
    if (c != 0) {
      MatTransposeMatMult(PBC, V, MAT_REUSE_MATRIX, PETSC_DEFAULT, &BCU);
      MatMatMult(PBC, BCU, MAT_REUSE_MATRIX, PETSC_DEFAULT, &BCW);
      MatAXPY(V, -1.0, BCW, SAME_NONZERO_PATTERN);
    }
		/*---------------------------------------------------------------------------
		 * M = V^T B V
		 *---------------------------------------------------------------------------*/
		MatMatMult(B, V, MAT_REUSE_MATRIX, PETSC_DEFAULT, &BV);
		MatTransposeMatMult(V, BV, MAT_REUSE_MATRIX, PETSC_DEFAULT, &M);
	
		/*---------------------------------------------------------------------------
		 * Perform eigen decomposition
		 *---------------------------------------------------------------------------*/
		JacobiEigenDecomposition(orders, s, w, M, X, TS);

		/*---------------------------------------------------------------------------
		 * M = S^{-1/2}
		 * N = X * M
		 * BY = BV * N
		 * Z = V * N
		 * AZ = A * Z 
		 * M = Z^T * AZ
		 *---------------------------------------------------------------------------*/
		VecSqrtAbs(TS);
		VecReciprocal(TS);
		MatZeroEntries(M);
		MatDiagonalSet(M, TS, INSERT_VALUES);
		MatMatMult(X, M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &N);
		MatMatMult(BV, N, MAT_REUSE_MATRIX, PETSC_DEFAULT, &BZ);
		MatMatMult(V, N, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Z);
		MatMatMult(A, Z, MAT_REUSE_MATRIX, PETSC_DEFAULT, &AZ);
		MatTransposeMatMult(Z, AZ, MAT_REUSE_MATRIX, PETSC_DEFAULT, &M);
		PetscScalar trace;
		MatGetTrace(M, &trace);
		//PetscPrintf(PETSC_COMM_SELF, "trace(Z^TAZ)=%.10lf\n", trace);

		//MatView(Z, PETSC_VIEWER_STDOUT_SELF);
		//MatView(N, PETSC_VIEWER_STDOUT_SELF);
		//MatView(M, PETSC_VIEWER_STDOUT_SELF);
		
		/*---------------------------------------------------------------------------
		 * Perform eigen decomposition
		 *---------------------------------------------------------------------------*/
		JacobiEigenDecomposition(orders, s, w, M, X, TS);
		
		/*---------------------------------------------------------------------------
		 * Sort the eigenvalues and get the permuation
		 *---------------------------------------------------------------------------*/
		VecCopy(TS, MS);
		VecAbs(MS);
		for (PetscInt r = 0; r < s; ++r) {
			perm[r] = r;
		}
		VecGetValues(MS, s, perm, eigenvalues);
		PetscSortRealWithPermutation(s, eigenvalues, perm);
		PetscSortReal(s, eigenvalues);
		ISCreateGeneral(PETSC_COMM_SELF, s, perm, PETSC_COPY_VALUES, &col);
		/*---------------------------------------------------------------------------
		 * Permute X and postmultiply Z, BZ and AZ by X
		 *---------------------------------------------------------------------------*/
		MatZeroEntries(XP);
		for (PetscInt j = 0; j < s; ++j) {
			MatSetValue(XP, perm[j], j, 1.0, INSERT_VALUES);
		}	
		MatAssemblyBegin(XP, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(XP, MAT_FINAL_ASSEMBLY);
		MatMatMult(X, XP, MAT_REUSE_MATRIX, PETSC_DEFAULT, &N);

		MatMatMult(BZ, N, MAT_REUSE_MATRIX, PETSC_DEFAULT, &BY);
		MatMatMult(AZ, N, MAT_REUSE_MATRIX, PETSC_DEFAULT, &AY);
		MatMatMult(Z, N, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y);
		
		VecPermute(TS, col, PETSC_FALSE);
		ISDestroy(&col);
		
//		PetscPrintf(PETSC_COMM_SELF, "Y:\n");
//    MatView(Y, PETSC_VIEWER_STDOUT_SELF);

		/*---------------------------------------------------------------------------
		 * M = diag(S)
		 *---------------------------------------------------------------------------*/
		MatZeroEntries(M);
		MatDiagonalSet(M, TS, INSERT_VALUES);
		MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);

		/*---------------------------------------------------------------------------
		 * R = AY - BY * M
		 *---------------------------------------------------------------------------*/
		MatMatMult(BY, M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &R);
		MatAYPX(R, -1.0, AY, SAME_NONZERO_PATTERN);

		/*---------------------------------------------------------------------------
		 * Test for convergence
		 *---------------------------------------------------------------------------*/
		MatGetColumnNorms(R, NORM_2, norms);
		//VecView(S, PETSC_VIEWER_STDOUT_SELF);
    uc = p - c;
		for (PetscInt r = 0; r < uc; ++r) {
      /*---------------------------------------------------------------------------
       * If any column is converged, move it to C and reset the column and
       * move eigenvalues from TS to S
       *---------------------------------------------------------------------------*/
			if (norms[r] <= EIGEN_CONVERGENCE_TOL * eigenvalues[r] ||
			    norms[r] <= EIGEN_CONVERGENCE_TOL) {
        converged = PETSC_TRUE;
        PetscMemmove(dataYC + (s + c) * n, dataYC + r * n, n * sizeof(*dataYC));
        PetscMemzero(dataYC + r * n, n * sizeof(*dataYC));
        for (PetscInt q = r; q < n; q += s) {
          dataYC[r * n + q] = 1.0;
        }
        VecGetValues(TS, 1, &r, &ev);
        VecSetValue(S, c, ev, INSERT_VALUES);
        ++c;
      }
			//PetscPrintf(PETSC_COMM_SELF, "norm[%d]=%.10lf\n", r, norms[r]);
		}
		if (k % 20 == 0) {
		  PetscPrintf(PETSC_COMM_SELF, "Iter[%d] : Number of converged columns = %d\n", k, c);
		}
		if (c == p) break;
		
		/*---------------------------------------------------------------------------
		 * if more columns have been converged, need new projection matrix for BC
		 *---------------------------------------------------------------------------*/
    if (converged == PETSC_TRUE) {
      converged = PETSC_FALSE;
      /*---------------------------------------------------------------------------
       * create new matrices C and BC since size has changed;
       * destroy the old ones if they were created
       *---------------------------------------------------------------------------*/
      if (C != NULL) {
        MatDestroy(&C);
        MatDestroy(&BC);
        MatDestroy(&BCU);
        MatDestroy(&U);
        MatDestroy(&PBYC);
      }
      MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, c, dataYC + n * s, &C);
      MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s + c, dataBYC, &PBYC);
      MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, c, s, NULL, &BCU);
      MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, s+c, s, NULL, &U);
      /*---------------------------------------------------------------------------
       * BC = B * C
       *---------------------------------------------------------------------------*/
      MatMatMult(B, C, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &BC);
      /*---------------------------------------------------------------------------
       * Perform QR factorization of BC and get PBC
       *---------------------------------------------------------------------------*/
      MatDuplicate(BC, MAT_COPY_VALUES, &PBC);
      QRFactorizationQ1(PBC);
    }

    /*---------------------------------------------------------------------------
     * If some columns have been converged, copy BC to BYC
     *---------------------------------------------------------------------------*/
    if (c != 0) {
      MatDenseGetArray(BC, &dataBC);
      PetscMemcpy(dataBYC + s * n, dataBC, c * n * sizeof(*dataBC));
    }

		/*---------------------------------------------------------------------------
		 * Perform QR factorization of BY and get Q1
		 *---------------------------------------------------------------------------*/
    QRFactorizationQ1(PBYC);

    //PetscPrintf(PETSC_COMM_SELF, "Q1\n");
    //MatView(BY, PETSC_VIEWER_STDOUT_SELF);

		/*---------------------------------------------------------------------------
		 * Compute the right-hand-side of the reduced system
		 *---------------------------------------------------------------------------*/
    MatTransposeMatMult(PBYC, AY, MAT_REUSE_MATRIX, PETSC_DEFAULT, &U);
    MatMatMult(PBYC, U, MAT_REUSE_MATRIX, PETSC_DEFAULT, &W);
    MatAXPY(AY, -1.0, W, SAME_NONZERO_PATTERN);

    //MatView(AY, PETSC_VIEWER_STDOUT_SELF);

		/*---------------------------------------------------------------------------
		 * CG / MINRES
		 *---------------------------------------------------------------------------*/
    cg_start = omp_get_wtime();
    tracemin_cg(A, PBYC, AY, V, n, s);
    cg_end = omp_get_wtime();
    cg_total += cg_end - cg_start;
/*
    MatMatMult(A, V, MAT_REUSE_MATRIX, PETSC_DEFAULT, &T);
    MatTransposeMatMult(BY, T, MAT_REUSE_MATRIX, PETSC_DEFAULT, &U);
    MatMatMult(BY, U, MAT_REUSE_MATRIX, PETSC_DEFAULT, &W);
    MatAXPY(T, -1.0, W, SAME_NONZERO_PATTERN);
    MatAXPY(T, -1.0, AY, SAME_NONZERO_PATTERN);
    MatGetColumnNorms(T, NORM_2, norms);
    for (PetscInt r =0 ; r < p; ++r) {
      PetscPrintf(PETSC_COMM_SELF, "cgnorm[%d]=%.10lf\n", r, norms[r]);
    }
*/
    MatAYPX(V, -1.0, Y, SAME_NONZERO_PATTERN);
//		PetscPrintf(PETSC_COMM_SELF, "V:\n");
//    MatView(V, PETSC_VIEWER_STDOUT_SELF);
  }

	/*---------------------------------------------------------------------------
	 * stop the timer
	 *---------------------------------------------------------------------------*/
	t_end = omp_get_wtime();
	PetscViewer viewer;
        PetscViewerASCIIOpen(PETSC_COMM_WORLD, fileO, &viewer);
	PetscPrintf(PETSC_COMM_SELF, "Total iter = %d\n", k);
	PetscPrintf(PETSC_COMM_SELF, "Total time = %.6lf\n", t_end - t_start);
	PetscPrintf(PETSC_COMM_SELF, "Linear time = %.6lf\n", cg_total);
        MatView(Y, viewer);
	VecView(S, viewer);
	PetscViewerDestroy(&viewer);
#if 0
	PetscPrintf(PETSC_COMM_SELF, "AY:\n");
	MatView(AY, PETSC_VIEWER_STDOUT_SELF);
	PetscPrintf(PETSC_COMM_SELF, "BY:\n");
	MatView(BY, PETSC_VIEWER_STDOUT_SELF);
#endif

	/*---------------------------------------------------------------------------
	 * copy back the vector
	 *---------------------------------------------------------------------------*/
  MatDestroy(&Y);
  MatDuplicate(C, MAT_COPY_VALUES, &Y);

	/*---------------------------------------------------------------------------
	 * deallocate the matrices, vectors and index sets
	 *---------------------------------------------------------------------------*/
	MatDestroy(&V);
	MatDestroy(&BV);
	MatDestroy(&BZ);
	MatDestroy(&BY);
	MatDestroy(&AZ);
	MatDestroy(&AY);
	MatDestroy(&M);
	MatDestroy(&N);
	MatDestroy(&X);
	MatDestroy(&XP);
	MatDestroy(&U);
	MatDestroy(&W);
	MatDestroy(&Z);
	MatDestroy(&R);
	VecDestroy(&MS);
	VecDestroy(&S);
	
	/*---------------------------------------------------------------------------
	 * deallocate the arrays
	 *---------------------------------------------------------------------------*/
	PetscFree(orders);
	PetscFree(eigenvalues);
	PetscFree(norms);
	PetscFree(perm);
}

