#include "TraceMin.h"
#include "JacobiEigenDecomposition.h"
#include "tracemin_cg_q1.h"
#include <omp.h>

/* Trace Minimization for generalized eigenvalue problem: AY = BY * diag(S)
 * @input n size of the system
 * @input p number of desired eigenvalues
 * @input A symmetric matrix A
 * @input B s.p.d. matrix B
 * @output Y eigenvectors of the system
 * @output S eigenvalues of the system
 */
void TraceMin1(const PetscInt n,
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
	PetscInt s = 2 * p,								// dimension of the subspace
					 c = p,										// number of converged columns
					 w;												// width of the matrix orders
	PetscInt *orders,									// order of annihilation
					 *idx,										// array of indices
					 *perm;										// permuatation
	PetscReal *eigenvalues,						// list of eigenvalues
						*norms;									// column norms
	Mat V,														// the matrix V
			BV,
			BZ,
			BY,
			AZ,
			AY,
			M,
			N,
			X,
			Xperm,
			Z,
			R;
	Vec MS;														// the magnitude of S
	IS col;														// index set for column

	/*---------------------------------------------------------------------------
	 * create the matries V, BV, M, X, Z, BY, AZ, AY, Y, R
	 *---------------------------------------------------------------------------*/
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &V);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &BV);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &BZ);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &BY);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, s, s, NULL, &M);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, s, s, NULL, &N);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, s, s, NULL, &X);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, s, s, NULL, &Xperm);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &Z);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &AZ);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &AY);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &Y);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &R);
	
	/*---------------------------------------------------------------------------
	 * create the vectors S and MS
	 *---------------------------------------------------------------------------*/
	VecCreate(MPI_COMM_SELF, &S);
	VecSetSizes(S, PETSC_DECIDE, s);
	VecSetFromOptions(S);
	VecCreate(MPI_COMM_SELF, &MS);
	VecSetSizes(MS, PETSC_DECIDE, s);
	VecSetFromOptions(MS);

	/*---------------------------------------------------------------------------
	 * allocate memory for arrays for eigenvalues and permutation
	 *---------------------------------------------------------------------------*/
	PetscMalloc1(s * sizeof(PetscReal), &eigenvalues);
	PetscMalloc1(s * sizeof(PetscReal), &norms);
	PetscMalloc1(s * sizeof(PetscInt), &perm);

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
  for (PetscInt k = 0; k < 3; ++k) {
		/*---------------------------------------------------------------------------
		 * M = V^T B V
		 *---------------------------------------------------------------------------*/
		MatMatMult(B, V, MAT_REUSE_MATRIX, PETSC_DEFAULT, &BV);
		MatTransposeMatMult(V, BV, MAT_REUSE_MATRIX, PETSC_DEFAULT, &M);
	
		/*---------------------------------------------------------------------------
		 * Perform eigen decomposition
		 *---------------------------------------------------------------------------*/
		JacobiEigenDecomposition(orders, s, w, M, X, S);

		/*---------------------------------------------------------------------------
		 * M = S^{-1/2}
		 * N = X * M
		 * BY = BV * N
		 * Z = V * N
		 * AZ = A * Z 
		 * M = Z^T * AZ
		 *---------------------------------------------------------------------------*/
		VecSqrtAbs(S);
		VecReciprocal(S);
		MatZeroEntries(M);
		MatDiagonalSet(M, S, INSERT_VALUES);
		MatMatMult(X, M, MAT_REUSE_MATRIX, PETSC_DEFAULT, &N);
		MatMatMult(BV, N, MAT_REUSE_MATRIX, PETSC_DEFAULT, &BZ);
		MatMatMult(V, N, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Z);
		MatMatMult(A, Z, MAT_REUSE_MATRIX, PETSC_DEFAULT, &AZ);
		MatTransposeMatMult(Z, AZ, MAT_REUSE_MATRIX, PETSC_DEFAULT, &M);
		//MatView(Z, PETSC_VIEWER_STDOUT_SELF);
		//MatView(N, PETSC_VIEWER_STDOUT_SELF);
		//MatView(M, PETSC_VIEWER_STDOUT_SELF);
		
		/*---------------------------------------------------------------------------
		 * Perform eigen decomposition
		 *---------------------------------------------------------------------------*/
		JacobiEigenDecomposition(orders, s, w, M, X, S);
		
		/*---------------------------------------------------------------------------
		 * Sort the eigenvalues and get the permuation
		 *---------------------------------------------------------------------------*/
		VecCopy(S, MS);
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
		MatZeroEntries(Xperm);
		for (PetscInt j = 0; j < s; ++j) {
			MatSetValue(Xperm, perm[j], j, 1.0, INSERT_VALUES);
		}	
		MatAssemblyBegin(Xperm, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(Xperm, MAT_FINAL_ASSEMBLY);
		MatMatMult(X, Xperm, MAT_REUSE_MATRIX, PETSC_DEFAULT, &N);

		MatMatMult(BZ, N, MAT_REUSE_MATRIX, PETSC_DEFAULT, &BY);
		MatMatMult(AZ, N, MAT_REUSE_MATRIX, PETSC_DEFAULT, &AY);
		MatMatMult(Z, N, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y);
		
		VecPermute(S, col, PETSC_FALSE);
		ISDestroy(&col);
		
//		PetscPrintf(PETSC_COMM_SELF, "Y:\n");
//    MatView(Y, PETSC_VIEWER_STDOUT_SELF);

		/*---------------------------------------------------------------------------
		 * M = diag(S)
		 *---------------------------------------------------------------------------*/
		MatZeroEntries(M);
		MatDiagonalSet(M, S, INSERT_VALUES);
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
		c = p;
		for (PetscInt r = 0; r < p; ++r) {
			if (norms[r] <= 1e-3 * eigenvalues[r]) --c;
		}
		PetscPrintf(PETSC_COMM_SELF, "Number of converged columns = %d\n", c);
		if (c == 0) break;
		
		/*---------------------------------------------------------------------------
		 * CG / MINRES
		 *---------------------------------------------------------------------------*/
    tracemin_cg(A, V, BY, AY, n, s); 
    MatAYPX(V, -1.0, Y, SAME_NONZERO_PATTERN);
//		PetscPrintf(PETSC_COMM_SELF, "V:\n");
//    MatView(V, PETSC_VIEWER_STDOUT_SELF);
  }

	/*---------------------------------------------------------------------------
	 * stop the timer
	 *---------------------------------------------------------------------------*/
	t_end = omp_get_wtime();
	PetscPrintf(PETSC_COMM_SELF, "Total time = %lf\n", t_end - t_start);

#if 0
	PetscPrintf(PETSC_COMM_SELF, "AY:\n");
	MatView(AY, PETSC_VIEWER_STDOUT_SELF);
	PetscPrintf(PETSC_COMM_SELF, "BY:\n");
	MatView(BY, PETSC_VIEWER_STDOUT_SELF);
#endif

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
	MatDestroy(&Xperm);
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

