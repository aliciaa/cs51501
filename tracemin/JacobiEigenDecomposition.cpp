#include "JacobiEigenDecomposition.h"
#include <omp.h>

/* Generate the order of annihilation
 * @input n dimension of the matrix
 * @output w width of the matrix orders
 * @output orders a matrix representing the order of annihilation with rows being the order and columns
 * 							 	storing the orders of the off-diagonals that can be annihilated simultaneously
 */
void GenerateAnnihilationOrder(const PetscInt n,
															 PetscInt & w,
															 PetscInt *& orders)
{
	const PetscInt m = n / 2;				// m = floor(n / 2)

	/*---------------------------------------------------------------------------
	 * allocate the memory
	 *---------------------------------------------------------------------------*/
	w = m * 2;											// width of the matrix orders
	PetscMalloc1(n * w * sizeof(PetscInt), &orders);

	/*---------------------------------------------------------------------------
	 * assign the values
	 *---------------------------------------------------------------------------*/
	for (PetscInt k = 0; k < n; ++k) {
		const PetscInt s = n - k,
									 b = s / 2 + 1;
		for (PetscInt p = 0; p < m; ++p) {
			const PetscInt j = b + p,
										 i = s - j;
			if (i + n == j) {
				orders[k * w + 2*p] = -1;
				orders[k * w + 2*p+1] = -1;
			} else if (i < 0) {
				orders[k * w + 2*p] = j;
				orders[k * w + 2*p+1] = i+n;
			} else {
				orders[k * w + 2*p] = i;
				orders[k * w + 2*p+1] = j;
			}
		}
	}
}

/* Compute the eigen decomposition using 2-sided Jacobi method, A = VDV^T
 * @input orders a matrix representing the order of annihilation with rows being the order and columns
 * 						 	 storing the orders of the off-diagonals that can be annihilated simultaneously
 * @input n dimension of the matrix
 * @input w width of the matrix orders
 * @input A the matrix to be decomposed
 * @output A the matrix V^T * A * V
 * @output V the matrix contains the eigenvectors of A
 * @output S the vector contains the eigenvalues of A
 */
void JacobiEigenDecomposition(const PetscInt * orders,
															const PetscInt n,
															const PetscInt w,
															Mat & A,
															Mat & V,
															Vec & S)
{

	/*---------------------------------------------------------------------------
	 * declare an orthogonal matrix U for intermediate step and a vector D
	 * create the matries U, V and the vector D
	 *---------------------------------------------------------------------------*/
	Mat U, W, B;
	Vec D;
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, n, NULL, &U);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, n, NULL, &W);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, n, NULL, &B);
	VecCreate(MPI_COMM_SELF, &D);
	VecSetSizes(D, PETSC_DECIDE, n);
	VecSetFromOptions(D);

	/*---------------------------------------------------------------------------
	 * set the vector Ai to be all ones and the matrix V to be the identity
	 *---------------------------------------------------------------------------*/
	VecSet(D, 1.0);
	MatDiagonalSet(V, D, INSERT_VALUES);
	MatAssemblyBegin(V, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(V, MAT_FINAL_ASSEMBLY);
	
	/*---------------------------------------------------------------------------
	 * declare variables
	 *---------------------------------------------------------------------------*/
	PetscInt nnz;												// number of nonzero off-diagonals
	const PetscReal tol = 1e-12;				// tolerance
	PetscReal nrm;											// norm of A
	const PetscInt m = n / 2;						// max number of annihiliations
	PetscScalar aii, ajj, aij,					// principal submatrix values of A
							a, t, c, s;							// intermediate values
	PetscInt i, j,											// indices
					 p,													// counting variables
					 niter = 0;
	PetscBool even = PETSC_TRUE;				// even number of rotations

	do {
		nnz = n * (n - 1) / 2;						// set the total number of nonzeros
		MatNorm(A, NORM_FROBENIUS, &nrm);

		for (PetscInt k = 0; k < n; ++k) {
			const PetscInt* currOrders = orders + k * w;
			/*---------------------------------------------------------------------------
			 * initialize the matrix U to be the identity
			 *---------------------------------------------------------------------------*/
			MatZeroEntries(U);
			for (PetscInt r = 0; r < n; ++r) {
				MatSetValue(U, r, r, 1.0, INSERT_VALUES);
			}

#pragma omp parallel for private(p, i, j, aii, ajj, aij, a, t, c, s)
			for (p = 0; p < m; ++p) {
				i = currOrders[2*p];
				j = currOrders[2*p+1];
				if (i != -1) {
					/*---------------------------------------------------------------------------
					 * get the values from A
					 *---------------------------------------------------------------------------*/
					MatGetValues(A, 1, &i, 1, &i, &aii);
					MatGetValues(A, 1, &j, 1, &j, &ajj);
					MatGetValues(A, 1, &i, 1, &j, &aij);

					if (fabs(aij) < tol * nrm) {
						/*---------------------------------------------------------------------------
						 * decrement the off-diagonal nnz
						 *---------------------------------------------------------------------------*/
						--nnz;
					} else {
						/*---------------------------------------------------------------------------
						 * compute a, t, c, s
						 *---------------------------------------------------------------------------*/
						a = (aii - ajj) / (2 * aij);
						t = (a >= 0 ? 1.0 : -1.0) / (fabs(a) + sqrt(a * a + 1));
						c = 1.0 / sqrt(t * t + 1);
						s = c * t;

						/*---------------------------------------------------------------------------
						 * set the values of matrix U
						 *---------------------------------------------------------------------------*/
						MatSetValue(U, i, i, c, INSERT_VALUES);
						MatSetValue(U, i, j, -1.0 * s, INSERT_VALUES);
						MatSetValue(U, j, i, s, INSERT_VALUES);
						MatSetValue(U, j, j, c, INSERT_VALUES);
					}
				}
			}

			/*---------------------------------------------------------------------------
			 * assemble the matrix U
			 *---------------------------------------------------------------------------*/
			MatAssemblyBegin(U, MAT_FINAL_ASSEMBLY);
			MatAssemblyEnd(U, MAT_FINAL_ASSEMBLY);

			//MatView(U, PETSC_VIEWER_STDOUT_SELF);

			/*---------------------------------------------------------------------------
			 * apply U: A = A * U
			 *---------------------------------------------------------------------------*/
			MatMatMult(A, U, MAT_REUSE_MATRIX, PETSC_DEFAULT, &B);
			MatTransposeMatMult(U, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &A);
			if (even == PETSC_TRUE) {
				MatMatMult(V, U, MAT_REUSE_MATRIX, PETSC_DEFAULT, &W);
				even = PETSC_FALSE;
			} else {
				MatMatMult(W, U, MAT_REUSE_MATRIX, PETSC_DEFAULT, &V);
				even = PETSC_TRUE;
			}
			//MatCopy(W, V, SAME_NONZERO_PATTERN);

			//MatView(A, PETSC_VIEWER_STDOUT_SELF);
		}
		++niter;
	} while (nnz > 0);

	//MatView(A, PETSC_VIEWER_STDOUT_SELF);

	/*---------------------------------------------------------------------------
	 * only keep the diagonal of A
	 *---------------------------------------------------------------------------*/
	MatGetDiagonal(A, S);
	
	/*---------------------------------------------------------------------------
	 * copy back from W to V if necessary
	 *---------------------------------------------------------------------------*/
	if (even == PETSC_FALSE) {
		MatCopy(W, V, SAME_NONZERO_PATTERN);
	};

	/*---------------------------------------------------------------------------
	 * deallocate U, B and D
	 *---------------------------------------------------------------------------*/
	MatDestroy(&U);
	MatDestroy(&W);
	MatDestroy(&B);
	VecDestroy(&D);
}
