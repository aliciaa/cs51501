#include <petscmat.h>
#include <iostream>
#include <omp.h>
#include "matio.h"

using std::cerr;
using std::endl;

static const char help[] = "2-sided Jacobi eigen decomposition A = VDV^T\n";
static const int MASTER = 0;

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
 * @output A the vector contains the eigenvalues of A
 * @output V the matrix contains the eigenvector of A
 */
void JacobiEigenDecomposition(const PetscInt * orders,
															const PetscInt n,
															const PetscInt w,
															Mat& A,
															Mat& V)
{

	/*---------------------------------------------------------------------------
	 * declare an orthogonal matrix U for intermediate step and a vector D
	 * create the matries U, V and the vector D
	 *---------------------------------------------------------------------------*/
	Mat U;
	Mat W;
	Mat B;
	Vec D;
	MatCreate(MPI_COMM_SELF, &U);
	MatSetSizes(U, PETSC_DECIDE, PETSC_DECIDE, n, n);
	MatSetType(U, MATDENSE);
	MatSetUp(U);
	MatCreate(MPI_COMM_SELF, &W);
	MatSetSizes(W, PETSC_DECIDE, PETSC_DECIDE, n, n);
	MatSetType(W, MATDENSE);
	MatSetUp(W);
	MatCreate(MPI_COMM_SELF, &B);
	MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, n, n);
	MatSetType(B, MATDENSE);
	MatSetUp(B);
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
	cerr << "iterations = " << niter << endl;

	/*---------------------------------------------------------------------------
	 * only keep the diagonal of A
	 *---------------------------------------------------------------------------*/
	MatGetDiagonal(A, D);
	MatZeroEntries(A);
	MatDiagonalSet(A, D, INSERT_VALUES);
	
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

int main(int argc, char *argv[])
{
	unsigned char read_error = 0;
	int num_tasks,										// number of tasks in partition
			task_id;											// task identifier
	double t_start,
				 t_end;
	char fileA[PETSC_MAX_PATH_LEN];		// file of matrix A
	PetscScalar *arrayA;							// array of A
	Mat A,														// the matrix A
			V;														// the matrix V
	PetscInt *orders,									// order of annihilation
					 *idx;										// array of indices
	PetscInt n,												// dimension of matrix
					 w;												// width of the matrix orders

	/*---------------------------------------------------------------------------
	 * initialize PETSc
	 *---------------------------------------------------------------------------*/
	PetscInitialize(&argc, &argv, (char*) 0, help);

	MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

	if (task_id == MASTER){ // initilization: A and b
		PetscInt dummy;

		/*---------------------------------------------------------------------------
		 * read in matrix
		 *---------------------------------------------------------------------------*/
		PetscOptionsGetString(NULL, "-fin", fileA, PETSC_MAX_PATH_LEN, NULL);

		if (readSymmMatrix(fileA, n, arrayA)) {
			// read_error in reading in the vector, set read_error to 1
			read_error = 1;
		}

		/*---------------------------------------------------------------------------
		 * set up the matrix A
		 *---------------------------------------------------------------------------*/
		MatCreate(PETSC_COMM_SELF, &A);
		MatSetType(A, MATDENSE);
		MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);
		MatSetFromOptions(A);
		MatSetUp(A);

		/*---------------------------------------------------------------------------
		 * set the values of matrix A
		 *---------------------------------------------------------------------------*/
		/*
		PetscMalloc1(n * sizeof(PetscInt), &idx);
		for (PetscInt i = 0; i < n; ++i) {
			idx[i] = i;
		}
		MatSetValues(A, n, idx, n, idx, arrayA, INSERT_VALUES);
		*/
		for (PetscInt i = 0; i < n; ++i) {
			for (PetscInt j = 0; j < n; ++j) {
				MatSetValue(A, i, j, arrayA[i * n + j], INSERT_VALUES);
			}
		}

		/*---------------------------------------------------------------------------
		 * assemble the matrix A
		 *---------------------------------------------------------------------------*/
		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	}

	/*---------------------------------------------------------------------------
	 * start the timer
	 *---------------------------------------------------------------------------*/
	t_start = omp_get_wtime();

	/*---------------------------------------------------------------------------
	 * generate the order of annihiliation
	 *---------------------------------------------------------------------------*/
	GenerateAnnihilationOrder(n, w, orders);

#if 0
	/*---------------------------------------------------------------------------
	 * print out the order
	 *---------------------------------------------------------------------------*/
	for (PetscInt i = 0; i < n; ++i) {
		for (PetscInt j = 0; j < w; ++j) {
			PetscPrintf(PETSC_COMM_SELF, "%2i ", orders[i * w + j]);
		}
		PetscPrintf(PETSC_COMM_SELF, "\n");
	}

	MatView(A, PETSC_VIEWER_STDOUT_SELF);
#endif

	/*---------------------------------------------------------------------------
	 * create the matrix V
	 *---------------------------------------------------------------------------*/
	MatCreate(PETSC_COMM_SELF, &V);
	MatSetSizes(V, PETSC_DECIDE, PETSC_DECIDE, n, n);
	MatSetType(V, MATDENSE);
	MatSetUp(V);

	/*---------------------------------------------------------------------------
	 * Perform eigen decomposition
	 *---------------------------------------------------------------------------*/
	JacobiEigenDecomposition(orders, n, w, A, V);

	/*---------------------------------------------------------------------------
	 * stop the timer
	 *---------------------------------------------------------------------------*/
	t_end = omp_get_wtime();
	cerr << "Total time = " << t_end - t_start << " seconds" << endl;

	MatView(A, PETSC_VIEWER_STDOUT_SELF);
	MatView(V, PETSC_VIEWER_STDOUT_SELF);

	/*---------------------------------------------------------------------------
	 * deallocate the matrix
	 *---------------------------------------------------------------------------*/
	PetscFree(orders);
	
	/*---------------------------------------------------------------------------
	 * finalize PETSc
	 *---------------------------------------------------------------------------*/
	PetscFinalize();

	return 0;
}
