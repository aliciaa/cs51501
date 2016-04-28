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
	Mat U, B;
	Vec D;
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, n, NULL, &U);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, n, NULL, &B);
	VecCreate(MPI_COMM_SELF, &D);
	VecSetSizes(D, PETSC_DECIDE, n);
	VecSetFromOptions(D);

	/*---------------------------------------------------------------------------
	 * set the vector Ai to be all ones and the matrix V to be the identity
	 *---------------------------------------------------------------------------*/
	VecSet(D, 1.0);
	MatZeroEntries(V);
	MatDiagonalSet(V, D, INSERT_VALUES);
	MatAssemblyBegin(V, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(V, MAT_FINAL_ASSEMBLY);
	
	/*---------------------------------------------------------------------------
	 * declare variables
	 *---------------------------------------------------------------------------*/
	PetscInt nnz;												// number of nonzero off-diagonals
	const PetscReal tol = 1e-12;				// tolerance
	const PetscInt m = n / 2;						// max number of annihiliations
	PetscScalar aii, ajj, aij,					// principal submatrix values of A
							a, b, t, c, s;					// intermediate values
  PetscScalar *dataA,                 // array to data of matrix A
              *dataB,                 // array to data of matrix B
              *dataU;                 // array to data of matrix U
  PetscReal *norms;                   // norms
	PetscInt i, j,											// indices
					 p,	q,											// counting variables
					 niter = 0;
	PetscBool even = PETSC_TRUE;				// even number of rotations

  PetscMalloc1(n, &norms);

  MatDenseGetArray(A, &dataA);
  MatDenseGetArray(B, &dataB);
  MatDenseGetArray(U, &dataU);

	do {
		nnz = n * (n - 1) / 2;						// set the total number of nonzeros

		for (PetscInt k = 0; k < n; ++k) {
			const PetscInt* currOrders = orders + k * w;
			/*---------------------------------------------------------------------------
			 * initialize the matrix U to be the identity
			 *---------------------------------------------------------------------------*/
			MatZeroEntries(U);
      
			for (PetscInt r = 0; r < n; ++r) {
        dataU[r * (n + 1)] = 1.0;
			}

//#pragma omp parallel for private(p, i, j, aii, ajj, aij, a, b, t, c, s) reduction(-:nnz)
			for (p = 0; p < m; ++p) {
        int thread_id = omp_get_thread_num();
				i = currOrders[2*p];
				j = currOrders[2*p+1];
				if (i != -1) {
					/*---------------------------------------------------------------------------
					 * compute the norms and dot product
					 *---------------------------------------------------------------------------*/
          aii = 0.0;
          ajj = 0.0;
          aij = 0.0;
          if (even == PETSC_TRUE) {
            for (q = 0; q < n; ++q) {
              a = dataA[i * n + q];
              b = dataA[j * n + q];
              aii += a * a;
              ajj += b * b;
              aij += a * b;
            }
          } else {
            for (q = 0; q < n; ++q) {
              a = dataB[i * n + q];
              b = dataB[j * n + q];
              aii += a * a;
              ajj += b * b;
              aij += a * b;
            }
          }
          aii = sqrt(aii);
          ajj = sqrt(ajj);

					if (fabs(aij) < tol * aii * ajj) {
						/*---------------------------------------------------------------------------
						 * decrement the off-diagonal nnz
						 *---------------------------------------------------------------------------*/
						--nnz;
					} else {
						/*---------------------------------------------------------------------------
						 * compute a, t, c, s
						 *---------------------------------------------------------------------------*/
						a = 2 * aij;
            b = aii * aii - ajj * ajj;
						t = sqrt(a * a + b * b);
            if (b > 0) {
  						c = sqrt((b + t) / (2 * t));
	  					s = a / (2 * t * c);
            } else {
  						s = sqrt((t - b) / (2 * t));
	  					c = a / (2 * t * s);
            }

						/*---------------------------------------------------------------------------
						 * set the values of matrix U
						 *---------------------------------------------------------------------------*/
            dataU[i * (n + 1)] = c;
            dataU[j * n + i] = -1.0 * s;
            dataU[i * n + j] = s;
            dataU[j * (n + 1)] = c;
#if 0
						MatSetValue(U, i, i, c, INSERT_VALUES);
						MatSetValue(U, i, j, -1.0 * s, INSERT_VALUES);
						MatSetValue(U, j, i, s, INSERT_VALUES);
						MatSetValue(U, j, j, c, INSERT_VALUES);
#endif
					}
				}
			}

			/*---------------------------------------------------------------------------
			 * assemble the matrix U
			 *---------------------------------------------------------------------------*/
			MatAssemblyBegin(U, MAT_FINAL_ASSEMBLY);
			MatAssemblyEnd(U, MAT_FINAL_ASSEMBLY);

			/*---------------------------------------------------------------------------
			 * apply U: A = A * U
			 *---------------------------------------------------------------------------*/
			if (even == PETSC_TRUE) {
				MatMatMult(A, U, MAT_REUSE_MATRIX, PETSC_DEFAULT, &B);
				even = PETSC_FALSE;
			} else {
				MatMatMult(B, U, MAT_REUSE_MATRIX, PETSC_DEFAULT, &A);
				even = PETSC_TRUE;
			}
		}
		++niter;
	} while (nnz > 0);
	//PetscPrintf(PETSC_COMM_SELF, "Jacobi1 total iter = %d\n", niter);

  MatDenseRestoreArray(A, &dataA);
  MatDenseRestoreArray(U, &dataU);

	/*---------------------------------------------------------------------------
	 * copy back from B to A if necessary
	 *---------------------------------------------------------------------------*/
	if (even == PETSC_FALSE) {
		MatCopy(B, A, SAME_NONZERO_PATTERN);
	}

	/*---------------------------------------------------------------------------
	 * compute the eigenvalues and eigenvectors
	 *---------------------------------------------------------------------------*/
  MatGetColumnNorms(A, NORM_2, norms);
  for (PetscInt i = 0; i < n; ++i) {
    VecSetValue(S, i, norms[i], INSERT_VALUES);
  }
  VecAssemblyBegin(S);
  VecAssemblyEnd(S);
  VecCopy(S, D);
  VecReciprocal(D);
  MatZeroEntries(U);
  MatDiagonalSet(U, D, INSERT_VALUES);
  MatMatMult(A, U, MAT_REUSE_MATRIX, PETSC_DEFAULT, &V);
	
	/*---------------------------------------------------------------------------
	 * deallocate U, B and D
	 *---------------------------------------------------------------------------*/
	MatDestroy(&U);
	MatDestroy(&B);
	VecDestroy(&D);
}
