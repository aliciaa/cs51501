#include "jacobi.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <mkl.h>
#include <omp.h>

/* Generate the order of annihilation
 * @input n dimension of the matrix
 * @output w width of the matrix orders
 * @output orders a matrix representing the order of annihilation with rows being the order and columns
 * 							 	storing the orders of the off-diagonals that can be annihilated simultaneously
 */
void GenerateAnnihilationOrder(const int n,
															 int & w,
															 int *& orders)
{
	const int m = n / 2;				// m = floor(n / 2)

	/*---------------------------------------------------------------------------
	 * allocate the memory
	 *---------------------------------------------------------------------------*/
	w = m * 2;											// width of the matrix orders
	orders = new int[n * w];

	/*---------------------------------------------------------------------------
	 * assign the values
	 *---------------------------------------------------------------------------*/
	for (int k = 0; k < n; ++k) {
		const int s = n - k,
							b = s / 2 + 1;
		for (int p = 0; p < m; ++p) {
			const int j = b + p,
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

/* Compute the eigen decomposition using 1-sided Jacobi method, A = VDV^T
 * @input orders a matrix representing the order of annihilation with rows being the order and columns
 * 						 	 storing the orders of the off-diagonals that can be annihilated simultaneously
 * @input n dimension of the matrix
 * @input w width of the matrix orders
 * @input A the matrix to be decomposed
 * @output A the matrix V^T * A * V
 * @output A the matrix contains the eigenvectors of A
 * @output S the vector contains the eigenvalues of A
 */
void Jacobi1(const int* orders,
					 	 const int n,
						 const int w,
						 double *A,
						 double *S)
{

	/*---------------------------------------------------------------------------
	 * declare an orthogonal matrix U for intermediate step and a matrix B
	 *---------------------------------------------------------------------------*/
  double *U = new double[n * n];
  double *B = new double[n * n];

	/*---------------------------------------------------------------------------
	 * declare variables
	 *---------------------------------------------------------------------------*/
	const double tol = 1e-12;	    			// tolerance
	const int m = n / 2;						    // max number of annihiliations
	double aii, ajj, aij,					      // norms of cols i and j; dot product of cols i and j
				 a, b, t, c, s;	      				// intermediate values
	int nnz,												    // number of nonzero off-diagonals
	    i, j,											      // indices
			p, q,											      // counting variables
			niter = 0;
	bool even = true;				            // even number of rotations

	do {
		nnz = n * (n - 1) / 2;						// set the total number of nonzeros

		for (int k = 0; k < n; ++k) {
			const int* currOrders = orders + k * w;
			/*---------------------------------------------------------------------------
			 * initialize the matrix U to be the identity
			 *---------------------------------------------------------------------------*/
			for (p = 0; p < n; ++p) {
        for (q = 0; q < n; ++q) {
          U[q * n + p] = (p == q) ? 1.0 : 0.0;
        }
			}

//#pragma omp parallel for private(p, q, i, j, aii, ajj, aij, a, b, t, c, s) reduction(-:nnz)
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
          if (even) {
            for (q = 0; q < n; ++q) {
              a = A[i * n + q];
              b = A[j * n + q];
              aii += a * a;
              ajj += b * b;
              aij += a * b;
            }
          } else {
            for (q = 0; q < n; ++q) {
              a = B[i * n + q];
              b = B[j * n + q];
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
            U[i * (n + 1)] = c;
            U[j * n + i] = -1.0 * s;
            U[i * n + j] = s;
            U[j * (n + 1)] = c;
					}
				}
			}

			/*---------------------------------------------------------------------------
			 * apply U: A = A * U
			 *---------------------------------------------------------------------------*/
			if (even) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, U, n, 0.0, B, n);
				even = false;
			} else {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, B, n, U, n, 0.0, A, n);
				even = true;
			}
		}
		++niter;
	} while (nnz > 0);

	/*---------------------------------------------------------------------------
	 * copy back from B to A if necessary
	 *---------------------------------------------------------------------------*/
	if (!even) {
    memcpy(A, B, n * n * sizeof(double));
	}

	/*---------------------------------------------------------------------------
	 * compute the eigenvalues and eigenvectors
	 *---------------------------------------------------------------------------*/
  for (int i = 0; i < n; ++i) {
    S[i] = cblas_dnrm2(n, A + i*n, 1);
    cblas_dscal(n, 1.0/S[i], A + i*n, 1);
  }
	
	/*---------------------------------------------------------------------------
	 * deallocate U, B and D
	 *---------------------------------------------------------------------------*/
  delete [] U;
  delete [] B;
}

/* Compute the eigen decomposition using 2-sided Jacobi method, A = VDV^T
 * @input orders a matrix representing the order of annihilation with rows being the order and columns
 * 						 	 storing the orders of the off-diagonals that can be annihilated simultaneously
 * @input n dimension of the matrix
 * @input w width of the matrix orders
 * @input A the matrix to be decomposed
 * @output A the matrix contains the eigenvectors of A
 * @output S the vector contains the eigenvalues of A
 */
void Jacobi2(const int* orders,
					 	 const int n,
						 const int w,
						 double *A,
						 double *S)
{
	/*---------------------------------------------------------------------------
	 * declare an orthogonal matrix U for intermediate step and a vector D
	 * create the matries U, V and the vector D
	 *---------------------------------------------------------------------------*/
  double *U = new double[n * n];
  double *V = new double[n * n];
  double *W = new double[n * n];

	/*---------------------------------------------------------------------------
	 * declare variables
	 *---------------------------------------------------------------------------*/
	const double tol = 1e-12;				// tolerance
	const int m = n / 2;						// max number of annihiliations
	double norm,                    // norm of A
	       aii, ajj, aij,					  // principal submatrix values of A
				 a, t, c, s;							// intermediate values
	int nnz,												// number of nonzero off-diagonals
	    i, j,											  // indices
			p, q,												// counting variables
			niter = 0;
	bool even = true;				        // even number of rotations

	/*---------------------------------------------------------------------------
	 * set the matrix V to be the identity
	 *---------------------------------------------------------------------------*/
  for (p = 0; p < n; ++p) {
    for (q = 0; q < n; ++q) {
      V[q * n + p] = (p == q) ? 1.0 : 0.0;
    }
  }

	do {
		nnz = n * (n - 1) / 2;						// set the total number of nonzeros
    norm = LAPACKE_dlansy(LAPACK_COL_MAJOR, 'F', 'U', n, A, n);

		for (int k = 0; k < n; ++k) {
			const int* currOrders = orders + k * w;
			/*---------------------------------------------------------------------------
			 * initialize the matrix U to be the identity
			 *---------------------------------------------------------------------------*/
      for (p = 0; p < n; ++p) {
        for (q = 0; q < n; ++q) {
          U[q * n + p] = (p == q) ? 1.0 : 0.0;
        }
      }

//#pragma omp parallel for private(p, i, j, aii, ajj, aij, a, t, c, s) reduction(-:nnz)
			for (p = 0; p < m; ++p) {
				i = currOrders[2*p];
				j = currOrders[2*p+1];
				if (i != -1) {
					/*---------------------------------------------------------------------------
					 * get the values from A
					 *---------------------------------------------------------------------------*/
          aii = A[i * (n + 1)];
          aij = A[j * n + i];
          ajj = A[j * (n + 1)];

					if (fabs(aij) < tol * norm) {
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
            U[i * (n + 1)] = c;
            U[j * n + i] = -1.0 * s;
            U[i * n + j] = s;
            U[j * (n + 1)] = c;
					}
				}
			}

			/*---------------------------------------------------------------------------
			 * apply U: A = U^T * A * U
       *          V/W = W/V * U
			 *---------------------------------------------------------------------------*/
			if (even) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, U, n, 0.0, W, n);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, n, 1.0, U, n, W, n, 0.0, A, n);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, V, n, U, n, 0.0, W, n);
				even = false;
			} else {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, U, n, 0.0, V, n);
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, n, 1.0, U, n, V, n, 0.0, A, n);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, W, n, U, n, 0.0, V, n);
				even = true;
			}
		}
		++niter;
	} while (nnz > 0);

	/*---------------------------------------------------------------------------
	 * only keep the diagonal of A
	 *---------------------------------------------------------------------------*/
  for (p = 0; p < n; ++p) {
    S[p] = A[p * (n + 1)];
  }
	
	/*---------------------------------------------------------------------------
	 * copy back from V/W to A
	 *---------------------------------------------------------------------------*/
	if (even) {
    memcpy(A, V, n * n * sizeof(double));
	} else {
    memcpy(A, W, n * n * sizeof(double));
  }

	/*---------------------------------------------------------------------------
	 * deallocate U, V and W
	 *---------------------------------------------------------------------------*/
  delete [] U;
  delete [] V;
  delete [] W;
}
