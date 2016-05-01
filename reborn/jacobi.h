#ifndef JACOBI
#define JACOBI

static const double JACOBI_TOL = 1e-12; // tolerance of Jacobi methods

/* Generate the order of annihilation
 * @input n dimension of the matrix
 * @output w width of the matrix orders
 * @output orders a matrix representing the order of annihilation with rows being the order and columns
 * 							 	storing the orders of the off-diagonals that can be annihilated simultaneously
 */
void GenerateAnnihilationOrder(const int n,
															 int & w,
															 int *& orders);

/* Compute the eigen decomposition using 1-sided Jacobi method, A = VDV^T
 * @input orders a matrix representing the order of annihilation with rows being the order and columns
 * 						 	 storing the orders of the off-diagonals that can be annihilated simultaneously
 * @input n dimension of the matrix
 * @input w width of the matrix orders
 * @input A the matrix to be decomposed
 * @output A the matrix contains the eigenvectors of A
 * @output S the vector contains the eigenvalues of A
 * @input spd indicate that the matrix A is spd and thus only has positive eigenvalues
 */
void Jacobi1(const int* orders,
						 const int n,
						 const int w,
						 double *&A,
						 double *&S,
             bool spd = true);

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
						 double *S);

#endif // JACOBI
