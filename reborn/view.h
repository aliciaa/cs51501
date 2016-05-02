#include <mkl_types.h>

/* Print a CSR matrix with size n
 * @input name name of the matrix
 * @input n number of rows
 * @input ia row pointers
 * @input ja column indices
 * @input va values
 */
void ViewCSR(char *name,
             MKL_INT n,
             MKL_INT *ia,
             MKL_INT *ja,
             double *va);

/* Print a dense matrix in column major
 * @input name name of the matrix
 * @input n number of rows
 * @input m number of columns
 * @input v values
 */
void ViewDense(char *name,
               MKL_INT n,
               MKL_INT m,
               double* v);
