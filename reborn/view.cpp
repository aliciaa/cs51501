#include "view.h"
#include <cstdio>

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
             double *va)
{
  printf("%s:\n", name);
  for (int i = 0; i < n; ++i) {
    printf("row %d\n", i);
    for (int j = ia[i]-1; j < ia[i+1]-1; ++j) {
      printf("col %d = %.6lf; ", ja[j], va[j]);
    }
    printf("\n");
  }
}

/* Print a dense matrix in column major
 * @input name name of the matrix
 * @input n number of rows
 * @input m number of columns
 * @input v values
 */
void ViewDense(char *name,
               MKL_INT n,
               MKL_INT m,
               double* v)
{
  printf("%s:\n", name);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      printf("%.6lf ", v[j * n + i]);
    }
    printf("\n");
  }
}
