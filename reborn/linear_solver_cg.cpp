
/*
 A = [2 1 0 0 0
      1 2 1 0 0 
      0 1 2 1 0
      0 0 1 2 1
      0 0 0 1 2]
 A_ia = {0 2 5 8 11 13}
 A_ja =     {0 1 0 1 2 1 2 3 2 3 4 3 4 5 4 5 }
 A_values = {2 1 1 2 1 1 2 1 1 2 1 1 2 1 1 2 }
*/

#include "linear_solver.h" // parameters
#include <omp.h>

/* implements CG linear solver with OpenMp */

void linear_solver(MKL_INT* A_ia,
                   MKL_INT* A_ja,
		   double* A_values, // CSR format of full matri A
                   double* Q1,          // n * s double, column major
                   double* RHS,         // n * s double, column major
		   double* solution,    // n * s double, column major
		   MKL_INT n,
		   MKL_INT s) {


}

