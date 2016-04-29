#include <cassert>
#include <iostream>
#include <cstdlib>
#include <cmath>

#include "linear_solver.h"

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

typedef int LINEAR_INT;

void dump_mat(int m, int n, double* mat, bool col_major, const char* name) {
 printf("\nDumping %s : \n", name);
 for (int i = 0; i < m; i++) {
   printf("[ ");
   for (int j = 0; j < n; j++) {
     if (col_major) { // column maor
       printf(" %.5lf ", mat[j*m+i]);
     } else {         // row major
       printf(" %.5lf ", mat[i*n+j]);
     }
   }
   printf(" ] \n");
 }
}

void csr_vec_mult(LINEAR_INT n,
                  LINEAR_INT* A_ia,
		  LINEAR_INT* A_ja,
		  double* A_values,
		  double* v,
		  double* Av) {
  for (int i = 0; i < n; i++) {
    Av[i] = 0;
    for (int j= A_ia[i]; j < A_ia[i+1]; j++) {
      Av[i] += A_values[j] * v[A_ja[j]];
    }
  }
}

double dot_prod(LINEAR_INT n,
                double* a,
		double* b) {
  double s = 0;
  for (int i = 0; i < n; i++) {
    s += a[i] * b[i];
  }
  std::cout << "dot_prod = " << s << std::endl;
  return s;
}

void vec_daxpy(LINEAR_INT n,
               double* y,
	       double alpha,
	       double* x) {
  for (int i = 0; i < n; i++){
    y[i] += alpha * x[i];
  }
}
void vec_scale(LINEAR_INT n,
               double* x,
	       double alpha) {
  for (int i = 0; i < n; i++){
    x[i] *= alpha;
  }
}
void arnoldi_process(LINEAR_INT n,
                     LINEAR_INT* A_ia,
		     LINEAR_INT* A_ja,
		     double* A_values,
		     LINEAR_INT k,
		     double* Vk,
		     double* Tk) {
  csr_vec_mult(n, A_ia, A_ja, A_values, &(Vk[(k-1)*n]), &(Vk[k*n]));
  dump_mat(n, 5, Vk, true, "Vk-1:"); // col major
  double alpha_k = dot_prod(n, &(Vk[(k-1)*n]), &(Vk[k*n])); 
  Tk[(k-1)*5+2] = alpha_k;
  vec_daxpy(n, &(Vk[k*n]), -alpha_k, &(Vk[(k-1)*n]));
  if (k>=2) {
    vec_daxpy(n, &(Vk[k*n]), -Tk[(k-1)*5+1], &(Vk[(k-2)*n]));
  }
  dump_mat(n, 5, Vk, true, "Vk-2:"); // col major
  double beta_kp1 = sqrt(dot_prod(n, &(Vk[k*n]), &(Vk[k*n])));
  Tk[(k-1)*5+3] = beta_kp1;
  Tk[k*5+1] = beta_kp1;
  vec_scale(n, &(Vk[k*n]), 1.0 / beta_kp1);
}

void linear_solver(LINEAR_INT* A_ia,
                   LINEAR_INT* A_ja,
		   double* A_values, // CSR format of full matri A
                   double* Q1,          // n * s double, column major
                   double* RHS,         // n * s double, column major
		   double* solution,    // n * s double, column major
		   LINEAR_INT n,
		   LINEAR_INT s) {
  assert(s == 1);
  // [-2, -1, 0, 1, 2]; 
  double* Vk = (double*)malloc(sizeof(double) * (LINEAR_SOLVER_MAX_ITER+1) * n);
  double* Tk = (double*)malloc(sizeof(double) * LINEAR_SOLVER_MAX_ITER * 5);
  double* Lkd = (double*)malloc(sizeof(double) * LINEAR_SOLVER_MAX_ITER * 5);
  double* Lk = (double*)malloc(sizeof(double) * LINEAR_SOLVER_MAX_ITER * 5);
  for (int i = 0; i < n; i++) {
    Vk[i] = RHS[i];
  }
  double alpha_1 = sqrt(dot_prod(n, RHS, RHS));
  std::cout << "alpha_1 = " << alpha_1 << std::endl;
  for (int i = 0; i < 5; i++) {Tk[i] = 0;}
  vec_scale(n, Vk, 1.0 / alpha_1);
  int k = 1;
  // 1 more arnoldi to get beta_kp1
  while (k < 5) {
    arnoldi_process(n, A_ia, A_ja, A_values, k, Vk, Tk);
    k++;
  }
  dump_mat(n, 5, Vk, true, "Vk:"); // col major
  dump_mat(5, 5, Tk, false, "Tk:"); // row major
}

int main() {
  int n = 10;
  int* A_ia = new int[n+1];
  int* A_ja = new int[n*3-2];
  double* A_values = new double[n*3-2];
  double* rhs = new double[n];
  double* sol = new double[n];
  int curr_num = 0;
  for (int i = 0; i < n; i++) {
    A_ia[i] = curr_num;
    if (i != 0) {
      A_ja[curr_num] = i-1;
      A_values[curr_num] = -1;
      curr_num++;
    }
    A_ja[curr_num] = i;
    A_values[curr_num] = 4;
    curr_num++;
    if (i != n-1) {
      A_ja[curr_num] = i + 1;
      A_values[curr_num] = -1;
      curr_num++;
    }
    rhs[i] = (i+1);
  }
  A_ia[n] = curr_num;
  double ddum;
  linear_solver(A_ia, A_ja, A_values, &ddum, rhs, sol, n, 1); 
}
