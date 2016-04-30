#include <cassert>
#include <iostream>
#include <cstdlib>
#include <cmath>

#include "linear_solver.h"

#include <omp.h>

#define USE_INTEL_MKL

#ifdef USE_INTEL_MKL
#include "mkl.h"
typedef MKL_INT LINEAR_INT;
#else
typedef int LINEAR_INT;
#endif

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



void dump_mat(LINEAR_INT m, LINEAR_INT n, double* mat, bool col_major, const char* name) {
 printf("\nDumping %s : \n", name);
 for (LINEAR_INT i = 0; i < m; i++) {
   printf("[ ");
   for (LINEAR_INT j = 0; j < n; j++) {
     if (col_major) { // column maor
       printf(" %.5lf ", mat[j*m+i]);
     } else {         // row major
       printf(" %.5lf ", mat[i*n+j]);
     }
   }
   printf(" ] \n");
 }
}

void vec_daxpy(LINEAR_INT n,
               double* y,
	       double alpha,
	       double* x) {
#ifdef USE_INTEL_MKL
cblas_daxpy(n, alpha, x, 1, y, 1);
#else

#pragma omp parallel for
  for (LINEAR_INT i = 0; i < n; i++){
    y[i] += alpha * x[i];
  }

#endif

}

void csr_vec_mult(LINEAR_INT n,
                  LINEAR_INT* A_ia,
		  LINEAR_INT* A_ja,
		  double* A_values,
		  LINEAR_INT r,
		  double* Q1,
		  double* v,
		  double* Av,
		  double* t1,
		  double* t2) {

#ifdef USE_INTEL_MKL
  // The interface does NOT match with the manual!!!
  //mkl_dcsrgemv(transa, m, a, ia, ja, x, y);
  char c = 'N';
  mkl_dcsrgemv(&c, &n, A_values, A_ia, A_ja, v, Av);

//void cblas_dgemv (const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans, const MKL_INT m, const MKL_INT n, const double alpha, const double *a, const MKL_INT lda, const double *x, const MKL_INT incx, const double beta, double *y, const MKL_INT incy);

  cblas_dgemv(CblasColMajor, CblasTrans, n, r, 1.0, Q1, n, Av, 1, 0, t1, 1);
  cblas_dgemv(CblasColMajor, CblasNoTrans, n, r, 1.0, Q1, n, t1, 1, 0, t2, 1);

#else
#pragma omp parallel for
  for (LINEAR_INT i = 0; i < n; i++) {
    Av[i] = 0;
    for (LINEAR_INT j= A_ia[i] - 1; j < A_ia[i+1] - 1; j++) {
      Av[i] += A_values[j] * v[A_ja[j] - 1];
    }
  }

#pragma omp parallel for
  for (LINEAR_INT i = 0; i < r; i++) {
    t1[i] = 0;
    for (LINEAR_INT j = 0 ; j < n; j++) {
      t1[i] += Q1[i*n+j]*Av[j];
    }
  }

#pragma omp parallel for
  for (LINEAR_INT i = 0; i < n; i++) {
    t2[i] = 0;
    for (LINEAR_INT j = 0; j < r; j++) {
      t2[i] += Q1[j*n+i]*t1[j];
    }
  }
#endif
  vec_daxpy(n, Av, -1.0, t2);
}

double dot_prod(LINEAR_INT n,
                double* a,
		double* b) {
  double s = 0;
#ifdef USE_INTEL_MKL
  s = cblas_ddot(n, a, 1, b, 1);
#else
#pragma omp parallel for reduction(+:s)
  for (LINEAR_INT i = 0; i < n; i++) {
    s += a[i] * b[i];
  }
  //std::cout << "dot_prod = " << s << std::endl;
#endif
  return s;
}


void vec_scale(LINEAR_INT n,
               double* x,
	       double alpha) {
#ifdef USE_INTEL_MKL
//cblas_dscal (const MKL_INT n, const double a, double *x, const MKL_INT incx);
cblas_dscal(n, alpha, x, 1);
#else
#pragma omp parallel for
  for (LINEAR_INT i = 0; i < n; i++){
    x[i] *= alpha;
  }
#endif
}
void arnoldi_process(LINEAR_INT n,
                     LINEAR_INT* A_ia,
		     LINEAR_INT* A_ja,
		     double* A_values,
		     LINEAR_INT r,
                     double* Q1,
		     LINEAR_INT k,
		     double* Vk,
		     double* Tk,
		     double* t1,
		     double* t2) {
  csr_vec_mult(n, A_ia, A_ja, A_values, r, Q1, &(Vk[(k-1)*n]), &(Vk[k*n]), t1, t2);
  //dump_mat(n, 5, Vk, true, "Vk-1:"); // col major
  double alpha_k = dot_prod(n, &(Vk[(k-1)*n]), &(Vk[k*n])); 
  Tk[(k-1)*5+2] = alpha_k;
  vec_daxpy(n, &(Vk[k*n]), -alpha_k, &(Vk[(k-1)*n]));
  if (k>=2) {
    vec_daxpy(n, &(Vk[k*n]), -Tk[(k-1)*5+1], &(Vk[(k-2)*n]));
  }
  //dump_mat(n, 5, Vk, true, "Vk-2:"); // col major
  double beta_kp1 = sqrt(dot_prod(n, &(Vk[k*n]), &(Vk[k*n])));
  Tk[(k-1)*5+0] = 0;
  Tk[(k-1)*5+2] = alpha_k;
  Tk[(k-1)*5+3] = beta_kp1;
  Tk[(k-1)*5+4] = 0;
  Tk[k*5+1] = beta_kp1;
  vec_scale(n, &(Vk[k*n]), 1.0 / beta_kp1);
}

void linear_solver(LINEAR_INT* A_ia,
                   LINEAR_INT* A_ja,
		   double* A_values, // CSR format of full matri A
                   double* Q1,          // n * s double, column major
                   double* rhs,         // n * s double, column major
		   double* sol,    // n * s double, column major
		   LINEAR_INT n,
		   LINEAR_INT r) {
  // [-2, -1, 0, 1, 2]; 
  double* t1 = (double*)malloc(sizeof(double) * r);
  double* t2 = (double*)malloc(sizeof(double) * n);
  double* Vk = (double*)malloc(sizeof(double) * (LINEAR_SOLVER_MAX_ITER+1) * n);
  double* Tk = (double*)malloc(sizeof(double) * LINEAR_SOLVER_MAX_ITER * 5);
  double* Lkd = (double*)malloc(sizeof(double) * LINEAR_SOLVER_MAX_ITER * 5);
  double* Lk = (double*)malloc(sizeof(double) * LINEAR_SOLVER_MAX_ITER * 5);
  double* ci = (double*)malloc(sizeof(double) * LINEAR_SOLVER_MAX_ITER);
  double* si = (double*)malloc(sizeof(double) * LINEAR_SOLVER_MAX_ITER);
  double* g = (double*)malloc(sizeof(double) * LINEAR_SOLVER_MAX_ITER);
  double* yk = (double*)malloc(sizeof(double) * LINEAR_SOLVER_MAX_ITER);
  double* mk = (double*)malloc(sizeof(double) * LINEAR_SOLVER_MAX_ITER * n);
  double* x = (double*)malloc(sizeof(double) * n);
  double* res = (double*)malloc(sizeof(double) * n);
for (LINEAR_INT l = 0; l < r; l++) {
  for (LINEAR_INT i = 0; i < n; i++) {
    Vk[i] = rhs[l*n+i];
    x[i] = 0;
  }
  //dump_mat(1, n, Vk, false, "rhs:");

  double beta_1 = sqrt(dot_prod(n, Vk, Vk));
  double res_norm = beta_1;
  //std::cout << "beta_1 = " << beta_1 << std::endl;
  vec_scale(n, Vk, 1.0 / beta_1);
  LINEAR_INT k = 1;

  // 1 more arnoldi to get beta_kp1
  while (k < LINEAR_SOLVER_MAX_ITER && 
  //while (k < 5 &&
         res_norm > LINEAR_SOLVER_ABS_TOL &&
	 res_norm > LINEAR_SOLVER_REL_TOL * beta_1) {
    arnoldi_process(n, A_ia, A_ja, A_values, 0, Q1, k, Vk, Tk, t1, t2);

    /*
    for (LINEAR_INT i = 0; i < k; i++) {
      for (LINEAR_INT j = 0; j < 5; j++) {
        Lk[i*5+j] = Tk[i*5+j];
      }
    }
    for (LINEAR_INT i = 0; i < k ;i++) {
      double c = Lk[i*5+2];
      double s = -Lk[i*5+3];
      ci[i] = c / sqrt(c*c+s*s);
      si[i] = s / sqrt(c*c+s*s);
      c = ci[i]; s = si[i];
      double tx, ty;
      tx = Lk[i*5+2]*c + Lk[i*5+3]*(-s);
      ty = Lk[i*5+2]*s + Lk[i*5+3]*c;
      Lk[i*5+2] = tx; Lk[i*5+3] = ty;
      tx = Lk[(i+1)*5+1]*c + Lk[(i+1)*5+2]*(-s);
      ty = Lk[(i+1)*5+1]*s + Lk[(i+1)*5+2]*c;
      Lk[(i+1)*5+1] = tx; Lk[(i+1)*5+2] = ty;
      tx = Lk[(i+2)*5+0]*c + Lk[(i+2)*5+1]*(-s);
      ty = Lk[(i+2)*5+0]*s + Lk[(i+2)*5+1]*c;
      Lk[(i+2)*5+0] = tx; Lk[(i+2)*5+1] = ty;
    }
    */

    for (LINEAR_INT i = 0; i < 5; i++) {
      Lk[(k-1)*5+i] = Tk[(k-1)*5+i]; // only copy line (k-1);
    }
    double c, s, tx, ty;
    if (k - 3 >= 0) { // perfrom on 0, 1
      c = ci[k-3]; s = si[k-3];
      tx = Lk[(k-1)*5+0] * c + Lk[(k-1)*5+1]*(-s);
      ty = Lk[(k-1)*5+0] * s + Lk[(k-1)*5+1]*c;
      Lk[(k-1)*5+0] = tx; Lk[(k-1)*5+1] = ty;
    }
    if (k - 2 >= 0) { // perform on 1, 2
      c = ci[k-2]; s = si[k-2];
      tx = Lk[(k-1)*5+1] * c + Lk[(k-1)*5+2]*(-s);
      ty = Lk[(k-1)*5+1] * s + Lk[(k-1)*5+2]*c;
      Lk[(k-1)*5+1] = tx; Lk[(k-1)*5+2] = ty;
    }
    // for this one perform on 2, 3
    c = Lk[(k-1)*5+2]; s = -Lk[(k-1)*5+3];
    ci[k-1] = c / sqrt(c*c+s*s);
    si[k-1] = s / sqrt(c*c+s*s);
    c = ci[k-1]; s = si[k-1];
    tx = Lk[(k-1)*5+2]*c + Lk[(k-1)*5+3]*(-s);
    ty = Lk[(k-1)*5+2]*s + Lk[(k-1)*5+3]*c;
    Lk[(k-1)*5+2] = tx; Lk[(k-1)*5+3] = ty;

    //dump_mat(n, 5, Vk, true, "Vk:"); // col major
    //dump_mat(5, 5, Tk, false, "Tk:"); // row major
    //dump_mat(n, 5, Lk, false, "Lk:");
    //dump_mat(1, k, ci, false, "ci:");
    //dump_mat(1, k, si, false, "si:");
    /*
    for (LINEAR_INT i = 0; i < k; i++) {
      g[i] = ci[i];
      for (LINEAR_INT j = 0; j < i; j++) {
        g[i] *= si[j];
      }
    }
    dump_mat(1, k, g, false, "g:");
    for (LINEAR_INT i = 0; i < k; i++) {yk[i] = g[i];}
    // fuck me, we are solving Lk^T yk = gk
    for (LINEAR_INT i = k-1; i >= 0; i--) {
      yk[i] = yk[i] / Lk[i*5+2];
      if (i-1 >= 0) {
        yk[i-1] -= Lk[i*5+1] * yk[i];
      }
      if (i-2 >= 0) {
        yk[i-2] -= Lk[i*5+0] * yk[i];
      }
      yk[i] *= beta_1;
    }
    for (LINEAR_INT i = 0; i < n; i++){
      x[i] = 0;
      for (LINEAR_INT j = 0; j < k; j++) {
        x[i] += yk[j] * Vk[j*n+i];
      }
    }
    */
    
    double tau_j = ci[k-1];
    for (LINEAR_INT i = 0; i < k-1; i++) {tau_j *= si[i];}
    //printf("tau_j = %.8lf\n", tau_j );
    for (LINEAR_INT i = 0; i < n; i++) {mk[(k-1)*n+i] = Vk[(k-1)*n+i];}
    for (LINEAR_INT i = 0; i < n; i++) {
      if (k - 3 >= 0) {
        mk[(k-1)*n+i] -= Lk[(k-1)*5+0] * mk[(k-3)*n+i];
      }
      if (k - 2 >= 0) {
        mk[(k-1)*n+i] -= Lk[(k-1)*5+1] * mk[(k-2)*n+i];
      }
      mk[(k-1)*n+i] /= Lk[(k-1)*5+2];
      x[i] = x[i] + beta_1 * tau_j * mk[(k-1)*n+i];
    }
    
    csr_vec_mult(n, A_ia, A_ja, A_values, 0, Q1, x, res, t1, t2);
    vec_daxpy(n, res, -1.0, &(rhs[n*l]));
    res_norm = dot_prod(n, res, res);
    //dump_mat(1, n, yk, false,"yk:");
    //dump_mat(1, n, &mk[(k-1)*n], false, "mk:");
    //dump_mat(1, n, x,false, "x:");
    //prLINEAR_INTf("Iter[%d] res = %.8lf\n", k, res_norm);
    if (res_norm < LINEAR_SOLVER_ABS_TOL ||
        res_norm < LINEAR_SOLVER_REL_TOL * beta_1) {
      printf("RHS[%d] Converged at iter %d, residual = %.6lf\n", l, k, res_norm);
    }
    k++;
  } // while
  for (LINEAR_INT i = 0; i < n; i++) {
    sol[l*n+i] = x[i];
  }
} // for l=1:s
}

/*
LINEAR_INT main() {
  LINEAR_INT n = 100000;
  LINEAR_INT s = 4;
  LINEAR_INT* A_ia = new LINEAR_INT[n+1];
  LINEAR_INT* A_ja = new LINEAR_INT[n*3-2];
  double* A_values = new double[n*3-2];
  double* rhs = new double[n * s];
  double* sol = new double[n * s];
  LINEAR_INT curr_num = 0;
  for (LINEAR_INT i = 0; i < n; i++) {
    A_ia[i] = curr_num+1;
    if (i != 0) {
      A_ja[curr_num] = i;
      A_values[curr_num] = -1;
      curr_num++;
    }
    A_ja[curr_num] = i+1;
    A_values[curr_num] = 4;
    curr_num++;
    if (i != n-1) {
      A_ja[curr_num] = i + 2;
      A_values[curr_num] = -1;
      curr_num++;
    }
  }
  A_ia[n] = curr_num + 1;
  for (LINEAR_INT i = 0; i < n; i++) {
    rhs[i] = 1;
    rhs[i+1*n] = (i+1);
    rhs[i+2*n] = sin(i);
    rhs[i+3*n] = sqrt(i);
  }
  double ddum;
  //dump_mat(5, 4, rhs, true, "RHS:");
  double t_start = omp_get_wtime();
  std::cout << "Running with " << omp_get_num_threads() << " threads" << std::endl;
  linear_solver(A_ia, A_ja, A_values, &ddum, rhs, sol, n, s); 
  double t_end = omp_get_wtime();
  std::cout << "time_elapsed = " << t_end - t_start << std::endl;
}
*/
