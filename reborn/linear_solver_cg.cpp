
/*
 A = [2 1 0 0 0
      1 2 1 0 0 
      0 1 2 1 0
      0 0 1 2 1
      0 0 0 1 2]
 A_ia     = {0 2 5 8 11 13}
 A_ja     = {0 1 0 1 2 1 2 3 2 3 4 3 4}
 A_values = {2 1 1 2 1 1 2 1 1 2 1 1 2}
*/

#include "linear_solver.h" // parameters
#include <omp.h>

#include <cmath>
#include <cstdio>
#include "mkl.h"

#define CG_MAX_ITER 1000

void imqqax  (const MKL_INT *,const MKL_INT *, const double *, const double *, const double *, double*, double *, int, int);
void cg_core (const MKL_INT *,const MKL_INT *, const double *, const double *, double *, double*, double*, double*, double*, MKL_INT, MKL_INT, int&	);
void cg_core2(MKL_INT*, MKL_INT*, double*, double*, double*, double*, double*, double*, double*, MKL_INT, MKL_INT, int&, double	);

/* implements CG linear solver with OpenMp */

//linear_solver(long long const*, long long const*, double const*, double const*, double const*, double*, long long, long long)'

void linear_solver(
                   const MKL_INT * A_ia,
                   const MKL_INT * A_ja,
                   const double  *  A_values, // CSR format of full matri A
                   const double  *  Q1,          // n * s double, column major
                   const double  *  RHS,         // n * s double, column major
                   double*  solution,    // n * s double, column major
                   MKL_INT n,
                   MKL_INT s) {
  
  printf("wocao/n");
  fflush(stdout);
  int its;
return;
  double  *RHSwrt = new double [n*s];   //workspace of p in cg
  double  *RHSCOPY = new double [n*s];   //workspace of p in cg
  double  *WKSPACE1 = new double [n*s];   //workspace of Ap in cg      each col need size== p
  double  *WKSPACE2 = new double [n*s];    //workspace of I-QQ'Ax in cg  each col need size== sizex	  
return;
#pragma omp parallel for
  for(int i=0; i<n*s; i++){
    RHSwrt[i]=RHSCOPY[i] = RHS[i];
  }

#pragma omp parallel for
  for(int j=0; j<s; j++){
    cg_core(A_ia, A_ja, A_values, Q1, RHSwrt+n*j, RHSCOPY+n*j, WKSPACE1+n*j, WKSPACE2+n*j, solution+n*j, n, s, its);
  }


  delete[] RHSCOPY;
  return;
}

void linear_solver2(
                   MKL_INT* A_ia,
                   MKL_INT* A_ja,
                   double*  A_values, // CSR format of full matri A
                   double*  Q1,          // n * s double, column major
                   double*  RHS,         // n * s double, column major
                   double*  solution,    // n * s double, column major
                   double*   sig,
                   MKL_INT n,
                   MKL_INT s) {
  int its;
  double  *RHSCOPY = new double [n*s];   //workspace of p in cg
  double  *WKSPACE1 = new double [n*s];   //workspace of Ap in cg      each col need size== p
  double  *WKSPACE2 = new double [n*s];    //workspace of I-QQ'Ax in cg  each col need size== sizex	  

 double last = sig[s-1];
  if(last==0){
    fprintf(stderr,"largest eigenvalue ==0\n");
    exit(0);
  }

  for(int i=0;i<s;i++){
    sig[i]/=last;
  }

  if(s<=1){
    fprintf(stderr,"err, s must >=2\n");
    exit(0);
  }
  sig[s-1]=sig[s-2];




#pragma omp parallel for
  for(int i=0; i<n*s; i++){
    RHSCOPY[i] = RHS[i];
  }

#pragma omp parallel for
  for(int j=0; j<s; j++){
    cg_core2(A_ia, A_ja, A_values, Q1, RHS+n*j, RHSCOPY+n*j, WKSPACE1+n*j, WKSPACE2+n*j, solution+n*j, n, s, its, sig[j]);
  }


  delete[] RHSCOPY;
  return;
}



/* implement (I-QQ')Ax */
void imqqax(
   const  MKL_INT* ia,
    const MKL_INT* ja,
    const double*  va,
    const double*  Q,
		const double*  X,
		double*  QtAX,
		double*  Y,
		MKL_INT  n,
		MKL_INT  s) {
  
  for(int i=0; i<n; i++){
    Y[i]=0.0;
    for(int j, jt=ia[i]; jt<ia[i+1]; jt++){
        Y[i]+=X[ ja[jt] ]*va[jt];
  	}
  }
  
  cblas_dgemv(CblasColMajor,CblasTrans,n,s,1.0,Q, n, Y,  1,  0,QtAX, 1);
  cblas_dgemv(CblasColMajor, CblasNoTrans, n,s,-1.0,Q,n, QtAX,1, 1,Y, 1);

}


double norm2(double *x,  MKL_INT n){
	return cblas_dnrm2 (n, x, 1);
  //double rlt=0.0;
	//for(int i=0; i<n; i++)
	//	rlt+=x[i]*x[i];
	//return sqrt(rlt);
}


void cg_core(const MKL_INT* A_ia, 
             const MKL_INT* A_ja,
             const double*  A_v,
             const double*  Q1,
             double *  rhs,
             double*  rhscopy,
             double*  wkspace1,
             double*  wkspace2,
             double*  solution,
             MKL_INT  n,
             MKL_INT  s,
             int      &itr_out
             ) {
	int    iter=0;
  //double *r =const_cast<double*>(rhs); 
  double *&r =rhs; 
	double *&p =rhscopy;
	double rnorm=0.0, rnorm2;
	double *&Ap = wkspace1;
	double *&qqtax = wkspace2;
	double alpha=0.0;
	double beta =1;
	double *&x = solution;
  double pAp = 0.0;
  rnorm = norm2(r,n);

  for(iter=1;  iter<=CG_MAX_ITER; iter++) {
      imqqax(A_ia, A_ja, A_v, Q1, p, qqtax, Ap, n,s);
      pAp   = cblas_ddot(n, p, 1, Ap,1); 
      if(pAp==0){
        fprintf(stderr,"err! pAp=0 in cg!");
        exit(1);
      }
      alpha = rnorm/pAp;
      cblas_daxpy (n, alpha, p, 1, x, 1);
      cblas_daxpy (n, -alpha,Ap,1, r, 1); 
        
      if( (rnorm2=norm2(r,n)) < 10e-6) {
        fprintf(stderr,"\nrnrom<10e-6\n");
        break;
      }
      beta = (rnorm2/rnorm);
      #pragma omp parallel for
      for(int i=0;i<n; i++){
        p[i] = beta*p[i]+ r[i];
      }
      rnorm=rnorm2;
  }
  itr_out = iter;
}



void cg_core2(MKL_INT* A_ia, 
              MKL_INT* A_ja,
              double*  A_v,
              double*  Q1,
              double*  rhs,
              double*  rhscopy,
              double*  wkspace1,
              double*  wkspace2,
              double*  solution,
              MKL_INT  n,
              MKL_INT  s,
              int      &itr_out,
		          double   sigm
             ) {
	int    iter=0;
  double *&r =rhs; 
	double *&p =rhscopy;
	double rnorm=0.0, rnorm2;
	double *&Ap = wkspace1;
	double *&qqtax = wkspace2;
	double alpha=0.0;
	double beta =1;
	double *&x = solution;
  double pAp = 0.0;
  rnorm = norm2(r,n);
  double b=rnorm ; //norm of rhs 
 
  for(iter=1;  iter<=CG_MAX_ITER; iter++) {
      imqqax(A_ia, A_ja, A_v, Q1, p, qqtax, Ap, n,s);
      pAp   = cblas_ddot(n, p, 1, Ap,1); 
      if(pAp==0){
        fprintf(stderr,"err! pAp=0 in cg!");
        exit(1);
      }
      alpha = rnorm/pAp;
      cblas_daxpy (n, alpha, p, 1, x, 1);
      cblas_daxpy (n, -alpha,Ap,1, r, 1); 
      
      if((alpha*rnorm*rnorm < sigm*sigm* b*b) ||
         (rnorm2=norm2(r,n)) < 10e-6) {
        fprintf(stderr,"\nrnrom<10e-6\n");
        break;
      }
      beta = (rnorm2/rnorm);
      #pragma omp parallel for
      for(int i=0;i<n; i++){
        p[i] = beta*p[i]+ r[i];
      }
      rnorm=rnorm2;
  }
  itr_out = iter;
}
