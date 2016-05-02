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

#define CG_MAX_ITER 100
#define CCCC  0
#define CONVERGENCETEST 0


void i_qqax (const MKL_INT* , const MKL_INT* , const double* , 
             const double*  , const double*  , double*       , 
             double*        , int            , int);
void cg_core(const MKL_INT* , const MKL_INT* , const double* ,
             const double*  , double*        , double*       ,
             double*        , double*        , double*       ,
             MKL_INT        , MKL_INT        , int&          ,
             double );

/* implements CG linear solver with OpenMp */

void linear_solver(
    const MKL_INT*  A_ia,
    const MKL_INT*  A_ja,
    const double*   A_values,    // CSR format of full matri A
    const double*   Q1,          // n * s double, column major
    const double*   RHS,         // n * s double, column major
    double*         solution,    // n * s double, column major
    MKL_INT         n,
    MKL_INT         s,
    double*         sig) {

  int its;
  double  *RHSwrt   = new double [n*s];   //workspace of p in cg
  double  *RHSCOPY  = new double [n*s];  //workspace of p in cg
  double  *WKSPACE1 = new double [n*s]; //workspace of Ap in cg 
  double  *WKSPACE2 = new double [n*s]; //workspace of I-QQ'Ax in cg 



#if CONVERGENCETEST==1
  double lst = sig[s-1];
  if( s<=1 || ((lst>=0)?lst:-lst)< 1e-7 ){
    printf("Err! largest eigenvaluse is 0");
    exit(1);
  }
#pragma omp parallel for
  for(int i=0; i<s-1; i++){
    sig[i]/=lst;
    sig[i]*=sig[i];
  }
  sig[s-1]=sig[s-2];
#endif




#pragma omp parallel for
  for(int i=0; i<n*s; i++){
    RHSwrt[i]=RHSCOPY[i] =-RHS[i];
    solution[i]=0.0;
  }

#if CCCC==1
  printf(" \nn,s= %d %d ",n,s);
  printf(" \nA  = [ "); for(int i=0; i<s  ; printf("%f ", A_values[i++]));
  printf("]\nQ  = [ "); for(int i=0; i<n*s; printf("%f ", Q1[i++])      ); 
  printf("]\nRHS= [ "); for(int i=0; i<n*s; printf("%f ", RHS[i++])     );
#endif

#pragma omp parallel for
  for(int j=0; j<s; j++){
    cg_core(A_ia        ,   A_ja      , A_values    , 
            Q1          , RHSwrt+n*j  , RHSCOPY+n*j , 
            WKSPACE1+n*j, WKSPACE2+n*j, solution+n*j, 
            n           , s           , its         ,
            sig[j]);
  }

#if CCCC==1
  printf("\nAfter linear solution of CG solution=\n");
  for(int i=0;i<n*s; printf("%f ", solution[i++]));
#endif

  delete[] RHSCOPY;
  delete[] RHSwrt;
  delete[] WKSPACE1;
  delete[] WKSPACE2;
  return;
}


/* implement (I-QQ')Ax */
void i_qqax(
    const MKL_INT* ia,
    const MKL_INT* ja,
    const double*  va,
    const double*  Q,
    const double*  X,
    double*        QtAX,
    double*        Y,
    MKL_INT        n,
    MKL_INT        s) {

  //sparse mat-vec
  for(int i=0; i<n; i++){
    Y[i]=0.0;
    for(int jt=ia[i]; jt<ia[i+1]; jt++){
      Y[i]+=X[ ja[jt-1]-1 ]*va[jt-1];
    }
  }

  cblas_dgemv(CblasColMajor, CblasTrans  , n,s, 1.0,Q, n, Y   , 1, 0,QtAX, 1);
  cblas_dgemv(CblasColMajor, CblasNoTrans, n,s,-1.0,Q, n, QtAX, 1, 1,Y   , 1);

}


double norm2(double *x,  MKL_INT n){
  return cblas_dnrm2 (n, x, 1);
  //double rlt=0.0;
  //for(int i=0; i<n; i++)
  //  rlt+=x[i]*x[i];
  //return sqrt(rlt);
}

/* core of CG */
void cg_core(
    const MKL_INT* A_ia, 
    const MKL_INT* A_ja,
    const double*  A_v,
    const double*  Q1,
    double*        rhs,
    double*        rhscopy,
    double*        wkspace1,
    double*        wkspace2,
    double*        solution,
    MKL_INT        n,
    MKL_INT        s,
    int            &itr_out,
    double         sigsig
    ) {

  int    iter   = 0;
  double *&r    = rhs; 
  double *&p    = rhscopy;   //rename
  double *&Ap   = wkspace1;  //rename
  double *&qqax = wkspace2;  //rename
  double *&x    = solution;  //rename
  double rnorm  = 0.0;
  double rnorm2 = 0.0;
  double alpha  = 0.0;
  double beta   = 1;
  double pAp    = 0.0;
  rnorm = norm2(r,n);

#if CONVERGENCETEST==1
  double sigsigbb = 0.0;
  sigsigbb = sigsig*rnorm*rnorm;
#endif

  for(iter=1;  iter<=CG_MAX_ITER; iter++) {
    i_qqax(A_ia, A_ja, A_v, Q1, p, qqax, Ap, n, s);
    pAp = cblas_ddot(n, p, 1, Ap,1); 
    if(pAp==0){
      fprintf(stderr,"err! pAp=0 in cg!");
      exit(1);
    }
    alpha = rnorm*rnorm/pAp;

    cblas_daxpy (n, alpha, p, 1, x, 1);//upD x
    cblas_daxpy (n,-alpha,Ap, 1, r, 1);//upD r
    if( (rnorm2=norm2(r,n)) < 10e-6) {
      break;
    }
#if CONVERGENCETEST==1
    if( alpha*rnorm2*rnorm2 < sigsigbb )
      break;
#endif
    beta = (rnorm2*rnorm2)/rnorm/rnorm;

#pragma omp parallel for
    for(int i=0; i<n; i++){
      p[i] = beta*p[i]+ r[i];
    }
    rnorm=rnorm2;
  }
  itr_out = iter;
  return;
}


#if CCCC==1
//just copy lines for easy 
void all_dump_debug_chunk(){
//linear_solver()a
#if CCCC==1
  printf(" \nn,s= %d %d ",n,s);
  printf(" \nA  = [ "); for(int i=0; i<s  ; printf("%f ", A_values[i++]));
  printf("]\nQ  = [ "); for(int i=0; i<n*s; printf("%f ", Q1[i++])      ); 
  printf("]\nRHS= [ "); for(int i=0; i<n*s; printf("%f ", RHS[i++])     );
#endif

//i_qqax()
#if CCCC == 1
  printf("\ni_qqax X = [ ");
  for(int i=0; i<n; printf("%f ",X[i++]));
  printf("]\ni_qqax va =[ ");
  for(int i=0; i<n; printf("%f ",va[i++]));
  printf("]\ni_qqax ia =[ ");
  for(int i=0; i<n+1; printf("%f ",ia[i++]));
#endif
#if CCCC == 1
  printf("]\ni_qqax Ap = [ ");
  for(int i=0; i<n; printf("%f ",Y[i++]));
#endif
#if CCCC == 1
  printf("]\ni_qqax QAp=[ ");
  for(int i=0; i<s; printf("%f ", QtAX[i++]));
#endif

#if CCCC == 1
  printf("]\ni_qqax I_QQAp = [:");
  for(int i=0; i<n; printf("%f ", Y[i++]));
#endif


}
#endif





