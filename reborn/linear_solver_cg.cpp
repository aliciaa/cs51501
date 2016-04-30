
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

  
  
  int its;
  double  *RHSwrt = new double [n*s];   //workspace of p in cg
  double  *RHSCOPY = new double [n*s];   //workspace of p in cg
  double  *WKSPACE1 = new double [n*s];   //workspace of Ap in cg      each col need size== p
  double  *WKSPACE2 = new double [n*s];    //workspace of I-QQ'Ax in cg  each col need size== sizex	  
#pragma omp parallel for
  for(int i=0; i<n*s; i++){
    RHSwrt[i]=RHSCOPY[i] =-RHS[i];
    solution[i]=0.0;
  }



#if CCCC==1
printf("\nn, s = %d %d ",n,s);
printf("\nPLOT A ");
for(int i=0;i<10; i++){
  printf(" %f", A_values[i]);
}
printf("\nPLOT Q ");
for(int i=0;i<n*s; i++){
  printf(" %f", Q1[i]);
}
printf("\nPLOT RHS");
for(int i=0;i<n*s; i++){
  printf(" %f", RHS[i]);
}
printf("\nPLOT RHSwrt");
for(int i=0;i<n*s; i++){
  printf(" %f", RHSwrt[i]);
}


#endif



#pragma omp parallel for
  for(int j=0; j<s; j++){
    cg_core(A_ia, A_ja, A_values, Q1, RHSwrt+n*j, RHSCOPY+n*j, WKSPACE1+n*j, WKSPACE2+n*j, solution+n*j, n, s, its);
  }

#if CCCC==1
printf("\nPLOT Solutions\n");
for(int i=0;i<n*s; i++){
  printf(" %f", solution[i]);
}



#endif





  delete[] RHSCOPY;
  delete[] RHSwrt;
  delete[] WKSPACE1;
  delete[] WKSPACE2;
//  exit(1);
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
#if CCCC == 1
  printf("\nin imqqax rhsX==p : ");
  for(int i=0; i<n; i++){
    printf("%f ", X[i]);
  }
  
  printf("\nin imqqax A : ");
  for(int i=0; i<n; i++){
    printf("%f ", va[i]);
  }

  printf("\nin imqqax ia : ");
  for(int i=0; i<n+1; i++){
    printf("%d ", ia[i]);
  }
  printf("\nin imqqax ja : ");
  for(int i=0; i<5; i++){
    printf("%d ", ja[i]);
  }
#endif

  for(int i=0; i<n; i++){
    Y[i]=0.0;
    for(int jt=ia[i]; jt<ia[i+1]; jt++){
      //printf("Y[%d]+=X[%d]*A[%d] y+=%f*%f   ", i, ja[jt-1]-1, jt-1, X[ja[jt-1]-1], va[jt-1]);  
      Y[i]+=X[ ja[jt-1]-1 ]*va[jt-1];
  	}
    //printf("@@Y[%d]=",Y[i]);
  }
#if CCCC == 1
  printf("\nin imqqax A p:\n ");
  for(int i=0; i<n; i++){
    printf("%f ", Y[i]);
  }
#endif
 
  cblas_dgemv(CblasColMajor,CblasTrans,n,s,1.0,Q, n, Y,  1,  0,QtAX, 1);
#if CCCC == 1
  printf("\nin imqqax QA p:\n ");
  for(int i=0; i<s; i++){
    printf("%f ", QtAX[i]);
  }
#endif

  cblas_dgemv(CblasColMajor, CblasNoTrans, n,s,-1.0,Q,n, QtAX,1, 1,Y, 1);
#if CCCC == 1
  printf("\nin imqqax I_QQAp:\n ");
  for(int i=0; i<n; i++){
    printf("%f ", Y[i]);
  }
#endif

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

#if CCCC == 1
  printf("\nin cg \n");
  printf("\nrhs: ");
  for(int i=0; i<n; i++){
    printf("%f ", rhs[i]);
  }
  printf("\nrhs copy: ");
  for(int i=0; i<n; i++){
    printf("%f ", rhscopy[i]);
  }
  printf("\nsolution initilal:");
  for(int i=0; i<n; i++){
    printf("%f ", solution[i]);
  }
#endif
  
  
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
#if CCCC == 1
  printf("\nbefoe imqqax p==rhs: ");
  for(int i=0; i<n; i++){
    printf("%f ", p[i]);
  }
#endif

  for(iter=1;  iter<=CG_MAX_ITER; iter++) {
      imqqax(A_ia, A_ja, A_v, Q1, p, qqtax, Ap, n,s);
#if CCCC == 1
  printf("\np: ");
  for(int i=0; i<n; i++){
    printf("%f ", p[i]);
  }

  printf("\nAtut p: ");
  for(int i=0; i<n; i++){
    printf("%f ", Ap[i]);
  }
#endif

      pAp   = cblas_ddot(n, p, 1, Ap,1); 
      if(pAp==0){
        fprintf(stderr,"err! pAp=0 in cg!");
        exit(1);
      }
    

      
      alpha = rnorm*rnorm/pAp;
#if CCCC == 1
  printf("\nalpha:%f=%f/%f ",alpha, rnorm, pAp);
  printf("\nxold: ");
  for(int i=0;i<n;i++){
    printf("%f ",x[i]);
  }
  printf("\np");
  for(int i=0;i<n;i++){
    printf("%f ",p[i]);
  }
  printf("\nrold: ");
  for(int i=0; i<n; i++){
    printf("%f ",r[i]);
  }
#endif
        
      cblas_daxpy (n, alpha, p, 1, x, 1);
      cblas_daxpy (n, -alpha,Ap,1, r, 1); 
#if CCCC == 1
  printf("\nxnew: ");
  for(int i=0;i<n;i++){
    printf("%f ",x[i]);
  }
  printf("\nrnew: ");
  for(int i=0; i<n; i++){
    printf("%f ",r[i]);
  }
#endif
       
      if( (rnorm2=norm2(r,n)) < 10e-6) {
        //fprintf(stderr,"\nrnrom<10e-6\n");
        break;
      }
     
      beta = (rnorm2*rnorm2)/rnorm/rnorm;

#if CCCC==1
      printf("\nbeta%f = %f/%f\n", beta,rnorm2, rnorm);
#endif
#pragma omp parallel for
      for(int i=0;i<n; i++){
        p[i] = beta*p[i]+ r[i];
      }
      rnorm=rnorm2;
  }
  
  
#if CCCC==1
  printf("\npnew: ");
  for(int i=0; i<n; i++)
    printf("%f ", p[i]);

  printf("\n");
#endif
  
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
