
#include <petscconf.h>
#if defined(PETSC_HAVE_MATHIMF_H)
#include <mathimf.h>           /* this needs to be included before math.h */
#endif

#include <petscdt.h>            /*I "petscdt.h" I*/
#include <petscblaslapack.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/dtimpl.h>
#include <petscviewer.h>
#include <petscdmplex.h>
#include <petscdmshell.h>


//#include <petscdt.h>            /*I "petscdt.h" I*/
//#include <petscblaslapack.h>
//#include <petsc-private/petscimpl.h>
//#include <petscviewer.h>


static char help[] = "Solves a tridiagonal linear system.\n\n";


/*T
   Concepts: KSP^basic parallel example;
   Processors: n
T*/

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners

  Note:  The corresponding uniprocessor example is ex1.c
*/
#include <petscksp.h>
#include <petscmat.h>

//#undef __FUNCT__
//#define __FUNCT__ "main"

#define SIZM 10
#define cxDebug 0


int getQ1(Mat *pA, PetscInt M, PetscInt N){
  int i,j;
  PetscBLASInt K,lda,ldwork,info;
  PetscScalar *tau,*work; 
  PetscInt worksize;
  PetscErrorCode ierr;


  //ierr = MatGetSize(*pA,&m,&n);CHKERRQ(ierr);
  //PetscBLASIntCast(m,&M);
  //PetscBLASIntCast(n,&N);
  
  worksize=M;
  PetscBLASIntCast(worksize,&ldwork);


  PetscMalloc1(M, &tau);//worksize,&work);
  PetscMalloc1(M, &work);

  K = N;     /*full rank*/   
  lda = M ; //N - row domain   M - col domain
 

  //ierr = PetscPrintf (PETSC_COMM_SELF,"L72\n");CHKERRQ(ierr);
  //ierr = MatView(*pA,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = PetscPrintf (PETSC_COMM_SELF,"L74\n");CHKERRQ(ierr);
  PetscScalar *v;//[4]={0};
  PetscInt *Is; //[4]={0,1,2,3};
  PetscInt nC;// = 4;
  
  PetscReal* arr;
  PetscMalloc(size_of(PetscReal)*M*N, arr);
  
  for(i=0; i<10; i++){
    ierr = MatGetRow(*pA,i,&nC,(const PetscInt **)&Is,(const PetscScalar **)&v); CHKERRQ(ierr);
    for(j=0; j<nC; j++){
      arr[i+10*Is[j]]=v[j];
    }
    ierr = MatRestoreRow(*pA,i,&nC,(const PetscInt**)&Is,(const PetscScalar**)&v); CHKERRQ(ierr);
  }


  /* Do QR */
  PetscFPTrapPush(PETSC_FP_TRAP_OFF);
  LAPACKgeqrf_(&M,&N,arr,&lda,tau,work,&ldwork,&info);
  PetscFPTrapPop();
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xGEQRF error");

  /*Extract an explicit representation of Q */
  LAPACKungqr_(&M,&N,&K,arr,&lda,tau,work,&ldwork,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xORGQR/xUNGQR error");
   
  for (i=0; i<10; i++) {
    for(j=0; j<4; j++) {
      ierr = MatSetValues(*pA,1,&i,1,&j,&arr[i+j*10],INSERT_VALUES);CHKERRQ(ierr);
   }
  }


  ierr = MatAssemblyBegin(*pA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*pA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = PetscFree(arr);CHKERRQ(ierr);
  return 0; 
}



int readmm(char s[], Mat *pA){
  FILE        *file;
  int         *Is,*Js;
  PetscScalar *Vs;
  PetscInt    m,n,nnz,i;
  PetscErrorCode ierr;

  ierr = PetscFOpen(PETSC_COMM_SELF,s,"r",&file);CHKERRQ(ierr);
  char buf[100];
  /* process header with comments */
  do fgets(buf,PETSC_MAX_PATH_LEN-1,file);
  while (buf[0] == '%');

  sscanf(buf,"%d %d %d\n",&m,&n,&nnz);
  //ierr = PetscPrintf (PETSC_COMM_SELF,"m = %d, n = %d, nnz = %d\n",m,n,nnz);

  /* reseve memory for matrices */
  ierr = PetscMalloc3(nnz,&Is, nnz,&Js, nnz,&Vs); CHKERRQ(ierr);

  for (i=0; i<nnz; i++) {
    ierr = fscanf(file,"%d %d %le\n",&Is[i],&Js[i],(double*)&Vs[i]);
    //ierr = PetscPrintf(PETSC_COMM_WORLD,"%d,%d,%le\n",Is[i],Js[i],Vs[i]);CHKERRQ(ierr);
    if (ierr == EOF) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"i=%d, reach EOF\n",i);
    Is[i]--; Js[i]--;    /* adjust from 1-based to 0-based */
  }
  fclose(file);
  //ierr = PetscPrintf(PETSC_COMM_SELF,"Read file completes.\n");CHKERRQ(ierr);

  /* Creat and asseble matrix */
  ierr = MatCreate(PETSC_COMM_SELF,pA);CHKERRQ(ierr);
  ierr = MatSetType(*pA, /*MATDENSE*/ MATSEQAIJ );CHKERRQ(ierr);
  ierr = MatSetSizes(*pA,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*pA);CHKERRQ(ierr);
  ierr = MatSetUp(*pA);CHKERRQ(ierr);

  for (i=0; i<nnz; i++) {
    ierr = MatSetValues(*pA,1,&Is[i],1,&Js[i],&Vs[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  
  ierr = MatAssemblyBegin(*pA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*pA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = MatView(*pA,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscFree3(Is,Js,Vs);CHKERRQ(ierr);
  return 0;
}



int tracemin_cg(  Mat *pA, \
                  Mat* pB, \
                  Mat *pY, \
                  Mat *pX, \
                  Mat *pBY, \
                  Mat *pAY, \
                  Vec *Sig, \
                  PetscInt M, \
                  PetscInt N,
                  ) {
  Mat            P;                     /* P=QR factorization*/
  Mat            Atut;  
  Vec            b,x;                   /* Atut*x = b;*/
  KSP            ksp;                   /* linear solver context */
  PetscErrorCode ierr;
  PetscInt       i,its;                 /*iteration numbers of KSP*/
  Petsc
  
  getQ1(pBY, M, N);  //BY stores Q1
  
  ierr = MatMatTransposeMult(*BY, *BY,MAT_INITIAL_MATRIX,  PETSC_DEFAULT ,&P);
  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(P,-1);
  ierr = MatScale(P,-1);
  
  //ierr = MatMatMult(A,Y,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&E);CHKERRQ(ierr);
  ierr = MatMatMult(P,*pAY,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Atut);CHKERRQ(ierr); 
  ierr = MatAssemblyBegin(Atut,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Atut,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatZeroEntries(*pX);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*pX,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*pX,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
 
  VecCreate(PETSC_COMM_WORLD,&b);
  VecSetSizes(b,PETSC_DECIDE,m);
  VecSetFromOptions(b);
  
  VecCreate(PETSC_COMM_WORLD,&x);
  VecSetSizes(x,PETSC_DECIDE,m);
  VecSetFromOptions(x);

  
  for(i=0;i<N;i++){
    MatGetColumnVector(E,b,i);
    KSPCreate(PETSC_COMM_WORLD,&ksp);
    KSPSetOperators(ksp,A,P);
    KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED);
    KSPSetCheckNormIteration(ksp,-1);
    KSPSetTolerances(ksp,1.0e-06,PETSC_DEFAULT,PETSC_DEFAULT,400);
    KSPSetInitialGuessNonzero(ksp,PETSC_FALSE);
    KSPSetFromOptions(ksp);
    KSPSolve(ksp,b,x);
    KSPGetIterationNumber(ksp,&its);
    
  }
  return 0;
  
}
