
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

#undef __FUNCT__
#define __FUNCT__ "main"

#define SIZM 10
#define cxDebug 0


int getQ1(Mat *pA){
  int i,j;
  PetscInt m,n;
  PetscBLASInt M,N,K,lda,ldwork,info;
  PetscScalar *tau,*work; 
  PetscInt worksize;
  PetscErrorCode ierr;


  ierr = MatGetSize(*pA,&m,&n);CHKERRQ(ierr);
  PetscBLASIntCast(m,&M);
  PetscBLASIntCast(n,&N);
  
  worksize=m;
  PetscBLASIntCast(worksize,&ldwork);


  PetscMalloc1(m, &tau);//worksize,&work);
  PetscMalloc1(worksize,&work);

  K = N;     /*full rank*/   
  lda = M ; //N - row domain   M - col domain
 

  //ierr = PetscPrintf (PETSC_COMM_SELF,"L72\n");CHKERRQ(ierr);
  //ierr = MatView(*pA,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = PetscPrintf (PETSC_COMM_SELF,"L74\n");CHKERRQ(ierr);
  PetscScalar *v;//[4]={0};
  PetscInt *Is; //[4]={0,1,2,3};
  PetscInt nC;// = 4;
  
  PetscReal arr[10*4];
  
  for(i=0; i<10; i++){
    ierr = MatGetRow(*pA,i,&nC,(const PetscInt **)&Is,(const PetscScalar **)&v); CHKERRQ(ierr);
    //ierr = PetscPrintf (PETSC_COMM_SELF,"get row %d %d\n", i, nC);CHKERRQ(ierr);
    for(j=0; j<nC; j++){
      arr[i+10*Is[j]]=v[j];
    }
    ierr = MatRestoreRow(*pA,i,&nC,(const PetscInt**)&Is,(const PetscScalar**)&v); CHKERRQ(ierr);
  }


  //ierr = PetscPrintf (PETSC_COMM_SELF,"new L72\n");CHKERRQ(ierr);
  //ierr = MatView(*pA,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = PetscPrintf (PETSC_COMM_SELF,"new L74\n");CHKERRQ(ierr);
  //for(i=0;i<40; i++){
  //  ierr = PetscPrintf (PETSC_COMM_SELF,"arr[%d] = %f\n",i,arr[i]);CHKERRQ(ierr);
  //}

  /* Do QR */
  PetscFPTrapPush(PETSC_FP_TRAP_OFF);
  LAPACKgeqrf_(&M,&N,arr,&lda,tau,work,&ldwork,&info);
  PetscFPTrapPop();
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xGEQRF error");

  /*Extract an explicit representation of Q */
  //PetscMemcpy(Q,A,mstride*n*sizeof(PetscScalar));
  LAPACKungqr_(&M,&N,&K,arr,&lda,tau,work,&ldwork,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xORGQR/xUNGQR error");


  //ierr = PetscPrintf (PETSC_COMM_SELF,"\nWe store Q1 in arr:\n");CHKERRQ(ierr);
  //for(i=0;i<40; i++){
  //  ierr = PetscPrintf (PETSC_COMM_SELF,"arr[%d] = %f\n",i,arr[i]);CHKERRQ(ierr);
  //}
  
   
  for (i=0; i<10; i++) {
    for(j=0; j<4; j++) {
      ierr = MatSetValues(*pA,1,&i,1,&j,&arr[i+j*10],INSERT_VALUES);CHKERRQ(ierr);
   }
  }


  ierr = MatAssemblyBegin(*pA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*pA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = PetscPrintf (PETSC_COMM_SELF,"\nThe Q1 we are going to return is\n");CHKERRQ(ierr);
  //ierr = MatView(*pA,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  //ierr = PetscFree(arr);CHKERRQ(ierr);
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



int main(int argc,char **args)
{
  Mat            A,B,Y;                 /*input : the three imput matrix */
  Mat            P;                     /* P=QR factorization*/
  PetscViewer    fd1,fd2;
  char           file[4][PETSC_MAX_PATH_LEN];
  //Mat            C;                     /*C=B*Y*/
  PetscInt       m,n;                   /*size of matrix C;*/
  Mat            E,X;                 /*lhs and rhs of the linear equations;*/
  Vec            b,x;                   /*column vector of rhs;*/
  KSP            ksp;                   /* linear solver context */
  PetscErrorCode ierr;
  PetscInt       i,its;                 /*iteration numbers of KSP*/
  PetscBool      flg;

  m=10;
  n=4;

  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = PetscOptionsGetString(NULL,"-fa",file[0],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if(!flg){SETERRQ(PETSC_COMM_WORLD,1,"Please provide matrix A!\n");}
  ierr = PetscOptionsGetString(NULL,"-fb",file[1],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if(!flg){SETERRQ(PETSC_COMM_WORLD,1,"Please provide matrix B!\n");}
  ierr = PetscOptionsGetString(NULL,"-fy",file[2],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if(!flg){SETERRQ(PETSC_COMM_WORLD,1,"Please provide matrix Y!\n");}
  ierr = PetscOptionsGetString(NULL,"-fp",file[3],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if(!flg){SETERRQ(PETSC_COMM_WORLD,1,"Please provide matrix P!\n");}


  /*Load the matrices*/
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[0],FILE_MODE_READ,&fd1);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[1],FILE_MODE_READ,&fd2);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);

  ierr = MatLoad(A,fd1);CHKERRQ(ierr);
  ierr = MatLoad(B,fd2);CHKERRQ(ierr);
  readmm(file[2], &Y);
  readmm(file[3], &P);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Read file completes.\n");CHKERRQ(ierr);
#if(cxDEBUG==1)
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
  ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
#endif

  Mat Q1;
  ierr = MatCreate(PETSC_COMM_WORLD, &Q1); CHKERRQ(ierr);
  ierr = MatSetFromOptions(Q1); CHKERRQ(ierr);
  ierr = MatMatMult(B,Y,MAT_INITIAL_MATRIX,PETSC_DEFAULT, &Q1);CHKERRQ(ierr); 
#if(cxDEBUG==1)
  ierr = PetscPrintf(PETSC_COMM_SELF,"Cao this is BY.\n");CHKERRQ(ierr);
  ierr = MatView(Q1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif

  getQ1(&Q1);
 
#if(cxDEBUG==1)
  ierr = PetscPrintf(PETSC_COMM_SELF,"\nTHIS IS Q1.\n");CHKERRQ(ierr);
  ierr = MatView(Q1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
  
  Mat SBP;
  ierr = MatMatTransposeMult(Q1,Q1,MAT_INITIAL_MATRIX,  PETSC_DEFAULT ,&SBP);
  ierr = MatAssemblyBegin(SBP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(SBP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(SBP,-1);
  ierr = MatScale(SBP,-1);
  
#if(cxDEBUG==1)  
  ierr = PetscPrintf(PETSC_COMM_SELF,"\nTHIS IS SBP = Q1*Q1'.\n");CHKERRQ(ierr);
  ierr = MatView(SBP,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\nBelows is act_P.\n");CHKERRQ(ierr);
  ierr = MatView(P  ,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif

  ierr = MatMatMult(A,Y,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&E);CHKERRQ(ierr);
  ierr = MatMatMult(SBP,E,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&E);CHKERRQ(ierr);
  
#if(cxDEBUG==1)
  ierr = PetscPrintf(PETSC_COMM_SELF,"\nBelows is PAY.\n");CHKERRQ(ierr);
  ierr = MatView(E,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
  ierr = MatAssemblyBegin(E,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(E,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


  //ierr = MatGetSize(E,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
  ierr = MatSetSizes(X,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(X);CHKERRQ(ierr);
  ierr = MatSetUp(X); CHKERRQ(ierr);
  ierr = MatZeroEntries(X);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

 // PetscInt ix[m];
 // for(i=0;i<m;i++){
 //   ix[i] = i;
 // }
  
  VecCreate(PETSC_COMM_WORLD,&b);
  VecSetSizes(b,PETSC_DECIDE,m);
  VecSetFromOptions(b);
  
  VecCreate(PETSC_COMM_WORLD,&x);
  VecSetSizes(x,PETSC_DECIDE,m);
  VecSetFromOptions(x);

//#if(cxDEBUG==1)
  ierr = PetscPrintf(PETSC_COMM_SELF,"\nBEFORE CG Check A\n");CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\nBEFORE CG Check SBP\n");CHKERRQ(ierr);
  ierr = MatView(SBP,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\nBEFORE CG Check E\n");CHKERRQ(ierr);
  ierr = MatView(E,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
//#endif
  
  for(i=0;i<n;i++){
    MatGetColumnVector(E,b,i);
    PetscPrintf(PETSC_COMM_WORLD,"the rhs\n");
    VecView(b,PETSC_VIEWER_STDOUT_WORLD);
    KSPCreate(PETSC_COMM_WORLD,&ksp);
    KSPSetOperators(ksp,A,SBP);
    KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED);
    KSPSetCheckNormIteration(ksp,-1);
    KSPSetTolerances(ksp,1.0e-06,PETSC_DEFAULT,PETSC_DEFAULT,400);
    KSPSetInitialGuessNonzero(ksp,PETSC_FALSE);
    KSPSetFromOptions(ksp);
    KSPSolve(ksp,b,x);
    KSPGetIterationNumber(ksp,&its);
    PetscPrintf(PETSC_COMM_WORLD,"%d-th equation,iteration=%D\n",i,its);
    VecView(x,PETSC_VIEWER_STDOUT_WORLD);
  }
  return 0;
  
}
