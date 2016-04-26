#include "tracemin_qrcg.h"
#define cxDebug 0
//=====================================
//read matrix market file for debug
//=====================================
int readmm(const char s[], Mat *pA);

//=====================================
//Do CG part of tracemin alg
//=====================================
int tracemin_cg(Mat A,
								Mat X,           /*out:*/
                Mat BY,
                Mat AY,
                PetscInt M,
                PetscInt N 
                ){

  Mat            RHS;                 
  Mat            P;                     // TODO need to be optimized
  Vec            b,x;                   // linear system:  Ax=b
  KSP            ksp;                   // linear solver context 
  PetscInt       i,its;                 // iteration numbers of KSP
  PetscInt      *idxm;                  // reserved for MatGet-VecGet-VecSet-MatSet
  PetscReal     *arr;                   // reserved for Mat and Vec operation
  PetscErrorCode ierr;                  
  
  PetscPrintf(PETSC_COMM_SELF, "source  BY: \n");
  MatView(BY, PETSC_VIEWER_STDOUT_SELF);
  Mat Q1;
  MatConvert(BY, MATSEQAIJ, MAT_INITIAL_MATRIX, &Q1);
  getQ1(Q1, M, N); 
  
  //ierr = MatMatTransposeMult(BY, BY,MAT_INITIAL_MATRIX,  PETSC_DEFAULT ,&P);
  ierr = MatMatTransposeMult(Q1, Q1,MAT_INITIAL_MATRIX,  PETSC_DEFAULT ,&P);
  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(P,-1);
  ierr = MatScale(P,-1);
  
  ierr = MatMatMult(P,AY,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&RHS);CHKERRQ(ierr); 
  ierr = MatAssemblyBegin(RHS,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(RHS,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatZeroEntries(X);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
 
  VecCreate(PETSC_COMM_WORLD,&b);
  VecSetSizes(b,PETSC_DECIDE,M);
  VecSetFromOptions(b);
  
  VecCreate(PETSC_COMM_WORLD,&x);
  VecSetSizes(x,PETSC_DECIDE,M);
  VecSetFromOptions(x);
  
  PetscMalloc1(sizeof(PetscInt)*M, &idxm);
  for(i=0;i<M;i++)
    idxm[i]=i; 

  PetscPrintf(PETSC_COMM_SELF, "before CG  A: \n");
  MatView(A, PETSC_VIEWER_STDOUT_SELF);
  PetscPrintf(PETSC_COMM_SELF, "before CG  P: \n");
  MatView(P, PETSC_VIEWER_STDOUT_SELF);
  PetscPrintf(PETSC_COMM_SELF, "before CG  RHS: \n");
  MatView(RHS, PETSC_VIEWER_STDOUT_SELF);

	KSPCreate(PETSC_COMM_WORLD, &ksp);
	KSPSetType(ksp, KSPCG);
	KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
	KSPSetOperators(ksp, A ,P);
	KSPSetCheckNormIteration(ksp, -1);
	KSPSetTolerances(ksp,1.0e-06,PETSC_DEFAULT,PETSC_DEFAULT,400);// KSPSetTolerances(ksp,(Sig[i]/SigMx)*(Sig[i]/SigMx),PETSC_DEFAULT,PETSC_DEFAULT,400);
	KSPSetInitialGuessNonzero(ksp,PETSC_FALSE);
	KSPSetFromOptions(ksp);

  for(i=0;i<N;i++){
    MatGetColumnVector(RHS,b,i);
    KSPSolve(ksp,b,x);
    KSPGetIterationNumber(ksp,&its);
    VecGetArray(x, &arr);
    MatSetValues(X, M, idxm, 1, &i, arr, INSERT_VALUES);
    //TODO
    MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY);//MatAssemblyBegin(X,MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY);//MatAssemblyEnd(X,MAT_FLUSH_ASSEMBLY);
    VecRestoreArray(x, &arr );
    
    PetscPrintf(PETSC_COMM_SELF, "CG  Col%d x: \n", i);
    VecView(x, PETSC_VIEWER_STDOUT_SELF);
  }
  
  PetscFree(idxm);//, idxn);
  MatDestroy(&P);
  VecDestroy(&b);
  VecDestroy(&x);
  KSPDestroy(&ksp);

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


