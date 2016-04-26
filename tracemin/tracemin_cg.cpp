#include "tracemin_cg.h"
#define SIZM 10
#define cxDebug 0


int getQ1(Mat *pA, PetscInt M, PetscInt N){
  int i,j;
  PetscBLASInt MM,NN,K,lda,ldwork,info;
  PetscScalar *tau,*work; 
  PetscInt worksize;
  PetscErrorCode ierr;


  //ierr = MatGetSize(*pA,&m,&n);CHKERRQ(ierr);
  
  PetscBLASIntCast(M,&MM);
  PetscBLASIntCast(N,&NN);
  
  worksize=MM;
  PetscBLASIntCast(worksize,&ldwork);


  PetscMalloc1(MM, &tau);//worksize,&work);
  PetscMalloc1(MM, &work);

  K = NN;     /*full rank*/   
  lda = MM ; //N - row domain   M - col domain
 

  //ierr = PetscPrintf (PETSC_COMM_SELF,"L72\n");CHKERRQ(ierr);
  //ierr = MatView(*pA,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = PetscPrintf (PETSC_COMM_SELF,"L74\n");CHKERRQ(ierr);
  PetscScalar *v;//[4]={0};
  PetscInt *Is; //[4]={0,1,2,3};
  PetscInt nC;// = 4;
  
  PetscReal* arr;
  PetscMalloc1(sizeof(PetscReal)*M*N, &arr);
  
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



int tracemin_cg(  Mat A, \
                  Mat X, \
                  Mat B,\
                  Mat BY_, \
                  Mat AY_, \
                  PetscInt M, \
                  PetscInt N \
                  ) {
  Mat            RHS;                     /* P=QR factorization*/
  Mat            P;                     /* P=QR factorization*/
  Vec            b,x;                   /* Atut*x = b;*/
  KSP            ksp;                   /* linear solver context */
  PetscErrorCode ierr;
  PetscInt       i,its;                 /*iteration numbers of KSP*/
  PetscInt      *idxm;
  PetscReal     *arr;

  
  Mat Y;
  readmm("/homes/cheng172/datStep/Y_step2.mtx",&Y);   
  Mat BY, AY;
  
  MatMatMult(B,Y,MAT_INITIAL_MATRIX,PETSC_DEFAULT, &BY);CHKERRQ(ierr); 
  MatMatMult(A,Y,MAT_INITIAL_MATRIX,PETSC_DEFAULT, &AY);CHKERRQ(ierr); 
  



  PetscPrintf(PETSC_COMM_SELF, "source  BY: \n");
  MatView(BY, PETSC_VIEWER_STDOUT_SELF);
  Mat Q1;
  MatConvert(BY, MATSEQAIJ, MAT_INITIAL_MATRIX, &Q1);
  getQ1(&Q1, M, N);  //BY stores Q1
  
  //ierr = MatMatTransposeMult(BY, BY,MAT_INITIAL_MATRIX,  PETSC_DEFAULT ,&P);
  ierr = MatMatTransposeMult(Q1, Q1,MAT_INITIAL_MATRIX,  PETSC_DEFAULT ,&P);
  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(P,-1);
  ierr = MatScale(P,-1);
  
  //ierr = MatMatMult(A,Y,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&E);CHKERRQ(ierr);
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



  for(i=0;i<N;i++){
    MatGetColumnVector(RHS,b,i);
    KSPCreate(PETSC_COMM_WORLD,&ksp);
    KSPSetOperators(ksp,A,P);
    KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED);
    KSPSetCheckNormIteration(ksp,-1);
    KSPSetTolerances(ksp,1.0e-06,PETSC_DEFAULT,PETSC_DEFAULT,400);// KSPSetTolerances(ksp,(Sig[i]/SigMx)*(Sig[i]/SigMx),PETSC_DEFAULT,PETSC_DEFAULT,400);
    KSPSetInitialGuessNonzero(ksp,PETSC_FALSE);
    KSPSetFromOptions(ksp);
    KSPSolve(ksp,b,x);
    KSPGetIterationNumber(ksp,&its);
    VecGetArray(x, &arr);
    MatSetValues(X,M,(const PetscInt *)idxm,1,(const PetscInt*)&i,(const PetscScalar*)arr, INSERT_VALUES);
    //TODO
    MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY);//MatAssemblyBegin(X,MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY);//MatAssemblyEnd(X,MAT_FLUSH_ASSEMBLY);
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
