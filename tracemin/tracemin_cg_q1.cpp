#include "tracemin_cg_q1.h"
#include "QRFactorization.h"
#include <petscksp.h>

PetscErrorCode ProjectedMatrix::MultVec(Mat PA_shell,
                                        Vec x,
																				Vec y)
{
	ProjectedMatrix *PA;
	PetscInt m, n;
	Vec u, v;

	MatShellGetContext(PA_shell, (void**) &PA);
	MatGetSize(PA->Q1_, &m, &n);
	VecCreate(MPI_COMM_SELF, &u);
	VecSetSizes(u, PETSC_DECIDE, n);
	VecSetFromOptions(u);
	VecDuplicate(x, &v);

	MatMult(PA->A_, x, y);
	MatMultTranspose(PA->Q1_, y, u);
	MatMult(PA->Q1_, u, v);
	VecAXPY(y, -1.0, v);

	VecDestroy(&u);
	VecDestroy(&v);

	return 0;
}

PetscErrorCode ProjectedMatrix::MultMat(Mat PA_shell,
                                        Mat X,
																				Mat Y)
{
	ProjectedMatrix *PA;
	PetscInt m, n, s;
	Mat U, V;

	MatShellGetContext(PA_shell, (void**) &PA);
	MatGetSize(PA->Q1_, &m, &n);
	MatGetSize(X, &m, &s);
	MatCreateDense(MPI_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, n, s, NULL, &U);
	MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &V);

	MatMatMult(PA->A_, X, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y);
	MatTransposeMatMult(PA->Q1_, Y, MAT_REUSE_MATRIX, PETSC_DEFAULT, &U);
	MatMatMult(PA->Q1_, U, MAT_REUSE_MATRIX, PETSC_DEFAULT, &V);
	MatAXPY(Y, -1.0, V, SAME_NONZERO_PATTERN);

	MatDestroy(&U);
	MatDestroy(&V);

	return 0;
}

PetscErrorCode tracemin_cg(Mat A,
                           Mat X,
													 Mat BY,
													 Mat AY,
													 PetscInt M,
													 PetscInt N)
{
  Mat            Q1;										// projection matrix
	Mat            PA_shell;							// matrix-free operator
  Mat            RHS;                  	// right-hand-side matrix
  Vec            b, x;                  // vectors for CG
  KSP            ksp;                   // CG iterative solver
	PC             pc;										// preconditioner
  PetscErrorCode ierr;
  PetscInt       i,its;                 /*iteration numbers of KSP*/
  PetscInt      *idxm;
  PetscReal     *arr;
  
  MatConvert(BY, MATSEQAIJ, MAT_INITIAL_MATRIX, &Q1);
  getQ1(Q1, M, N);

	ProjectedMatrix PA(A, Q1);
	MatCreateShell(PETSC_COMM_WORLD, M, M, PETSC_DETERMINE, PETSC_DETERMINE, &PA, &PA_shell);
	MatShellSetOperation(PA_shell, MATOP_MULT, (void(*)(void))ProjectedMatrix::MultVec);
  
	MatDuplicate(AY, MAT_DO_NOT_COPY_VALUES, &RHS);
	ProjectedMatrix::MultMat(PA_shell, AY, RHS);

  MatZeroEntries(X);
 
  VecCreate(PETSC_COMM_WORLD,&b);
  VecSetSizes(b,PETSC_DECIDE,M);
  VecSetFromOptions(b);
  
  VecCreate(PETSC_COMM_WORLD,&x);
  VecSetSizes(x,PETSC_DECIDE,M);
  VecSetFromOptions(x);
  
  PetscMalloc1(M * sizeof(PetscInt), &idxm);
  for (PetscInt i = 0; i < M; ++i) {
    idxm[i] = i;
	}

  PetscPrintf(PETSC_COMM_SELF, "before CG  A: \n");
  MatView(A, PETSC_VIEWER_STDOUT_SELF);
  PetscPrintf(PETSC_COMM_SELF, "before CG  P: \n");
  MatView(P, PETSC_VIEWER_STDOUT_SELF);
  PetscPrintf(PETSC_COMM_SELF, "before CG  RHS: \n");
  MatView(RHS, PETSC_VIEWER_STDOUT_SELF);

	KSPCreate(PETSC_COMM_WORLD, &ksp);
	KSPGetPC(ksp, &pc);
	PCSetType(pc, PCNONE);
	KSPSetType(ksp, KSPCG);
	KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
	KSPSetOperators(ksp, PA_shell, A);
	KSPSetCheckNormIteration(ksp, -1);
	KSPSetTolerances(ksp, 1.0e-06, PETSC_DEFAULT, PETSC_DEFAULT, 400);
	KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
	KSPSetFromOptions(ksp);

  for(i=0;i<N;i++){
    MatGetColumnVector(RHS,b,i);
    KSPSolve(ksp,b,x);
    KSPGetIterationNumber(ksp,&its);
    VecGetArray(x, &arr);
    MatSetValues(X, M, idxm, 1, &i, arr, INSERT_VALUES);
    MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY);
    VecRestoreArray(x, &arr );
    
    PetscPrintf(PETSC_COMM_SELF, "CG  Col%d x: \n", i);
    VecView(x, PETSC_VIEWER_STDOUT_SELF);
  }
  
  PetscFree(idxm);
	MatDestroy(&PA_shell);
	MatDestroy(&Q1);
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

  /* Create and asseble matrix */
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

