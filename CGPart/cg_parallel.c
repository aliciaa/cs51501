
/*
    This file implements the conjugate gradient method in PETSc as part of
    KSP. You can use this as a starting point for implementing your own
    Krylov method that is not provided with PETSc.

    The following basic routines are required for each Krylov method.
        KSPCreate_XXX()          - Creates the Krylov context
        KSPSetFromOptions_XXX()  - Sets runtime options
        KSPSolve_XXX()           - Runs the Krylov method
        KSPDestroy_XXX()         - Destroys the Krylov context, freeing all
                                   memory it needed
    Here the "_XXX" denotes a particular implementation, in this case
    we use _CG (e.g. KSPCreate_CG, KSPDestroy_CG). These routines are
    are actually called vai the common user interface routines
    KSPSetType(), KSPSetFromOptions(), KSPSolve(), and KSPDestroy() so the
    application code interface remains identical for all preconditioners.

    Other basic routines for the KSP objects include
        KSPSetUp_XXX()
        KSPView_XXX()             - Prints details of solver being used.

    Detailed notes:
    By default, this code implements the CG (Conjugate Gradient) method,
    which is valid for real symmetric (and complex Hermitian) positive
    definite matrices. Note that for the complex Hermitian case, the
    VecDot() arguments within the code MUST remain in the order given
    for correct computation of inner products.

    Reference: Hestenes and Steifel, 1952.

    By switching to the indefinite vector inner product, VecTDot(), the
    same code is used for the complex symmetric case as well.  The user
    must call KSPCGSetType(ksp,KSP_CG_SYMMETRIC) or use the option
    -ksp_cg_type symmetric to invoke this variant for the complex case.
    Note, however, that the complex symmetric code is NOT valid for
    all such matrices ... and thus we don't recommend using this method.
*/
/*
       cgimpl.h defines the simple data structured used to store information
    related to the type of matrix (e.g. complex symmetric) being solved and
    data used during the optional Lanczo process used to compute eigenvalues
*/
#include <../src/ksp/ksp/impls/cg/cgimpl.h>       /*I "petscksp.h" I*/
extern PetscErrorCode KSPComputeExtremeSingularValues_CG(KSP,PetscReal*,PetscReal*);
extern PetscErrorCode KSPComputeEigenvalues_CG(KSP,PetscInt,PetscReal*,PetscReal*,PetscInt*);

int xinVecScalar(PetscScalar alpha, PetscScalar *B, PetscInt M);
int xinMatMult(Mat *pA, Mat *pB);

int CGreadmm(char s[], Mat *pA){
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









/*
     KSPSetUp_CG - Sets up the workspace needed by the CG method.

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_CG"
PetscErrorCode KSPSetUp_CG(KSP ksp)
{
  KSP_CG         *cgP = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       maxit = ksp->max_it,nwork = 3;

  PetscFunctionBegin;
  /* get work vectors needed by CG */
  if (cgP->singlereduction) nwork += 2;
  ierr = KSPSetWorkVecs(ksp,nwork);CHKERRQ(ierr);

  /*
     If user requested computations of eigenvalues then allocate work
     work space needed
  */
  if (ksp->calc_sings) {
    /* get space to store tridiagonal matrix for Lanczos */
    ierr = PetscMalloc4(maxit+1,&cgP->e,maxit+1,&cgP->d,maxit+1,&cgP->ee,maxit+1,&cgP->dd);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)ksp,2*(maxit+1)*(sizeof(PetscScalar)+sizeof(PetscReal)));CHKERRQ(ierr);

    ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_CG;
    ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_CG;
  }
  PetscFunctionReturn(0);
}

/*
       KSPSolve_CG - This routine actually applies the conjugate gradient  method

   This routine is MUCH too messy. I has too many options (norm type and single reduction) embedded making the code confusing and likely to be buggy.

   Input Parameter:
.     ksp - the Krylov space object that was set to use conjugate gradient, by, for
            example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPCG);
*/
#undef __FUNCT__
#define __FUNCT__ "KSPSolve_CG"
PetscErrorCode  KSPSolve_CG(KSP ksp)
{
  
  

  Vec           X_orig,B_orig,R_orig,Z_orig,P_orig,S_orig,W_origi,Lamda;
  Mat           Y;  

  PetscErrorCode ierr;
  PetscBool      diagonalscale;
  KSP_CG         *cg;
  Mat            Amat,Pmat;
  Vec            Dp,Beta,Beta_old,Bc,Bnorm,Alpha,Dpi,Dpiold;
  Vec            d,W,W2;
  Mat            B,X,P,R,Bt,Xt,Pt,Rt; 
  PetscScalar    dp,bnorm,dpidpiold,dpi,total_dpi;
  PetscScalar    a,b,total_beta,total_betaold=1.0,beta,betaold;
  PetscInt       mRow,nCol;//num of rows and cols of rhs;
  PetscInt       i,j,k,stored_max_it;
  //Vec            r;
  PetscInt  ncol;
  PetscInt  *cols;
  PetscScalar *vals;
  PetscPrintf(PETSC_COMM_WORLD,"Entering CG\n");
  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  cg            = (KSP_CG*)ksp->data;
  //eigs          = ksp->calc_sings;
  stored_max_it = ksp->max_it;
  X_orig             = ksp->vec_sol;
  B_orig             = ksp->vec_rhs;
  
  CGreadmm("/homes/cheng172/datStep/Y_step2.mtx", &Y); //path name should not contain ~
  //ierr = PetscPrintf(PETSC_COMM_SELF,"\nThe Matrix in CG , Wocao_Y\n");CHKERRQ(ierr);
  //ierr = MatView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  R_orig            = ksp->work[0];
  Z_orig             = ksp->work[1];
  P_orig             = ksp->work[2];


  if (cg->singlereduction) {
    S_orig = ksp->work[3];
    W_orig= ksp->work[4];
  } else {
    S_orig = 0;                      /* unused */
    W_orig = Z_orig;
  }

#define VecXDot(x,y,a) (((cg->type) == (KSP_CG_HERMITIAN)) ? VecDot(x,y,a) : VecTDot(x,y,a))

  //if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; }
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  ksp->its = 0;

  //PetscViewer   fd;
  //PetscViewerBinaryOpen(PETSC_COMM_WORLD,"B_petsc",FILE_MODE_READ,&fd);
  
  //MatCreate(PETSC_COMM_WORLD,&B);
  //MatSetFromOptions(B);
  //MatLoad(B,fd);
  //ierr = MatView(Amat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  MatTranspose(Y, MAT_INITIAL_MATRIX,&Bt);
  xinMatMult(&Amat,&Bt);
  xinMatMult(&Pmat,&Bt);
  
  //MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&P);
  //MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&R);
  //MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&X);
  
  MatDuplicate(Bt,MAT_DO_NOT_COPY_VALUES,&Pt);
  MatDuplicate(Bt,MAT_DO_NOT_COPY_VALUES,&Rt);
  MatDuplicate(Bt,MAT_DO_NOT_COPY_VALUES,&Xt);
  //MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&AP);
  //MatTranspose(B, MAT_INITIAL_MATRIX,&Bt);
  //MatTranspose(R, MAT_INITIAL_MATRIX,&Rt);
  //MatTranspose(P, MAT_INITIAL_MATRIX,&Pt);
  //MatTranspose(X, MAT_INITIAL_MATRIX,&Xt);
  //MatTranspose(AP,MAT_INITIAL_MATRIX,&APt);
  MatGetSize(Bt,&mRow,&nCol);

  VecCreate(PETSC_COMM_WORLD,&Dp);
  VecSetSizes(Dp,PETSC_DECIDE,mRow);
  VecSetFromOptions(Dp);
  VecDuplicate(Dp,&Beta);
  VecDuplicate(Dp,&Beta_old);
  VecDuplicate(Dp,&Bc);
  VecDuplicate(Dp,&Bnorm);
  VecDuplicate(Dp,&Alpha);
  VecDuplicate(Dp,&Dpi);
  VecDuplicate(Dp,&Dpiold);
  for(j=0;j<mRow;j++){VecSetValue(Dpiold,j,1.0,INSERT_VALUES);}
  VecCreate(PETSC_COMM_WORLD,&d);
  VecSetSizes(d,PETSC_DECIDE,nCol);
  VecSetFromOptions(d);
  //VecDuplicate(W_orig,&d);
  VecDuplicate(d,&W2);
  VecDuplicate(d,&W);
  VecCopy(B_orig,Lamda);//lamda is to save the eigenvalues;
  
  if (!ksp->guess_zero) {
    PetscPrintf(PETSC_COMM_WORLD,"The initial guess in not zero\n");
  } else {
    for(j=0;j<mRow;j++){
      MatGetRow(Bt,j,&ncol,(const PetscInt**)&cols,(const PetscScalar**)&vals);//here is nCol, as we get row from Bt;
      bnorm = 0;
      for(k=0;k<ncol;k++){
              bnorm += vals[cols[k]]*vals[cols[k]];
          }
      VecSetValue(Bnorm,j,sqrt(bnorm),INSERT_VALUES);
      MatRestoreRow(Bt,j,&ncol,(const PetscInt**)&cols,(const PetscScalar**)&vals);//here is nCol, as we get row from Bt;
    }
    VecAssemblyBegin(Bnorm);
    VecAssemblyEnd(Bnorm);
    MatZeroEntries(Rt);
    ierr = MatCopy(Bt,Rt,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);                         /*     r <- b (x is 0) */
    //PetscPrintf(PETSC_COMM_WORLD,"inital residue\n");
    //MatView(Rt,PETSC_VIEWER_STDOUT_WORLD);
    //PetscPrintf(PETSC_COMM_WORLD,"FINISH inital residue\n");

  }
  
  /* xin cheng modified in Apri 20 for cs515 tracemin */
  switch (ksp->normtype) {
  case KSP_NORM_UNPRECONDITIONED:
    for(j=0;j<mRow;j++){
        dp = 0;
        MatGetRow(Rt,j,&ncol,(const PetscInt**)&cols,(const PetscScalar**)&vals);
        for(k=0;k<ncol;k++){
            dp += vals[k]*vals[k];
        }
        VecSetValue(Dp,j,sqrt(dp),INSERT_VALUES);//save this residue norm to DP;
        VecSetValue(Beta,j,dp,INSERT_VALUES);
        MatRestoreRow(Rt,j,&ncol,(const PetscInt**)&cols,(const PetscScalar**)&vals);
    }
    VecAssemblyBegin(Dp);
    VecAssemblyEnd(Dp);
    VecAssemblyBegin(Beta);
    VecAssemblyEnd(Beta);
    //VecView(Beta,PETSC_VIEWER_STDOUT_WORLD);
    break;
  case KSP_NORM_NONE:
    PetscPrintf(PETSC_COMM_WORLD,"Do nothing about the residue norm\n");
    dp = 0.0;
    break;
  default: SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
  }

  VecNorm(Beta,NORM_2,&total_beta);
  i = 0;
  do {
    ksp->its = i+1;
    if (total_beta == 0.0) {
      ksp->reason = KSP_CONVERGED_ATOL;
      ierr        = PetscInfo(ksp,"converged due to beta = 0\n");CHKERRQ(ierr);
      break;
#if !defined(PETSC_USE_COMPLEX)
    } else if ((i > 0) && (total_beta*total_betaold < 0.0)) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr        = PetscInfo(ksp,"diverging due to indefinite preconditioner\n");CHKERRQ(ierr);
      break;
#endif
    }
    if (!i) {
      ierr = MatZeroEntries(Pt);CHKERRQ(ierr);
      ierr = MatCopy(Rt,Pt,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = VecZeroEntries(Bc);CHKERRQ(ierr);
    } else {
      //PetscPrintf(PETSC_COMM_WORLD,"check beta and betaold\n");
      //VecView(Beta,PETSC_VIEWER_STDOUT_WORLD);
      //VecView(Beta_old,PETSC_VIEWER_STDOUT_WORLD);
      //PetscPrintf(PETSC_COMM_WORLD,"finish check beta and betaold\n");
      for(j=0;j<mRow;j++){
        ierr = VecGetValues(Beta,1,&j,&beta);CHKERRQ(ierr);
        ierr = VecGetValues(Beta_old,1,&j,&betaold);CHKERRQ(ierr);
        b = beta/betaold;
        ierr = VecSetValue(Bc,j,b,INSERT_VALUES);CHKERRQ(ierr);
      }
      VecAssemblyBegin(Bc);
      VecAssemblyEnd(Bc);
      //PetscPrintf(PETSC_COMM_WORLD,"the b vector\n");
      //ierr = VecView(Bc,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      //PetscPrintf(PETSC_COMM_WORLD,"finish the b vector\n");
      for(j=0;j<mRow;j++){
        ierr = VecGetValues(Bc,1,&j,&b);CHKERRQ(ierr);
        ierr = MatGetRow(Pt,j,&ncol,(const PetscInt**)&cols,(const PetscScalar**)&vals);CHKERRQ(ierr);
        xinVecScalar(b,vals,ncol);
        ierr = MatRestoreRow(Pt,j,&ncol,(const PetscInt**)&cols,(const PetscScalar**)&vals);CHKERRQ(ierr);
      }
      MatAYPX(Pt,1.0,Rt,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);    /*     p <- z + b* p   */
    }
    VecCopy(Dpi,Dpiold);
    VecZeroEntries(Dpi);
    if (!cg->singlereduction || !i) {
      //MatView(Pt,PETSC_VIEWER_STDOUT_WORLD);
      for(j=0;j<mRow;j++){
        ierr = MatGetRow(Pt,j,&ncol,(const PetscInt**)&cols,(const PetscScalar**)&vals);
        VecZeroEntries(d);
        for(k=0;k<ncol;k++){
          //PetscPrintf(PETSC_COMM_WORLD,"id=%D,val=%g\n",cols[k],vals[k]);
          VecSetValue(d,cols[k],vals[k],INSERT_VALUES);//save this residue norm to DP;
        }
        VecAssemblyBegin(d);
        VecAssemblyEnd(d);
        //VecView(d,PETSC_VIEWER_STDOUT_WORLD);
        ierr = KSP_MatMult(ksp,Amat,d,W);CHKERRQ(ierr);          //    w <- Ap         /
        ierr = KSP_MatMult(ksp,Pmat,d,W2);CHKERRQ(ierr);
        ierr = VecXDot(W,W2,&dpi);CHKERRQ(ierr);     
        ierr = VecSetValue(Dpi,j,dpi,INSERT_VALUES);
        ierr = MatRestoreRow(Pt,j,&ncol,(const PetscInt**)&cols,(const PetscScalar**)&vals);
      }
      VecAssemblyBegin(Dpi);
      VecAssemblyEnd(Dpi);
      //ierr = VecView(Dpi,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    } 
    VecCopy(Beta,Beta_old);
    VecNorm(Beta,NORM_2,&total_beta);
    KSPCheckDot(ksp,total_beta);
    VecNorm(Dpi,NORM_2,&total_dpi);
    VecXDot(Dpi,Dpiold,&dpidpiold);
    
    if ((total_dpi == 0.0) || ((i > 0) && (dpidpiold) <= 0.0)) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_MAT;
      ierr        = PetscInfo(ksp,"diverging due to indefinite or negative definite matrix\n");CHKERRQ(ierr);
      break;
    }
    for(j=0;j<mRow;j++){
        ierr = VecGetValues(Beta,1,&j,&beta);CHKERRQ(ierr);
        ierr = VecGetValues(Dpi,1,&j,&dpi);CHKERRQ(ierr);
        a = beta/dpi;
        ierr = VecSetValue(Alpha,j,a,INSERT_VALUES);CHKERRQ(ierr);
    }
    VecAssemblyBegin(Alpha);
    VecAssemblyEnd(Alpha);
    for(j=0;j<mRow;j++){
        ierr = VecGetValues(Alpha,1,&j,&a);CHKERRQ(ierr);
        ierr = MatGetRow(Pt,j,&ncol,(const PetscInt**)&cols,(const PetscScalar**)&vals);CHKERRQ(ierr);
        ierr = xinVecScalar(a,vals,ncol);CHKERRQ(ierr);
        ierr = MatRestoreRow(Pt,j,&ncol,(const PetscInt**)&cols,(const PetscScalar**)&vals);CHKERRQ(ierr);
    }
    //PetscPrintf(PETSC_COMM_WORLD,"*******************************\n");
    //MatView(Xt,PETSC_VIEWER_STDOUT_WORLD);
    //MatView(Pt,PETSC_VIEWER_STDOUT_WORLD);
    //VecView(Alpha,PETSC_VIEWER_STDOUT_WORLD);   
    MatAYPX(Xt,1.0,Pt,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);          /*     x <- x + ap     */
    //MatView(Xt,PETSC_VIEWER_STDOUT_WORLD);
    PetscPrintf(PETSC_COMM_WORLD,"*******************************\n");
    //MatView(Xt,PETSC_VIEWER_STDOUT_WORLD);
    MatZeroEntries(Rt);
    MatCopy(Xt,Rt,DIFFERENT_NONZERO_PATTERN);
    xinMatMult(&Amat,&Rt);
    xinMatMult(&Pmat,&Rt);
    //MatView(Rt,PETSC_VIEWER_STDOUT_WORLD);
    //MatView(Bt,PETSC_VIEWER_STDOUT_WORLD); 
    
    ierr = MatAYPX(Rt,-1.0,Bt,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_WORLD,"current residue\n");
    //MatView(Rt,PETSC_VIEWER_STDOUT_WORLD);
    if (ksp->normtype == KSP_NORM_UNPRECONDITIONED && ksp->chknorm < i+2) {
        for(j=0;j<mRow;j++){
          dp = 0;
          MatGetRow(Rt,j,&ncol,(const PetscInt**)&cols,(const PetscScalar**)&vals);
          for(k=0;k<ncol;k++){
              dp += vals[k]*vals[k];
          }
        VecSetValue(Dp,j,sqrt(dp),INSERT_VALUES);//save this residue norm to DP;
        VecSetValue(Beta,j,dp,INSERT_VALUES);
        MatRestoreRow(Rt,j,&ncol,(const PetscInt**)&cols,(const PetscScalar**)&vals);
    }
    VecAssemblyBegin(Dp);
    VecAssemblyEnd(Dp);          /*    dp <- r'*r       */
    VecAssemblyBegin(Beta);
    VecAssemblyEnd(Beta);          /*    dp <- r'*r       */
    
    } 
    
    /* xin cheng modified in Apri 20 for cs515 tracemin */
    PetscReal dp_val,a_val,lamda_val;
    for(j=0;j<mRow;j++){
      VecGetValues(Dp,1,&j,&dp_val);
      VecGetValues(Alpha,1,&j,&a_val);
      VecGetValues(Lamda,1,&j,lamda_val);
      if(dp_val*dp_val*a_val<lamda_val){
        PetscPrintf(PETSC_COMM_WORLD,"cg reach smallest m=%D\n",i+1);
        break;}
    }

    VecNorm(Beta,NORM_2,&total_beta);
    KSPLogResidualHistory(ksp,total_beta);CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_WORLD,"total_beta=%g\n",total_beta);
    ierr = KSPMonitor(ksp,i+1,total_beta);CHKERRQ(ierr);
    /*ierr = (*ksp->converged)(ksp,i+1,total_beta,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;
    */
    i++;
    PetscPrintf(PETSC_COMM_WORLD,"FINAL RESULT\n");
    MatView(Xt,PETSC_VIEWER_STDOUT_WORLD);
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  //if (eigs) cg->ned = ksp->its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_CG"
PetscErrorCode KSPDestroy_CG(KSP ksp)
{
  KSP_CG         *cg = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* free space used for singular value calculations */
  if (ksp->calc_sings) {
    ierr = PetscFree4(cg->e,cg->d,cg->ee,cg->dd);CHKERRQ(ierr);
  }
  ierr = KSPDestroyDefault(ksp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGSetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGUseSingleReduction_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     KSPView_CG - Prints information about the current Krylov method being used

      Currently this only prints information to a file (or stdout) about the
      symmetry of the problem. If your Krylov method has special options or
      flags that information should be printed here.

*/
#undef __FUNCT__
#define __FUNCT__ "KSPView_CG"
PetscErrorCode KSPView_CG(KSP ksp,PetscViewer viewer)
{
#if defined(PETSC_USE_COMPLEX)
  KSP_CG         *cg = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  CG or CGNE: variant %s\n",KSPCGTypes[cg->type]);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/*
    KSPSetFromOptions_CG - Checks the options database for options related to the
                           conjugate gradient method.
*/
#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_CG"
PetscErrorCode KSPSetFromOptions_CG(PetscOptions *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CG         *cg = (KSP_CG*)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP CG and CGNE options");CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscOptionsEnum("-ksp_cg_type","Matrix is Hermitian or complex symmetric","KSPCGSetType",KSPCGTypes,(PetscEnum)cg->type,
                          (PetscEnum*)&cg->type,NULL);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsBool("-ksp_cg_single_reduction","Merge inner products into single MPI_Allreduce()","KSPCGUseSingleReduction",cg->singlereduction,&cg->singlereduction,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    KSPCGSetType_CG - This is an option that is SPECIFIC to this particular Krylov method.
                      This routine is registered below in KSPCreate_CG() and called from the
                      routine KSPCGSetType() (see the file cgtype.c).
*/
#undef __FUNCT__
#define __FUNCT__ "KSPCGSetType_CG"
static PetscErrorCode  KSPCGSetType_CG(KSP ksp,KSPCGType type)
{
  KSP_CG *cg = (KSP_CG*)ksp->data;

  PetscFunctionBegin;
  cg->type = type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPCGUseSingleReduction_CG"
static PetscErrorCode  KSPCGUseSingleReduction_CG(KSP ksp,PetscBool flg)
{
  KSP_CG *cg = (KSP_CG*)ksp->data;

  PetscFunctionBegin;
  cg->singlereduction = flg;
  PetscFunctionReturn(0);
}

/*
    KSPCreate_CG - Creates the data structure for the Krylov method CG and sets the
       function pointers for all the routines it needs to call (KSPSolve_CG() etc)

    It must be labeled as PETSC_EXTERN to be dynamically linkable in C++
*/
/*MC
     KSPCG - The preconditioned conjugate gradient (PCG) iterative method

   Options Database Keys:
+   -ksp_cg_type Hermitian - (for complex matrices only) indicates the matrix is Hermitian, see KSPCGSetType()
.   -ksp_cg_type symmetric - (for complex matrices only) indicates the matrix is symmetric
-   -ksp_cg_single_reduction - performs both inner products needed in the algorithm with a single MPI_Allreduce() call, see KSPCGUseSingleReduction()

   Level: beginner

   Notes: The PCG method requires both the matrix and preconditioner to be symmetric positive (or negative) (semi) definite
          Only left preconditioning is supported.

   For complex numbers there are two different CG methods. One for Hermitian symmetric matrices and one for non-Hermitian symmetric matrices. Use
   KSPCGSetType() to indicate which type you are using.

   Developer Notes: KSPSolve_CG() should actually query the matrix to determine if it is Hermitian symmetric or not and NOT require the user to
   indicate it to the KSP object.

   References:
   Methods of Conjugate Gradients for Solving Linear Systems, Magnus R. Hestenes and Eduard Stiefel,
   Journal of Research of the National Bureau of Standards Vol. 49, No. 6, December 1952 Research Paper 2379
   pp. 409--436.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPCGSetType(), KSPCGUseSingleReduction(), KSPPIPECG, KSPGROPPCG

M*/
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_CG"
PETSC_EXTERN PetscErrorCode KSPCreate_CG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CG         *cg;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&cg);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  cg->type = KSP_CG_SYMMETRIC;
#else
  cg->type = KSP_CG_HERMITIAN;
#endif
  ksp->data = (void*)cg;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,2);CHKERRQ(ierr);

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup          = KSPSetUp_CG;
  ksp->ops->solve          = KSPSolve_CG;
  ksp->ops->destroy        = KSPDestroy_CG;
  ksp->ops->view           = KSPView_CG;
  ksp->ops->setfromoptions = KSPSetFromOptions_CG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;

  /*
      Attach the function KSPCGSetType_CG() to this object. The routine
      KSPCGSetType() checks for this attached function and calls it if it finds
      it. (Sort of like a dynamic member function that can be added at run time
  */
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGSetType_C",KSPCGSetType_CG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGUseSingleReduction_C",KSPCGUseSingleReduction_CG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}






























/*int xinVecMult(Vec *a, Vec *b, Vec *c, PetscInt M){
  PetscInt i;
  PetscInt *idx;
  PetscScalar *x, *y, *z;
  PetscMalloc4(M*sizeof(PetscInt), idx, M*sizeof(PetscScalar), x, M*sizeof(PetscScalar), y, M*sizeof(PetscScalar), z);
  for(i=0; i<M; i++){
    idx[i]=i;
  }
  PetscGetValues(*a, M, idx, x);
  PetscGetValues(*b, M, idx, y);
  for(i=0; i<M; i++){
    z[i]=x[i]*y[i];
  }
  PetscSetValues(*c, M, (const PetscInt *)idx, z, INSERT_VALUES);
  PetscFree4(idx,x,y,z);
  return 1;
}*/

int xinVecScalar(PetscScalar alpha, PetscScalar *B, PetscInt M){
  int i;
  for(i=0; i<M; i++){
    B[i]*=alpha;
  }
  return 0;
}

//A should be M by M
//B should be M by N =[r1',r2',...rn']'
//B would be overwrite by [(A*r1)', ... (A*rn)']'
int xinMatMult(Mat *pA, Mat *pB) {
  PetscInt i,M,N,nCols;
  PetscInt *cols, *vals;
  Vec x;
  Vec y;
  
  MatGetSize(*pB, &M, &N);

  VecCreate(PETSC_COMM_WORLD,&x);
  VecSetSizes(x,PETSC_DECIDE,N);
  VecSetFromOptions(x);
  VecCreate(PETSC_COMM_WORLD,&y);
  VecSetSizes(y,PETSC_DECIDE,N);
  VecSetFromOptions(y);

  for(i=0; i<M; i++){
    MatGetRow(*pB, i, &nCols, (const PetscInt**)&cols, (const PetscScalar**)&vals);
    VecZeroEntries(x);
    VecZeroEntries(y);
    //if(nCols==0)
      //continue;
    VecSetValues(x, nCols, (const PetscInt*)cols, (const PetscScalar*)vals,INSERT_VALUES);
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    MatMult(*pA, x, y);
    VecGetValues(y, nCols, (const PetscInt*)cols, (PetscScalar*)vals);
    MatSetValues(*pB, 1, &i, (const PetscInt)nCols, (const PetscInt*)cols, (const PetscScalar*)vals, INSERT_VALUES); 
    MatAssemblyBegin(*pB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (*pB, MAT_FINAL_ASSEMBLY);
  }
  return 1;
}




