#include "tracemin_qrcg.h"

//QR fact A, overwrite A with Q1
int getQ1(Mat A, PetscInt M, PetscInt N){
  int i,j;
  PetscBLASInt   MM,NN,K,lda,ldwork,info;
  PetscScalar   *tau,*work; 
  PetscInt       worksize;
  PetscErrorCode ierr;

  //ierr = MatGetSize(*pA,&M,&N);CHKERRQ(ierr);
  PetscBLASIntCast(M,&MM);
  PetscBLASIntCast(N,&NN);
  
  worksize=MM;
  PetscBLASIntCast(worksize,&ldwork);
  PetscMalloc1(MM, &tau);//worksize,&work);
  PetscMalloc1(MM, &work);
  K = NN;        /*full rank*/   
  lda = MM ;     //col domain

  //PETSc does not have QR, LAPACK does.  Prepare the memory for LAPACK
  PetscScalar *v;
  PetscInt *Is; 
  PetscInt nC;
  PetscReal* arr;
  PetscMalloc1(sizeof(PetscReal)*M*N, &arr);
  for(i=0; i<M*N; i++)
    arr[i]=0.0;
  for(i=0; i<M; i++){
    ierr = MatGetRow(A,i,&nC,(const PetscInt **)&Is,(const PetscScalar **)&v); CHKERRQ(ierr);
    if((&nC)!=NULL)
      for(j=0; j<nC; j++)
        arr[i+M*Is[j]]=v[j];
    ierr = MatRestoreRow(A,i,&nC,(const PetscInt**)&Is,(const PetscScalar**)&v); CHKERRQ(ierr);
  }


  // Do QR and Extract an explicit representation of Q
  PetscFPTrapPush(PETSC_FP_TRAP_OFF);
  LAPACKgeqrf_(&M,&N,arr,&lda,tau,work,&ldwork,&info);
  PetscFPTrapPop();
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xGEQRF error");
  LAPACKungqr_(&M,&N,&K,arr,&lda,tau,work,&ldwork,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xORGQR/xUNGQR error");
  

  //GO back from LAPACK to PETSc.    TODO: needed optimization
  for (i=0; i<M; i++) {
    for(j=0; j<N; j++) {
      ierr = MatSetValues(A,1,&i,1,&j,&arr[i+j*M],INSERT_VALUES);CHKERRQ(ierr);
   }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(  A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(arr);CHKERRQ(ierr);
  return 0; 
}

