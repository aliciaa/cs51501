#include "QRFactorization.h"
#include <petscdt.h>
#include <petscblaslapack.h>

//QR fact A, overwrite A with Q1
PetscErrorCode QRFactorizationQ1(Mat A)
{
  PetscInt       m, n;
  PetscScalar    *a, *tau, *work;
  PetscBLASInt   info;

  MatGetSize(A, &m, &n);

  PetscMalloc1(m, &tau);
  PetscMalloc1(m, &work);

  MatDenseGetArray(A, &a);

  // Do QR and extract an explicit representation of Q1
  PetscFPTrapPush(PETSC_FP_TRAP_OFF);
  LAPACKgeqrf_(&m, &n, a, &m, tau, work, &m, &info);
  PetscFPTrapPop();
  if (info) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "xGEQRF error");
  LAPACKungqr_(&m, &n, &n, a, &m, tau, work, &m, &info);
  if (info) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "xORGQR/xUNGQR error");
  
  MatDenseRestoreArray(A, &a);
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  
  PetscFree(tau);
  PetscFree(work);

  return 0; 
}

