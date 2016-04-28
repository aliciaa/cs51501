#include "TraceMin.h"
#include "matio.h"

static const char help[] = "Trace Minimization Algorithm\n";
static const int MASTER = 0;

int main(int argc, char *argv[])
{
	unsigned char error = 0;
	int num_tasks,										// number of tasks in partition
			task_id;											// task identifier
	char fileA[PETSC_MAX_PATH_LEN],		// file of matrix A
             fileB[PETSC_MAX_PATH_LEN],		// file of matrix B
	     fileO[PETSC_MAX_PATH_LEN];
	Mat A,														// the matrix A
			B,														// the matrix B
			Y;
	Vec S;														// the eigenvalues
	PetscInt n,												// dimension of matrix
					 p = 5;										// number of eigenvalues needed

	/*---------------------------------------------------------------------------
	 * initialize PETSc
	 *---------------------------------------------------------------------------*/
	PetscInitialize(&argc, &argv, (char*) 0, help);

	MPI_Comm_rank(PETSC_COMM_WORLD, &task_id);
	MPI_Comm_size(PETSC_COMM_WORLD, &num_tasks);

	if (task_id == MASTER){ // initilization: A and b
		PetscInt m;

		/*---------------------------------------------------------------------------
		 * read in matrices A and B
		 *---------------------------------------------------------------------------*/
		PetscBool dummy_bool;
    //Petsc 3.6.3
		PetscOptionsGetString(NULL, "-fA", fileA, PETSC_MAX_PATH_LEN, NULL);
		PetscOptionsGetString(NULL, "-fB", fileB, PETSC_MAX_PATH_LEN, NULL);
		PetscOptionsGetString(NULL, "-fO", fileO, PETSC_MAX_PATH_LEN, NULL);
		//Petsc 3.6.4
    //PetscOptionsGetString(NULL, NULL, "-fA", fileA, PETSC_MAX_PATH_LEN, &dummy_bool);
		//PetscOptionsGetString(NULL, NULL, "-fB", fileB, PETSC_MAX_PATH_LEN, &dummy_bool);
		//PetscOptionsGetString(NULL, NULL, "-fO", fileO, PETSC_MAX_PATH_LEN, &dummy_bool);


		error = MatRead(fileA, n, A);
		error = MatRead(fileB, m, B);

		if (n != m) {
			PetscPrintf(PETSC_COMM_SELF, "Size of A must be equalt to size of B\n");
			MatDestroy(&A);
			MatDestroy(&B);
			return 1;
		}

//		MatView(A, PETSC_VIEWER_STDOUT_SELF);
//		MatView(B, PETSC_VIEWER_STDOUT_SELF);
	}
        //A = A - u *B;
        MatAXPY(A, 1.0, B, DIFFERENT_NONZERO_PATTERN);
	TraceMin1(fileO, n, p, A, B, Y, S);

	MatDestroy(&A);
	MatDestroy(&B);
	MatDestroy(&Y);
	VecDestroy(&S);
	
	/*---------------------------------------------------------------------------
	 * finalize PETSc
	 *---------------------------------------------------------------------------*/
	PetscFinalize();

	return 0;
}
