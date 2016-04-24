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
			 fileB[PETSC_MAX_PATH_LEN];		// file of matrix B
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
		PetscOptionsGetString(NULL, "-fA", fileA, PETSC_MAX_PATH_LEN, NULL);
		PetscOptionsGetString(NULL, "-fB", fileB, PETSC_MAX_PATH_LEN, NULL);
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
	
	TraceMin1(n, p, A, B, Y, S);

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
