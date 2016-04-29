#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
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
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
	PETSC_COMM_WORLD = MPI_COMM_SELF;
	PetscInitialize(&argc, &argv, (char*) 0, help);
        PetscPrintf(PETSC_COMM_SELF, "I'm task %d of %d\n", task_id, num_tasks);
        
        PetscInt m;

		/*---------------------------------------------------------------------------
		 * read in matrices A and B
		 *---------------------------------------------------------------------------*/
		PetscBool dummy_bool;

		//PetscOptionsGetString(NULL, "-fA", fileA, PETSC_MAX_PATH_LEN, NULL);
		PetscOptionsGetString(NULL, NULL, "-fA", fileA, PETSC_MAX_PATH_LEN, &dummy_bool);
		PetscOptionsGetString(NULL, NULL, "-fB", fileB, PETSC_MAX_PATH_LEN, &dummy_bool);
		PetscOptionsGetString(NULL, NULL, "-fO", fileO, PETSC_MAX_PATH_LEN, &dummy_bool);

	char suffix[20] = "_0.res";
	suffix[1] = 48+task_id;
	strcat(fileO, suffix);
	FILE* fp;
	fp = fopen("./intervals.txt", "r");
	int N;
	fscanf(fp, "%d\n", &N);
	PetscPrintf(PETSC_COMM_SELF, "N = %d\n", N);
	assert(N == num_tasks);
	double* intervals = new double[N+1];
	int* num_eigs = new int[N];
	for (int i = 0; i <= N; i++) {
	  fscanf(fp, "%lf\n", &(intervals[i]));
	}
	for (int i = 0; i < N; i++) {
	  fscanf(fp, "%d\n", &(num_eigs[i]));
	}
        PetscPrintf(PETSC_COMM_SELF, "task[%d] working on [%.6lf, %.6lf] with %d eigs, Ofile = %s\n", task_id,  intervals[task_id], intervals[task_id+1], num_eigs[task_id], fileO); 
	fclose(fp);

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
        //A = A - u *B;
        MatAXPY(A, -(intervals[task_id] + intervals[task_id+1])/2, B, DIFFERENT_NONZERO_PATTERN);
	TraceMin1(task_id, fileO, n, num_eigs[task_id], A, B, Y, S);

	MatDestroy(&A);
	MatDestroy(&B);
	MatDestroy(&Y);
	VecDestroy(&S);
	
	/*---------------------------------------------------------------------------
	 * finalize PETSc
	 *---------------------------------------------------------------------------*/
	PetscFinalize();
        MPI_Finalize();
	return 0;
}
