#include "tracemin.h"
#include "matio.h"
#include "view.h"
#include <cstring>
#include <cstdlib>
#include <mpi.h>

static const int MASTER       = 0,
                 MAX_PATH_LEN = 256;

int main(int argc, char *argv[])
{
  unsigned char error = 0;
  int num_tasks,										// number of tasks in partition
      task_id;											// task identifier
  std::string fileA,	              // file of matrix A
              fileB,	              // file of matrix B
              fileO;                // output file 
  MKL_INT n,												// dimension of matrix
          p;   										  // number of eigenvalues needed
  MKL_INT *AI = NULL, 							// row pointers of the matrix A
          *AJ = NULL,								// column indices of the matrix A
          *BI = NULL,               // row pointers of the matrix B
          *BJ = NULL;               // column indices of the matrix B
  double  *AV = NULL,	              // values of the matrix A
          *BV = NULL,               // values of the matrix B
          *Y  = NULL,               // eigenvectors
          *S  = NULL;               // eigenvalues

  /*---------------------------------------------------------------------------
   * initialize MPI
   *---------------------------------------------------------------------------*/
	MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

  if (task_id == MASTER) { // initilization: A and b
    MKL_INT m;

    /*---------------------------------------------------------------------------
     * read in the command line options
     *---------------------------------------------------------------------------*/
    for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "-p")) p = atoi(argv[++i]);
      if (!strcmp(argv[i], "-fA")) fileA = argv[++i];
      if (!strcmp(argv[i], "-fB")) fileB = argv[++i];
      if (!strcmp(argv[i], "-fC")) fileO = argv[++i];
    }

    /*---------------------------------------------------------------------------
     * read in matrices A and B
     *---------------------------------------------------------------------------*/
    error = readSymmSparseMatrix(fileA, n, AI, AJ, AV);
    error = readSymmSparseMatrix(fileB, m, BI, BJ, BV);

    if (n != m) {
      printf("Size of A must be equal to size of B\n");
      error = 1;
    }
  }
#if 0
  ViewCSR("A", n, AI, AJ, AV);
  ViewCSR("B", n, BI, BJ, BV);
#endif

#if 1
  if (error == 0) {
    TraceMin1(n, p, AI, AJ, AV, BI, BJ, BV, Y, S);
  }
#endif

	MPI_Finalize();
  
  if (AI != NULL) delete [] AI;
  if (AJ != NULL) delete [] AJ;
  if (AV != NULL) delete [] AV;
  if (BI != NULL) delete [] BI;
  if (BJ != NULL) delete [] BJ;
  if (BV != NULL) delete [] BV;
  if (Y  != NULL) delete [] Y;
  if (S  != NULL) delete [] S;

  return 0;
}
