#include "tracemin.h"
#include "matio.h"
#include "view.h"
#include <cstring>
#include <cstdlib>
#include <mpi.h>
#include <cassert>

static const int MASTER       = 0,
                 MAX_PATH_LEN = 256;

void a_plus_mu_b(MKL_INT n,
                 MKL_INT* AI, MKL_INT* AJ, double* AV,
		 MKL_INT* BI, MKL_INT* BJ, double* BV,
		 double mu,
		 MKL_INT** CI, MKL_INT** CJ, double** CV) {
  int p,q;
  int total_nnz = 0;
  for (int i = 0; i < n; i++) {
    p = AI[i] - 1;
    q = BI[i] - 1;
    while (p < AI[i+1] - 1 && q < BI[i+1] - 1) {
      if (AJ[p] < BJ[q]) {
        total_nnz++;
        p++;
      } else if (AJ[p] > BJ[q]) {
        total_nnz++;
        q++;
      } else {
        total_nnz++;
        p++;q++;
      }
    }
    if (p < AI[i+1] - 1) {
      while (p < AI[i+1] - 1) {total_nnz++; p++;}
    }
    if (q < BI[i+1]) {
      while (q < BI[i+1] - 1) {total_nnz++; q++;}
    }
  }
  (*CI) = new MKL_INT[n+1];
  (*CJ) = new MKL_INT[total_nnz];
  (*CV) = new double[total_nnz];
  int curr_nnz = 0;
  for (int i = 0; i < n; i++) {
    (*CI)[i] = curr_nnz + 1;
    p = AI[i] - 1;
    q = BI[i] - 1;
    while (p < AI[i+1] - 1 && q < BI[i+1] - 1) {
      if (AJ[p] < BJ[q]) {
        (*CJ)[curr_nnz] = AJ[p];
        (*CV)[curr_nnz] = AV[p];
        curr_nnz++;
        p++;
      } else if (AJ[p] > BJ[q]) {
        (*CJ)[curr_nnz] = BJ[q];
        (*CV)[curr_nnz] = mu * BV[q];
        curr_nnz++;
        q++;
      } else {
        (*CJ)[curr_nnz] = AJ[p];
        (*CV)[curr_nnz] = AV[p] + mu * BV[q];
        curr_nnz++;
        p++;q++;
      }
    }
    if (p < AI[i+1] - 1) {
      while (p < AI[i+1] - 1) {
        (*CJ)[curr_nnz] = AJ[p];
        (*CV)[curr_nnz] = AV[p];
        curr_nnz++; p++;
      }
    }
    if (q < BI[i+1] - 1) {
      while (q < BI[i+1] - 1) {
        (*CJ)[curr_nnz] = BJ[q];
        (*CV)[curr_nnz] = mu * BV[q];
        curr_nnz++; q++;
      }
    }
  }
  assert(total_nnz = curr_nnz);
  (*CI)[n] = total_nnz + 1;
}

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
	MKL_INT* CI = NULL;
	MKL_INT* CJ = NULL;
	double* CV = NULL;
	double mu = 0;
	FILE* fp;
	fp = fopen("./intervals.txt", "r");
	int N;
	fscanf(fp, "%d\n", &N);
	assert(N == num_tasks);
	double* intervals = new double[N+1];
	int* num_eigs = new int[N];
	for (int i = 0; i <= N; i++) {
		fscanf(fp, "%lf\n", &(intervals[i]));
	}
	for (int i = 0; i < N; i++) {
		fscanf(fp, "%d\n", &(num_eigs[i]));
	}
	printf("task[%d] working on [%.6lf, %.6lf] with %d eigs\n", task_id, intervals[task_id], intervals[task_id+1],
			num_eigs[task_id]);
	fclose(fp);
	mu = -(intervals[task_id] + intervals[task_id+1])/2;
	a_plus_mu_b(n, AI, AJ, AV, BI, BJ, BV, mu, &CI, &CJ, &CV); 

#if 0
  ViewCSR("A", n, AI, AJ, AV);
  ViewCSR("B", n, BI, BJ, BV);
  ViewCSR("C", n, CI, CJ, CV);
#endif

  if (error == 0) {
    TraceMin1(n, num_eigs[task_id], CI, CJ, CV, BI, BJ, BV, Y, S);
  }
	
	printf("Final eigenvalues: ");
	for (int j = 0; j < num_eigs[task_id]; ++j) {
		printf("%12.10lf ", S[j] - mu);
	}
	printf("\n");

	MPI_Finalize();
  
  if (AI != NULL) delete [] AI;
  if (AJ != NULL) delete [] AJ;
  if (AV != NULL) delete [] AV;
  if (BI != NULL) delete [] BI;
  if (BJ != NULL) delete [] BJ;
  if (BV != NULL) delete [] BV;
  if (Y  != NULL) delete [] Y;
  if (S  != NULL) delete [] S;
  if (CI != NULL) delete [] CI;
  if (CJ != NULL) delete [] CJ;
  if (CV != NULL) delete [] CV;

  return 0;
}
