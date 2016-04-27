#include <iostream>
#include <cassert>
#include "row_compressed_matrix.hpp"
#include "mmio.h"

using namespace std;

/* PARDISO prototype. */
extern "C" void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
extern "C" void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
                  double *, int    *,    int *, int *,   int *, int *,
                     int *, double *, double *, int *, double *);
extern "C" void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
extern "C" void pardiso_chkvec     (int *, int *, double *, int *);
extern "C" void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                           double *, int *);

void qsort(int* ia, int* ja, double* a, int lower, int upper) {
  // pivot = lower;
  if (lower >= upper) {return;}
  int midpos = lower;
  int ti;
  int tj;
  double ta;
  for (int i = lower + 1; i < upper; i++) {
    if (ia[i] < ia[midpos] || (ia[i] == ia[midpos] && ja[i] < ja[midpos])) {
      ti = ia[i]; tj = ja[i]; ta = a[i];
      ia[i] = ia[midpos+1]; ja[i] = ja[midpos+1]; a[i] = a[midpos+1];
      ia[midpos+1] = ia[midpos]; ja[midpos+1] = ja[midpos]; a[midpos+1] = a[midpos];
      ia[midpos] = ti; ja[midpos] = tj; a[midpos] = ta;
      midpos++;
    }
  }
  qsort(ia, ja, a, lower, midpos);
  qsort(ia, ja, a, midpos+1, upper);
}

RowCompressedMatrix::RowCompressedMatrix(const char* file_name) {
  FILE *fp;
  MM_typecode matcode;
  fp = fopen(file_name, "r");
  mm_read_banner(fp, &matcode);
  int M, N, NNZ;
  mm_read_mtx_crd_size(fp, &M, &N, &NNZ);
  assert(M == N);
  _num_rows = N;
  _num_nnz = NNZ;
  _allocate_memory();
  int* dummy_ia = new int[_num_nnz];
  //mtx stores the lower, pardiso needs the upper
  for (int i = 0; i < NNZ; i++) {
    fscanf(fp, "%d %d %lf\n", &(_ja[i]), &(dummy_ia[i]), &(_a[i]));
    dummy_ia[i]--;
    _ja[i]--;
  }
  bool sorted = true;
  for (int i = 1; i < NNZ; i++) {
    if (dummy_ia[i] < dummy_ia[i-1]) {
      sorted = false;
    } else if (dummy_ia[i] == dummy_ia[i-1] && _ja[i] < _ja[i-1]) {
      sorted = false;
    }
  }
  if (!sorted) {
    std::cerr << "MTX matrix is not sorted, please sort" << std::endl;
    exit(-1);
  }
  //qsort(dummy_ia, _ja, _a, 0, _num_nnz);
  /*
  for (int i = 0; i < NNZ; i++) {
    printf("%d : <%d, %d> = %.5lf\n", i, dummy_ia[i], _ja[i], _a[i]);
  }
  */
  int l = 0;
  int curr_count = 0;
  _ia[0] = 0;
  for (int i = 0; i < _num_rows; i++) {
    while (dummy_ia[l] <= i && l < _num_nnz) {
      l++;
    }
    _ia[i+1] = l;
  }
  //dump();
  delete[] dummy_ia;
}

RowCompressedMatrix a_plus_mu_b(const RowCompressedMatrix& A,
                                double mu,
                                const RowCompressedMatrix& B) {

  assert(A._num_rows == B._num_rows);

  //std::cout << "a_plus_mu_b" << std::endl;

  int p,q;
  int total_nnz = 0;
  for (int i = 0; i < A._num_rows; i++) {
    p = A._ia[i];
    q = B._ia[i];
    while (p < A._ia[i+1] && q < B._ia[i+1]) {
      if (A._ja[p] < B._ja[q]) {
        total_nnz++;
	p++;
      } else if (A._ja[p] > B._ja[q]) {
        total_nnz++;
	q++;
      } else {
        total_nnz++;
	p++;q++;
      }
    }
    if (p < A._ia[i+1]) {
      while (p < A._ia[i+1]) {total_nnz++; p++;}
    }
    if (q < B._ia[i+1]) {
      while (q < B._ia[i+1]) {total_nnz++; q++;}
    }
  }
  int curr_nnz = 0;
  RowCompressedMatrix ret(A._num_rows, total_nnz);
  for (int i = 0; i < A._num_rows; i++) {
    ret._ia[i] = curr_nnz;
    p = A._ia[i];
    q = B._ia[i];
    while (p < A._ia[i+1] && q < B._ia[i+1]) {
      if (A._ja[p] < B._ja[q]) {
        ret._ja[curr_nnz] = A._ja[p];
	ret._a[curr_nnz] = A._a[p];
        curr_nnz++;
	p++;
      } else if (A._ja[p] > B._ja[q]) {
        ret._ja[curr_nnz] = B._ja[q];
	ret._a[curr_nnz] = mu * B._a[q];
        curr_nnz++;
	q++;
      } else {
        ret._ja[curr_nnz] = A._ja[p];
	ret._a[curr_nnz] = A._a[p] + mu * B._a[q];
        curr_nnz++;
	p++;q++;
      }
    }
    if (p < A._ia[i+1]) {
      while (p < A._ia[i+1]) {
        ret._ja[curr_nnz] = A._ja[p];
	ret._a[curr_nnz] = A._a[p];
        curr_nnz++; p++;
      }
    }
    if (q < B._ia[i+1]) {
      while (q < B._ia[i+1]) {
        ret._ja[curr_nnz] = B._ja[q];
	ret._a[curr_nnz] = mu * B._a[q];
        curr_nnz++; q++;
      }
    }
  }
  assert(total_nnz = curr_nnz);
  ret._ia[ret._num_rows] = total_nnz;
  return ret; 
}

void RowCompressedMatrix::count_eigen(int& pos_eigen, int& neg_eigen) {
  //std::cout << "count_eigen" << std::endl; 
  bool index_change_needed = _c_index_style;
  if (index_change_needed) {
    _to_fortran_index();
  }

  int mtype = -2; // Real symmetric matrix
  void *pt[64]; // Internal solver memory pointer
  int iparm[64]; // Pardiso control parameters, int
  double dparm[64]; // double
  int maxfct, mnum, phase, error, msglvl, solver;
  int num_procs;
  int nrhs = 0;
  int idum = 0;
  double ddum = 0.0;

  error = 0;
  solver = 0; // sparse direct solver
  pardisoinit(pt, &mtype, &solver, iparm, dparm, &error);

  if (error != 0) {
    if (error == -10 )
      printf("No license file found \n");
    if (error == -11 )
      printf("License is expired \n");
    if (error == -12 )
      printf("Wrong username or hostname \n");
    exit(1);
  } else {
    //printf("[PARDISO]: License check was successful ... \n");
  }

  char* var = getenv("OMP_NUM_THREADS");
  if(var != NULL) {
      sscanf( var, "%d", &num_procs );
  }
  else {
    printf("Set environment OMP_NUM_THREADS to 1");
    exit(1);
  }
  iparm[2]  = num_procs;

  maxfct = 1;
  mnum = 1;
  msglvl = 0; // for now
  error = 0;
/*
  pardiso_chkmatrix  (&mtype, &_num_rows, _a, _ia, _ja, &error);
  if (error != 0) {
    printf("\nERROR in consistency of matrix: %d", error);
    exit(1);
  }
*/
  phase = 12; // analyze and factorization

  pardiso(pt, &maxfct, &mnum, &mtype, &phase,
          &_num_rows, _a, _ia, _ja, &idum, &nrhs,
          iparm, &msglvl, &ddum, &ddum, &error, dparm);
   
  if (error != 0) {
    printf("\nERROR during numerical factorization: %d", error);
    exit(2);
  }

  //printf("\n Positive Eigen Values = %d\n", iparm[21]);
  //printf("\n Negative Eigen Values = %d\n", iparm[22]);

  pos_eigen = iparm[21];
  neg_eigen = iparm[22];

  phase = -1; // release internal memory
  pardiso(pt, &maxfct, &mnum, &mtype, &phase,
          &_num_rows, _a, _ia, _ja, &idum, &nrhs,
          iparm, &msglvl, &ddum, &ddum, &error, dparm);

  if (index_change_needed) {
    _to_c_index();
  }
}

void RowCompressedMatrix::_to_fortran_index() {
  if (!_c_index_style) {return;}
  _c_index_style = false;
  for (int i = 0; i <= _num_rows; i++) {_ia[i]++;}
  for (int i = 0; i < _num_nnz; i++) {_ja[i]++;}
}

void RowCompressedMatrix::_to_c_index() {
  if (_c_index_style) {return;}
  _c_index_style = true;
  for (int i = 0; i <= _num_rows; i++) {_ia[i]--;}
  for (int i = 0; i < _num_nnz; i++) {_ja[i]--;}
}

void RowCompressedMatrix::dump() const {
  std::cout << "Dump: " << std::endl;
  std::cout << "ia = {";
  for (int i = 0; i <= _num_rows; i++) {
    std::cout << _ia[i] << " ";
  }
  std::cout << "}" << std::endl << "ja = {";
  for (int i = 0; i < _num_nnz; i++) {
    std::cout << _ja[i] << " ";
  }
  std::cout << "}" << std::endl << "a = {";
  for (int i = 0; i < _num_nnz; i++) {
    std::cout << _a[i] << " ";
  }
  std::cout << "}" << std::endl;
}
/*
int main( void ) 
{
    int    n = 8;
    int    ia[ 9] = { 0, 4, 7, 9, 11, 14, 16, 17, 18 };
    int    ja[18] = { 0,    2,       5, 6,
                         1, 2,    4,
                            2,             7,
                               3,       6,
                                  4, 5, 6,
                                     5,    7,
                                        6,
                                           7 };
    double  a[18] = { 7.0,      1.0,           2.0, 7.0,
                          -4.0, 8.0,           2.0,
                                1.0,                     5.0,
                                     7.0,           9.0,
                                          5.0, 1.0, 5.0,
                                               0.0,      5.0,
                                                   11.0,
                                                         5.0 };

    int      nnz = ia[n];
    int      mtype = -2;       

    / RHS and solution vectors. /
    double   b[8], x[8];
    int      nrhs = 1;          / Number of right hand sides. /

    / Internal solver memory pointer pt,                  /
    / 32-bit: int pt[64]; 64-bit: long int pt[64]         /
    / or void pt[64] should be OK on both architectures  / 
    void    pt[64]; 

    / Pardiso control parameters. /
    int      iparm[64];
    double   dparm[64];
    int      maxfct, mnum, phase, error, msglvl, solver;

    / Number of processors. /
    int      num_procs;

    / Auxiliary variables. /
    char    var;
    int      i;

    double   ddum;              / Double dummy /
    int      idum;              / Integer dummy. /

   
/ -------------------------------------------------------------------- /
/ ..  Setup Pardiso control parameters.                                /
/ -------------------------------------------------------------------- /

    error = 0;
    solver = 0; / use sparse direct solver /
    pardisoinit (pt,  &mtype, &solver, iparm, dparm, &error); 

    if (error != 0) 
    {
        if (error == -10 )
           printf("No license file found \n");
        if (error == -11 )
           printf("License is expired \n");
        if (error == -12 )
           printf("Wrong username or hostname \n");
         return 1; 
    }
    else
        printf("[PARDISO]: License check was successful ... \n");
    
    / Numbers of processors, value of OMP_NUM_THREADS /
    var = getenv("OMP_NUM_THREADS");
    if(var != NULL)
        sscanf( var, "%d", &num_procs );
    else {
        printf("Set environment OMP_NUM_THREADS to 1");
        exit(1);
    }
    iparm[2]  = num_procs;

    maxfct = 1;		/ Maximum number of numerical factorizations.  /
    mnum   = 1;         / Which factorization to use. /
    
    msglvl = 1;         / Print statistical information  /
    error  = 0;         / Initialize error flag /

/ -------------------------------------------------------------------- /
/ ..  Convert matrix from 0-based C-notation to Fortran 1-based        /
/     notation.                                                        /
/ -------------------------------------------------------------------- /
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }

    / Set right hand side to one. /
    for (i = 0; i < n; i++) {
        b[i] = i;
    }

/ -------------------------------------------------------------------- /
/  .. pardiso_chk_matrix(...)                                          /
/     Checks the consistency of the given matrix.                      /
/     Use  functionality only for debugging purposes               /
/ -------------------------------------------------------------------- /
    
    pardiso_chkmatrix  (&mtype, &n, a, ia, ja, &error);
    if (error != 0) {
        printf("\nERROR in consistency of matrix: %d", error);
        exit(1);
    }

/ -------------------------------------------------------------------- /
/ ..  pardiso_chkvec(...)                                              /
/     Checks the given vectors for infinite and NaN values             /
/     Input parameters (see PARDISO user manual for a description):    /
/     Use  functionality only for debugging purposes               /
/ -------------------------------------------------------------------- /

    pardiso_chkvec (&n, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR  in right hand side: %d", error);
        exit(1);
    }

/ -------------------------------------------------------------------- /
/ .. pardiso_printstats(...)                                           /
/    prints information on the matrix to STDOUT.                       /
/    Use  functionality only for debugging purposes                /
/ -------------------------------------------------------------------- /

    pardiso_printstats (&mtype, &n, a, ia, ja, &nrhs, b, &error);
    if (error != 0) {
        printf("\nERROR right hand side: %d", error);
        exit(1);
    }
 
/ -------------------------------------------------------------------- /
/ ..  Reordering and Symbolic Factorization.  This step also allocates /
/     all memory that is necessary for the factorization.              /
/ -------------------------------------------------------------------- /
    phase = 11; 

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
	     &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error, dparm);
    
    if (error != 0) {
        printf("\nERROR during symbolic factorization: %d", error);
        exit(1);
    }
    printf("\n Positive Eigen Values = %d\n", iparm[21]);
    printf("\n Negative Eigen Values = %d\n", iparm[22]);
    printf("\nReordering completed ... ");
    printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
    printf("\nNumber of factorization MFLOPS = %d", iparm[18]);
   
/ -------------------------------------------------------------------- /
/ ..  Numerical factorization.                                         /
/ -------------------------------------------------------------------- /    
    phase = 22;
    iparm[32] = 1; / compute determinant /

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
    }
    printf("\n Positive Eigen Values = %d\n", iparm[21]);
    printf("\n Negative Eigen Values = %d\n", iparm[22]);
    printf("\nFactorization completed ...\n ");

/ -------------------------------------------------------------------- /    
/ ..  Back substitution and iterative refinement.                      /
/ -------------------------------------------------------------------- /    
    phase = 33;

    iparm[7] = 1;       / Max numbers of iterative refinement steps. /
   
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);
   
    if (error != 0) {
        printf("\nERROR during solution: %d", error);
        exit(3);
    }

    printf("\nSolve completed ... ");
    printf("\nThe solution of the system is: ");
    for (i = 0; i < n; i++) {
        printf("\n x [%d] = % f", i, x[i] );
    }
    printf ("\n");

/ -------------------------------------------------------------------- /    
/ ..  Convert matrix back to 0-based C-notation.                       /
/ -------------------------------------------------------------------- / 
    for (i = 0; i < n+1; i++) {
        ia[i] -= 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] -= 1;
    }

/ -------------------------------------------------------------------- /    
/ ..  Termination and release of memory.                               /
/ -------------------------------------------------------------------- /    
    phase = -1;                 / Release internal memory. /
    
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, &ddum, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);

    return 0;
}
*/
